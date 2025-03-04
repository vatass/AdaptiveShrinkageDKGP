import pandas as pd
import numpy as np
import sys
import torch
import gpytorch
import pickle
from .single_task_deep_kernel import SingleTaskDeepKernel
import argparse
import matplotlib.pyplot as plt 
import seaborn as sns 
from tqdm import tqdm 
import time 
import json
import math 
sns.set_style("white", {'axes.grid' : False})
from pathlib import Path
from .base import BaseDeepKernel


class PopulationDKGP(BaseDeepKernel):
    def __init__(self, input_dim, hidden_dim=64, feature_dim=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(PopulationDKGP, self).__init__(
            train_x=None,  # Placeholder, should be set during training
            train_y=None,  # Placeholder, should be set during training
            likelihood=gpytorch.likelihoods.GaussianLikelihood(),
            input_dim=input_dim,
            latent_dim=feature_dim
        )
        self.device = device
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        
        # Initialize model components
        self.likelihood = self.likelihood.to(device)
        self.model = self.model.to(device)

    def process_data(self, data, roi_idx=-1):
        """Process input data and move to device."""
        X = torch.tensor(np.stack(data['X'].values), dtype=torch.float32)
        y = torch.tensor(data['Y'].values, dtype=torch.float32)
        
        if roi_idx != -1:
            y = y[:, roi_idx]
        y = y.squeeze()
        
        return X.to(self.device), y.to(self.device)

    def fit(self, train_data, roi_idx=-1, num_epochs=500, lr=0.01844, weight_decay=0.01):
        """Train the population model."""
        X_train, y_train = self.process_data(train_data, roi_train)
        
        # Set to training mode
        self.model.train()
        self.likelihood.train()
        
        # Optimizer
        optimizer = torch.optim.Adam([
            {'params': self.model.feature_extractor.parameters(), 'lr': lr},
            {'params': self.model.covar_module.parameters(), 'lr': lr},
            {'params': self.model.mean_module.parameters(), 'lr': lr},
            {'params': self.likelihood.parameters(), 'lr': lr}
        ], weight_decay=weight_decay)
        
        # Loss function
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        # Training loop
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            output = self.model(X_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 50 == 0:
                print(f'Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}')
        
        return self.evaluate(X_train, y_train)

    def evaluate(self, X, y):
        """Evaluate model performance."""
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f_preds = self.model(X)
            y_preds = self.likelihood(f_preds)
            mean = y_preds.mean
            lower, upper = y_preds.confidence_region()
        
        # Move to CPU for numpy calculations
        mean_np = mean.cpu().numpy()
        y_np = y.cpu().numpy()
        lower_np = lower.cpu().numpy()
        upper_np = upper.cpu().numpy()
        
        # Calculate metrics
        mae_result, ae = mae(y_np, mean_np)
        mse_result, rmse_result, se = mse(y_np, mean_np)
        r2_result = R2(y_np, mean_np)
        coverage, interval_width, mean_coverage, mean_interval_width = calc_coverage(
            predictions=mean_np,
            groundtruth=y_np,
            intervals=[lower_np, upper_np]
        )
        
        metrics = {
            'mae': np.mean(mae_result),
            'mse': mse_result,
            'rmse': rmse_result,
            'r2': r2_result,
            'coverage': np.mean(coverage),
            'interval_width': mean_interval_width
        }
        
        return metrics

    def predict(self, test_data, roi_idx=-1):
        """Make predictions with uncertainty."""
        X_test, _ = self.process_data(test_data, roi_test)
        
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(X_test))
            mean = observed_pred.mean
            lower, upper = observed_pred.confidence_region()
        
        # Move predictions to CPU
        return mean.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy()

    def save_model(self, model_path, weights_path):
        """Save model and feature extractor weights."""
        # Save full model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'likelihood_state_dict': self.likelihood.state_dict()
        }, model_path)
        
        # Save feature extractor weights separately
        feature_extractor_weights = self.model.feature_extractor.state_dict()
        with open(weights_path, 'wb') as f:
            pickle.dump(feature_extractor_weights, f)

    def load_model(self, model_path):
        """Load model from disk."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.likelihood.load_state_dict(checkpoint['likelihood_state_dict'])

def main():
    parser = argparse.ArgumentParser(description='Train Population DKGP Model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--model_save_path', type=str, required=True, help='Path to save model')
    parser.add_argument('--weights_save_path', type=str, required=True, help='Path to save feature extractor weights')
    parser.add_argument('--roi_idx', type=int, default=-1, help='ROI index to predict (-1 for all)')
    parser.add_argument('--learning_rate', type=float, default=0.01844, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--feature_dim', type=int, default=32, help='Feature dimension size')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    data = pd.read_csv(args.data_path)
    input_dim = len(eval(data['X'].iloc[0]))
    
    # Initialize and train model
    print("Initializing model...")
    model = PopulationDKGP(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        feature_dim=args.feature_dim
    )

    print("Training model...")
    t0 = time.time()
    metrics = model.fit(
        train_data=data,
        roi_idx=args.roi_idx,
        num_epochs=args.epochs,
        lr=args.learning_rate
    )
    
    # Print metrics
    print("\nTraining Results:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Save model
    print("\nSaving model and weights...")
    model.save_model(args.model_save_path, args.weights_save_path)
    
    print(f"\nTotal time elapsed: {time.time() - t0:.2f} seconds")

if __name__ == "__main__":
    main()
