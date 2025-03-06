import pandas as pd
import numpy as np
import sys
import torch
import gpytorch
import pickle
import argparse
import matplotlib.pyplot as plt 
import seaborn as sns 
from tqdm import tqdm 
import time 
import json
import math 
sns.set_style("white", {'axes.grid' : False})
from pathlib import Path
from .base import BaseDeepKernel, LargeFeatureExtractor
from typing import Dict


class SingleTaskDeepKernel(gpytorch.models.ExactGP): 
    def __init__(self, input_dim, train_x, train_y, likelihood, depth, dropout, activation, pretrained,latent_dim, feature_extractor, gphyper, kernel_choice='RBF', mean='CONSTANT'):
        super(SingleTaskDeepKernel, self).__init__(train_x, train_y, likelihood)
        self.likelihood = likelihood
        self.mean_module = gpytorch.means.LinearMean(input_size=latent_dim)
        # self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=latent_dim))

        self.pretrained = pretrained

        if not pretrained: 
            self.feature_extractor = LargeFeatureExtractor(datadim=input_dim, depth=depth, dr=dropout, activ=activation)
        else: 
            self.feature_extractor = feature_extractor
        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        if gphyper is not None: 
            self.initialize(**gphyper)


    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"


        if self.pretrained:
            projected_x = projected_x.detach()

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class PopulationDKGP(BaseDeepKernel):
    def __init__(self, train_x, train_y, input_dim, hidden_dim=64, feature_dim=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(PopulationDKGP, self).__init__(
            train_x=train_x,
            train_y=train_y,
            likelihood=gpytorch.likelihoods.GaussianLikelihood(),
            input_dim=input_dim,
            latent_dim=feature_dim
        )
        self.device = device
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        
        # Initialize model components
        self.likelihood = self.likelihood.to(device)
        
        # Initialize the model attribute
        self.model = SingleTaskDeepKernel(
            input_dim=input_dim,
            train_x=train_x,
            train_y=train_y,
            likelihood=self.likelihood,
            depth = [(train_x.shape[1], int(train_x.shape[1]/2) )], 
            dropout=0.1,  # Dropout 0.1
            activation='relu',  # Adjust activation as needed
            pretrained=False,  # Adjust based on your needs
            latent_dim=feature_dim,
            feature_extractor=None,  # Adjust based on your needs
            gphyper=None  # Adjust based on your needs
        ).to(device)

    def fit(self, train_x, train_y, num_epochs=500, lr=0.01844, weight_decay=0.01):
        """Train the population model."""

        # convert to tensor if not and move to gpu if not already
        train_x = torch.tensor(train_x).to(self.device)
        train_y = torch.tensor(train_y).to(self.device)

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
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 50 == 0:
                print(f'Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}')
        
        return self.evaluate(train_x, train_y)

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

        # print('test_data', test_data.shape)

        X_test = test_data
        
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(X_test))
            mean = observed_pred.mean
            lower, upper = observed_pred.confidence_region()
            variance = observed_pred.variance

        return mean.cpu().numpy(), lower.cpu().numpy(), upper.cpu().numpy(), variance.cpu().numpy()

    def save_model(self, model_path, weights_path):
        """Save model and feature extractor weights."""
        # Save full model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'likelihood_state_dict': self.likelihood.state_dict()
        }, model_path)
        
        # Save feature extractor weights separately
        feature_extractor_weights = {k: v for k, v in self.model.feature_extractor.state_dict().items()}
        with open(weights_path, 'wb') as f:
            pickle.dump(feature_extractor_weights, f)

    def load_model(self, model_path):
        """Load model from disk."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.likelihood.load_state_dict(checkpoint['likelihood_state_dict'])

    def get_deep_params(self):
        """Get deep kernel parameters including only feature extractor."""
        feature_extractor_params = {}
        for param_name, param in self.model.named_parameters():
            if param_name.startswith('feature_extractor'):
                feature_extractor_params[param_name] = param
        return feature_extractor_params


def train_population_model(
    data: Dict[str, torch.Tensor],
    input_dim: int,
    latent_dim: int,
    model_save_path: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> PopulationDKGP:
    """Train the population DKGP model."""
    
    print("Training population DKGP model...")
    
    # Δημιουργία του μοντέλου χωρίς τα περιττά ορίσματα
    model = PopulationDKGP(
        train_x=data['train_x'],
        train_y=data['train_y'],
        input_dim=input_dim,
        hidden_dim=64,  
        feature_dim=latent_dim,
        device=device
    ) 
    
    history = model.fit(
        train_x=data['train_x'],
        train_y=data['train_y'],
        num_epochs=500,  
        lr=0.01844 
    )
    
    # Αποθήκευση του μοντέλου
    model.save_model(model_save_path)
    
    return model

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred)), np.abs(y_true - y_pred)

def mse(y_true, y_pred):
    se = (y_true - y_pred) ** 2
    mse_value = np.mean(se)
    rmse_value = np.sqrt(mse_value)
    return mse_value, rmse_value, se

def R2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def calc_coverage(predictions, groundtruth, intervals):
    lower, upper = intervals
    coverage = (groundtruth >= lower) & (groundtruth <= upper)
    interval_width = upper - lower
    mean_coverage = np.mean(coverage)
    mean_interval_width = np.mean(interval_width)
    return coverage.astype(int), interval_width, mean_coverage, mean_interval_width


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
        train_x=data['train_x'],
        train_y=data['train_y'],
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
