import numpy as np
import pandas as pd
from models.adaptive_dkgp import AdaptiveDKGP
from sklearn.model_selection import train_test_split
import torch
from pathlib import Path

def load_data(data_path):
    """Load and preprocess the data."""
    data = pd.read_csv(data_path)
    
    # Convert string representation of arrays to numpy arrays
    data['X'] = data['X'].apply(lambda x: np.array(eval(x)))
    
    return data

def train_model(data, input_dim, hidden_dim=64, feature_dim=32, test_size=0.2):
    """Train the Adaptive DKGP model."""
    # Split data into train and test sets
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    
    # Create and train model
    model = AdaptiveDKGP(input_dim=input_dim, hidden_dim=hidden_dim, feature_dim=feature_dim)
    
    # Convert data to tensors
    X_train = torch.tensor(np.stack(train_data['X'].values), dtype=torch.float32)
    y_train = torch.tensor(train_data['Y'].values, dtype=torch.float32)
    
    print("Training model...")
    model.fit(X_train, y_train, num_epochs=100)
    
    return model, test_data

def evaluate_model(model, test_data):
    """Evaluate the model on test data."""
    X_test = torch.tensor(np.stack(test_data['X'].values), dtype=torch.float32)
    y_test = test_data['Y'].values
    
    # Get predictions
    mean, lower, upper = model.predict(X_test)
    
    # Calculate metrics
    mse = np.mean((mean.numpy() - y_test) ** 2)
    mae = np.mean(np.abs(mean.numpy() - y_test))
    
    # Calculate coverage of confidence intervals
    in_interval = np.logical_and(y_test >= lower.numpy(), y_test <= upper.numpy())
    coverage = np.mean(in_interval)
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"95% CI Coverage: {coverage:.4f}")
    
    return mse, mae, coverage

def main():
    # Create necessary directories
    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    # Load data
    print("Loading data...")
    data = load_data('data/biomarker_data_processed.csv')
    
    # Get input dimension from the first sample
    input_dim = len(data['X'].iloc[0])
    
    # Train model
    model, test_data = train_model(data, input_dim)
    
    # Evaluate model
    print("\nEvaluating model...")
    mse, mae, coverage = evaluate_model(model, test_data)
    
    # Save model
    print("\nSaving model...")
    model.save_model('models/trained_model.pt')
    
    # Save evaluation metrics
    metrics = pd.DataFrame({
        'metric': ['MSE', 'MAE', 'Coverage'],
        'value': [mse, mae, coverage]
    })
    metrics.to_csv('results/evaluation_metrics.csv', index=False)
    
    print("\nTraining completed! Model and metrics have been saved.")

if __name__ == "__main__":
    main() 