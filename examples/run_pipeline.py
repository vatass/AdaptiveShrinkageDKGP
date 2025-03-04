"""
Complete pipeline for biomarker prediction using Deep Kernel Regression with Adaptive Shrinkage Estimation
"""
import os
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List

from adaptive_shrinkage_dkgp.models.population_dkgp import PopulationDKGP
from adaptive_shrinkage_dkgp.models.ss_dkgp import SubjectSpecificDKGP
from adaptive_shrinkage_dkgp.models.adaptive_shrinkage import AdaptiveShrinkage

def load_and_preprocess_data(
    data_path: str,
    train_size: float = 0.6,
    val_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Dict[str, torch.Tensor], List[str], List[str], List[str]]:
    """Load and preprocess the biomarker data.
    
    Args:
        data_path: Path to the data CSV file
        train_size: Proportion of data to use for training
        val_size: Proportion of data to use for validation
        random_state: Random seed
        
    Returns:
        Dictionary containing data tensors and lists of subject IDs
    """
    print("Loading and preprocessing data...")
    
    # Load data
    data = pd.read_csv(data_path)
    
    # Get unique subject IDs
    subject_ids = data['PTID'].unique()
    
    # First split into train+val and test
    train_val_size = train_size + val_size
    train_val_ids, test_ids = train_test_split(
        subject_ids,
        train_size=train_val_size,
        random_state=random_state
    )
    
    # Then split train+val into train and val
    train_ids, val_ids = train_test_split(
        train_val_ids,
        train_size=train_size/train_val_size,
        random_state=random_state
    )
    
    print(f'Train IDs: {len(train_ids)}')
    print(f'Val IDs: {len(val_ids)}')
    print(f'Test IDs: {len(test_ids)}')
    
    # Extract data for each set
    def extract_data(ids):
        subset = data[data['PTID'].isin(ids)]
        x = torch.FloatTensor(subset['X'].values)
        y = torch.FloatTensor(subset['Y'].values)
        return x, y, subset['PTID'].tolist()
    
    train_x, train_y, train_ids = extract_data(train_ids)
    val_x, val_y, val_ids = extract_data(val_ids)
    test_x, test_y, test_ids = extract_data(test_ids)
    
    # Squeeze target dimensions
    train_y = train_y.squeeze()
    val_y = val_y.squeeze()
    test_y = test_y.squeeze()
    
    # Move to GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        train_x = train_x.to(device)
        train_y = train_y.to(device)
        val_x = val_x.to(device)
        val_y = val_y.to(device)
        test_x = test_x.to(device)
        test_y = test_y.to(device)
    
    print(f'Train Data: {train_x.shape}')
    print(f'Train Targets: {train_y.shape}')
    print(f'Val Data: {val_x.shape}')
    print(f'Val Targets: {val_y.shape}')
    print(f'Test Data: {test_x.shape}')
    print(f'Test Targets: {test_y.shape}')
    
    sys.exit(0)


    return {
        'train_x': train_x,
        'train_y': train_y,
        'val_x': val_x,
        'val_y': val_y,
        'test_x': test_x,
        'test_y': test_y
    }, train_ids, val_ids, test_ids

def train_population_model(
    data: Dict[str, torch.Tensor],
    input_dim: int,
    latent_dim: int,
    model_save_path: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> PopulationDKGP:
    """Train the population DKGP model.
    
    Args:
        data: Dictionary containing data tensors
        input_dim: Input dimension
        latent_dim: Latent dimension
        model_save_path: Path to save the model
        device: Device to run the model on
        
    Returns:
        Trained population DKGP model
    """
    print("Training population DKGP model...")
    
    model = PopulationDKGP(
        train_x=data['train_x'],
        train_y=data['train_y'],
        input_dim=input_dim,
        latent_dim=latent_dim,
        device=device
    )
    
    history = model.fit(
        train_x=data['train_x'],
        train_y=data['train_y'],
        val_x=data['val_x'],
        val_y=data['val_y'],
        verbose=True
    )
    
    # Save model
    model.save_model(model_save_path)
    
    return model

def train_adaptive_shrinkage(
    pop_model: PopulationDKGP,
    data: Dict[str, torch.Tensor],
    model_save_path: str
) -> AdaptiveShrinkage:
    """Train the adaptive shrinkage estimator.
    
    Args:
        pop_model: Trained population model
        data: Dictionary containing data tensors
        model_save_path: Path to save the model
        
    Returns:
        Trained adaptive shrinkage model
    """
    print("Training adaptive shrinkage estimator...")
    
    # Get population model predictions
    pop_mean, pop_std = pop_model.predict(data['val_x'])
    
    # Initialize and train subject-specific models for validation subjects
    ss_means = []
    ss_stds = []
    n_obs_list = []
    
    for i in range(len(data['val_x'])):
        # Get subject data up to current observation
        x_sub = data['val_x'][:i+1]
        y_sub = data['val_y'][:i+1]
        
        # Train subject-specific model
        ss_model = SubjectSpecificDKGP(
            train_x=x_sub,
            train_y=y_sub,
            input_dim=pop_model.input_dim,
            latent_dim=pop_model.latent_dim,
            population_params=pop_model.get_deep_params()
        )
        ss_model.fit(x_sub, y_sub, verbose=False)
        
        # Get predictions
        mean, std = ss_model.predict(data['val_x'][i:i+1])
        ss_means.append(mean)
        ss_stds.append(std)
        n_obs_list.append(i + 1)
    
    ss_means = torch.cat(ss_means)
    ss_stds = torch.cat(ss_stds)
    n_obs = torch.tensor(n_obs_list)
    
    # Train adaptive shrinkage
    shrinkage = AdaptiveShrinkage()
    shrinkage.fit(
        pop_pred=pop_mean,
        ss_pred=ss_means,
        true_values=data['val_y'],
        n_obs=n_obs,
        pop_std=pop_std,
        ss_std=ss_stds
    )
    
    # Save model
    shrinkage.save_model(model_save_path)
    
    return shrinkage

def evaluate_personalization(
    pop_model: PopulationDKGP,
    shrinkage: AdaptiveShrinkage,
    data: Dict[str, torch.Tensor],
    test_ids: List[str],
    max_history: int = 10
) -> pd.DataFrame:
    """Evaluate personalization on test subjects.
    
    Args:
        pop_model: Trained population model
        shrinkage: Trained adaptive shrinkage model
        data: Dictionary containing data tensors
        test_ids: List of test subject IDs
        max_history: Maximum number of history points to consider
        
    Returns:
        DataFrame with evaluation results
    """
    print("Evaluating personalization on test subjects...")
    
    results = {
        'subject_id': [],
        'n_observations': [],
        'mse_population': [],
        'mse_subject_specific': [],
        'mse_combined': []
    }
    
    for subject_id in test_ids:
        print(f"Processing subject {subject_id}...")
        
        # Get subject data
        subject_mask = data['test_x'][:, 0] == subject_id  # Assuming first column is subject ID
        x_subject = data['test_x'][subject_mask]
        y_subject = data['test_y'][subject_mask]
        
        # Get population predictions
        pop_mean, pop_std = pop_model.predict(x_subject)
        
        for n_obs in range(1, min(len(x_subject), max_history + 1)):
            # Train subject-specific model on history
            x_history = x_subject[:n_obs]
            y_history = y_subject[:n_obs]
            
            ss_model = SubjectSpecificDKGP(
                train_x=x_history,
                train_y=y_history,
                input_dim=pop_model.input_dim,
                latent_dim=pop_model.latent_dim,
                population_params=pop_model.get_deep_params()
            )
            ss_model.fit(x_history, y_history, verbose=False)
            
            # Get predictions
            ss_mean, ss_std = ss_model.predict(x_subject)
            
            # Get combined predictions
            combined_pred = shrinkage.predict(
                pop_pred=pop_mean,
                ss_pred=ss_mean,
                n_obs=torch.tensor([n_obs] * len(x_subject)),
                pop_std=pop_std,
                ss_std=ss_std
            )
            
            # Calculate MSE
            mse_pop = ((pop_mean - y_subject) ** 2).mean().item()
            mse_ss = ((ss_mean - y_subject) ** 2).mean().item()
            mse_combined = ((combined_pred - y_subject) ** 2).mean().item()
            
            # Store results
            results['subject_id'].append(subject_id)
            results['n_observations'].append(n_obs)
            results['mse_population'].append(mse_pop)
            results['mse_subject_specific'].append(mse_ss)
            results['mse_combined'].append(mse_combined)
    
    return pd.DataFrame(results)

def train_population_dkgp(data_path: str, model_save_path: str, weights_save_path: str):
    # Load data
    print("Loading data...")
    data = pd.read_csv(data_path)
    input_dim = len(eval(data['X'].iloc[0]))

    # Initialize model
    print("Initializing model...")
    model = PopulationDKGP(
        input_dim=input_dim,
        hidden_dim=64,
        feature_dim=32
    )

    # Train model
    print("Training model...")
    metrics = model.fit(
        train_data=data,
        roi_idx=13,
        num_epochs=500,
        lr=0.01844
    )

    # Print training results
    print("\nTraining Results:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")

    # Save model
    print("\nSaving model and weights...")
    model.save_model(model_save_path, weights_save_path)

def main():
    # Parameters
    data_path = "data/biomarker_data.csv"  # Path to your data file
    model_dir = "models"
    results_dir = "results"
    
    # Create output directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load and preprocess data
    data, train_ids, val_ids, test_ids = load_and_preprocess_data(data_path)
    
    # Set dimensions based on data
    input_dim = data['train_x'].shape[1]
    latent_dim = input_dim // 2
    
    # Train population model
    pop_model = train_population_model(
        data,
        input_dim,
        latent_dim,
        os.path.join(model_dir, "population_dkgp.pt")
    )
    
    # Train adaptive shrinkage
    shrinkage = train_adaptive_shrinkage(
        pop_model,
        data,
        os.path.join(model_dir, "adaptive_shrinkage.json")
    )
    
    # Evaluate personalization
    results = evaluate_personalization(
        pop_model,
        shrinkage,
        data,
        test_ids
    )
    
    # Save results
    results.to_csv(os.path.join(results_dir, "personalization_results.csv"), index=False)
    
    # Print summary
    print("\nResults Summary:")
    print("Mean MSE by number of observations:")
    summary = results.groupby('n_observations').agg({
        'mse_population': ['mean', 'std'],
        'mse_subject_specific': ['mean', 'std'],
        'mse_combined': ['mean', 'std']
    })
    print(summary)

if __name__ == "__main__":
    train_population_dkgp(
        data_path='data/biomarker_data_processed.csv',
        model_save_path='models/population_dkgp.pt',
        weights_save_path='models/feature_extractor_weights.pkl'
    ) 