"""
Complete pipeline for biomarker prediction using Adaptive Shrinkage Deep Kernel GP.
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
    biomarker_name: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Dict[str, torch.Tensor], List[str], List[str], List[str]]:
    """Load and preprocess the biomarker data.
    
    Args:
        data_path: Path to the data file
        biomarker_name: Name of the biomarker column
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation
        random_state: Random seed
        
    Returns:
        Dictionary containing data tensors and lists of subject IDs
    """
    # Load data
    data = pd.read_csv(data_path)
    
    # Get unique subject IDs
    subject_ids = data['PTID'].unique()
    
    # Split subjects into train, validation, and test
    train_val_ids, test_ids = train_test_split(
        subject_ids, test_size=test_size, random_state=random_state
    )
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=val_size, random_state=random_state
    )
    
    # Create data tensors
    def create_tensors(ids):
        subset = data[data['PTID'].isin(ids)]
        X = torch.FloatTensor(subset.drop([biomarker_name, 'PTID'], axis=1).values)
        y = torch.FloatTensor(subset[biomarker_name].values).reshape(-1, 1)
        return X, y
    
    train_x, train_y = create_tensors(train_ids)
    val_x, val_y = create_tensors(val_ids)
    test_x, test_y = create_tensors(test_ids)
    
    return {
        'train_x': train_x,
        'train_y': train_y,
        'val_x': val_x,
        'val_y': val_y,
        'test_x': test_x,
        'test_y': test_y
    }, list(train_ids), list(val_ids), list(test_ids)

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

def main():
    # Parameters
    data_path = "data/biomarker_data.csv"
    biomarker_name = "SPARE_AD"
    input_dim = 10  # Update with actual input dimension
    latent_dim = 5  # Update with actual latent dimension
    
    # Create output directory
    os.makedirs("models", exist_ok=True)
    
    # Load and preprocess data
    data, train_ids, val_ids, test_ids = load_and_preprocess_data(
        data_path, biomarker_name
    )
    
    # Train population model
    pop_model = train_population_model(
        data,
        input_dim,
        latent_dim,
        "models/population_dkgp.pt"
    )
    
    # Train adaptive shrinkage
    shrinkage = train_adaptive_shrinkage(
        pop_model,
        data,
        "models/adaptive_shrinkage.json"
    )
    
    # Evaluate personalization
    results = evaluate_personalization(
        pop_model,
        shrinkage,
        data,
        test_ids
    )
    
    # Save results
    results.to_csv("results/personalization_results.csv", index=False)
    
    # Print summary
    print("\nResults Summary:")
    print("Mean MSE by number of observations:")
    summary = results.groupby('n_observations').agg({
        'mse_population': 'mean',
        'mse_subject_specific': 'mean',
        'mse_combined': 'mean'
    })
    print(summary)

if __name__ == "__main__":
    main() 