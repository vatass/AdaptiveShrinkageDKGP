"""
Complete pipeline for biomarker prediction using Deep Kernel Regression with Adaptive Shrinkage Estimation
"""
import os
import sys
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from src.adaptive_shrinkage_dkgp.models.population_dkgp import PopulationDKGP
from src.adaptive_shrinkage_dkgp.models.ss_dkgp import SubjectSpecificDKGP
from src.adaptive_shrinkage_dkgp.models.adaptive_shrinkage import AdaptiveShrinkage
from src.adaptive_shrinkage_dkgp.models.alpha_models import optimize_alpha_with_subject_simple

def load_and_preprocess_data(
    data_path: str,
    train_size: float = 0.6,
    val_size: float = 0.1,
    random_state: int = 42, 
    target: str = 'ROI_48'
) -> Tuple[Dict[str, torch.Tensor], List[str], List[str], List[str], Dict[str, Tuple[int, int]], Dict[str, Tuple[int, int]], Dict[str, Tuple[int, int]]]:
    """Load and preprocess the biomarker data.
    
    Args:
        data_path: Path to the data CSV file
        train_size: Proportion of data to use for training the population model 
        val_size: Proportion of data to use for training the adaptive shrinkage estimator
        random_state: Random seed
        
    Returns:
        Dictionary containing data tensors and lists of subject IDs
    """
    print("Loading and preprocessing data...")
    
    # Load data
    data = pd.read_csv(data_path)

    subjects = data['PTID'].unique()

    # print('Subjects', subjects)

    rois = [f"ROI_{i}" for i in range(1, 146)]

    covariates = [f"Covariate{i}" for i in range(1, 6)]  # Αντικαταστήστε το n με τον αριθμό των συνδιακυμάνσεων

    features = rois + covariates

    # print(features)
    
    # Convert the Data to the X,Y, PTID format that is ideal for our problem. 
    # X should be the ROIs, the Covariates at the first acquisition of the subejct and then the Time the Y should be the ROI Value at time T 
    samples = {'PTID': [], 'X': [], 'Y': []}

    for i, subject_id in enumerate(subjects):

        subject = data[data['PTID']==subject_id]

        for k in range(0, subject.shape[0]):
            samples['PTID'].append(subject_id)
    
            # Baseline Input 
            x = subject[features].iloc[0].to_list()

            delta = subject['Time'].iloc[k]

            x.extend([delta])

            t = subject[target].iloc[k] #.to_list()

            samples['X'].append(x)
            samples['Y'].append(t.tolist())

    assert len(samples['PTID']) == len(samples['X'])
    assert len(samples['X']) == len(samples['Y'])

    samples_df = pd.DataFrame(data=samples)

    subject_ids = samples_df['PTID'].unique()

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
    
    # print(f'Train IDs: {len(train_ids)}')
    # print(f'Val IDs: {len(val_ids)}')
    # print(f'Test IDs: {len(test_ids)}')
    
    # Extract data for each set
    def extract_data(ids):
        subset = samples_df[samples_df['PTID'].isin(ids)]

        print(subset)

        print(type(subset['X'].values))

        x_numeric = np.array([np.array(xi, dtype=np.float32) for xi in subset['X'].values])

        x = torch.FloatTensor(x_numeric)
        y = torch.FloatTensor(subset['Y'].values)
        ptids = subset['PTID'].tolist()
        return x, y, ptids
    
    train_x, train_y, train_ids = extract_data(train_ids)
    val_x, val_y, val_ids = extract_data(val_ids)
    test_x, test_y, test_ids = extract_data(test_ids)
    
    # Track start and end indices for each training subject
    train_subject_indices = {}
    for subject_id in train_ids:
        indices = [i for i, ptid in enumerate(train_ids) if ptid == subject_id]
        train_subject_indices[subject_id] = (indices[0], indices[-1])

    # Track start and end indices for each test subject
    test_subject_indices = {}
    for subject_id in test_ids:
        indices = [i for i, ptid in enumerate(test_ids) if ptid == subject_id]
        test_subject_indices[subject_id] = (indices[0], indices[-1])

    # Track start and end indices for each validation subject
    val_subject_indices = {}
    for subject_id in val_ids:
        indices = [i for i, ptid in enumerate(val_ids) if ptid == subject_id]
        val_subject_indices[subject_id] = (indices[0], indices[-1])
    
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

    print("Data loading and preprocessing completed.")

    return {
        'train_x': train_x,
        'train_y': train_y,
        'val_x': val_x,
        'val_y': val_y,
        'test_x': test_x,
        'test_y': test_y,
        'val_ptids': val_ids
    }, train_ids, val_ids, test_ids, train_subject_indices, val_subject_indices, test_subject_indices

def train_population_model(
    data: Dict[str, torch.Tensor],
    input_dim: int,
    latent_dim: int,
    model_save_path: str,
    weights_save_path: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> PopulationDKGP:
    """Train the population DKGP model.
    
    Args:
        data: Dictionary containing data tensors
        input_dim: Input dimension
        latent_dim: Latent dimension
        model_save_path: Path to save the model
        weights_save_path: Path to save the weights
        device: Device to run the model on
        
    Returns:
        Trained population DKGP model
    """
    print("Training population DKGP model...")
    

    model = PopulationDKGP(
        train_x=data['train_x'],
        train_y=data['train_y'],
        input_dim=input_dim,
        hidden_dim=64,  # Μπορείτε να προσαρμόσετε το hidden_dim αν χρειάζεται
        feature_dim=latent_dim,
        device=device
    )
    
    history = model.fit(
        train_x=data['train_x'],
        train_y=data['train_y'],
        num_epochs=500, 
        lr=0.01844 
    )
    
    # Save model
    print("\nSaving model and weights...")
    model.save_model(model_save_path, weights_save_path)
    
    return model

def train_adaptive_shrinkage(
    pop_model: PopulationDKGP,
    data: Dict[str, torch.Tensor],
    model_save_path: str,
    weights_save_path: str,
    val_ids: List[str],
    val_subject_indices: Dict[str, Tuple[int, int]]
) -> AdaptiveShrinkage:
    """Train the adaptive shrinkage estimator.
    
    Args:
        pop_model: Trained population model
        data: Dictionary containing data tensors
        model_save_path: Path to save the model
        val_ids: List of validation subject IDs
        val_subject_indices: Dictionary mapping subject IDs to their start and end indices in the data
    Returns:
        Trained adaptive shrinkage model
    """
    print("Training adaptive shrinkage estimator...")
    
    # Set publication-quality plot parameters
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'serif',
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.figsize': (10, 6),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })
    
    # Initialize and train subject-specific models for validation subjects
    y_ss_list, V_ss_list, y_pp_list, V_pp_list, n_obs_list, T_obs_list, oracle_alpha_list = [], [], [], [], [], [], []
    for subject_id in val_ids:

        start_idx = val_subject_indices[subject_id][0]
        end_idx = val_subject_indices[subject_id][1]

        subject_data_x = data['val_x'][start_idx:end_idx+1]
        subject_data_y = data['val_y'][start_idx:end_idx+1]
    
        for i in range(1, subject_data_x.shape[0] - 1):  # Start from the second observation to the second to last

            # Get subject data up to current observation for training the subject-specific model
            x_sub_observed = data['val_x'][:i+1]
            y_sub_observed = data['val_y'][:i+1] 

            # Train subject-specific model
            ss_model = SubjectSpecificDKGP(
                train_x=x_sub_observed,
                train_y=y_sub_observed,
                input_dim=pop_model.input_dim,
                latent_dim=pop_model.latent_dim,
                population_params=pop_model.get_deep_params(),
                learning_rate=0.01,  # Αυξημένος ρυθμός μάθησης
                n_epochs=200,        # Αυξημένος αριθμός εποχών
                weight_decay=0.01    # Μειωμένο weight decay
            )
            
            # Εκπαίδευση με παρακολούθηση της απώλειας
            history = ss_model.fit(x_sub_observed, y_sub_observed, verbose=True)
            
            # Έλεγχος αν η απώλεια μειώνεται
            if len(history['train_loss']) > 1:
                initial_loss = history['train_loss'][0]
                final_loss = history['train_loss'][-1]
                loss_reduction = (initial_loss - final_loss) / initial_loss * 100
                print(f"Loss reduction: {loss_reduction:.2f}% (Initial: {initial_loss:.4f}, Final: {final_loss:.4f})")
            
            # Get population model prediction for the whole trajectory 
            y_true = subject_data_y
            y_pp, _, _, V_pp = pop_model.predict(subject_data_x)

            # Get predictions for the whole trajectory
            y_ss, _, _, V_ss = ss_model.predict(subject_data_x)

            # Convert to numpy for plotting
            y_true_np = y_true.cpu().numpy()
            y_ss_np = y_ss.cpu().numpy()
            
            # Calculate confidence intervals (2 standard deviations)
            if isinstance(V_ss, np.ndarray):
                std_ss = np.sqrt(V_ss)
            else:
                std_ss = torch.sqrt(V_ss).cpu().numpy()
                
            upper_bound = y_ss_np + 1.96 * std_ss
            lower_bound = y_ss_np - 1.96 * std_ss
            
            # Get time values
            time = subject_data_x[:, -1].cpu().numpy()
            
            # Plot trajectories with confidence intervals
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot confidence interval
            ax.fill_between(time, lower_bound.flatten(), upper_bound.flatten(), 
                            alpha=0.3, color='green', label='95% Confidence Interval')
            
            # Plot ground truth and predictions
            ax.scatter(time, y_true_np, s=80, marker='o', color='blue', 
                       label='Ground Truth', zorder=3, edgecolors='black', linewidths=0.5)
            ax.plot(time, y_ss_np, linewidth=2.5, color='green', 
                    label='Subject-Specific Prediction', zorder=2)
            
            # Customize plot
            ax.set_title(f'Subject {subject_id} Trajectory Prediction (Obs: {i+1})')
            ax.set_xlabel('Time from baseline (in months)')
            ax.set_ylabel('Hippocampal Volume (standardized)')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add legend with shadow
            legend = ax.legend(frameon=True, fancybox=True, shadow=True, loc='best')
            frame = legend.get_frame()
            frame.set_facecolor('white')
            frame.set_alpha(0.9)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(f'results/subject_{subject_id}_trajectory_obs_{i}.png')
            plt.close()
            
            y_ss_list.append(y_ss)
            V_ss_list.append(V_ss)
            y_pp_list.append(y_pp)
            V_pp_list.append(V_pp)
            n_obs_list.append(i + 1)
            T_obs_list.append(subject_data_x[-1, -1])

            # find the oracle alpha for this predicted y_ss, y_pp and y_true    
            oracle_alpha =  optimize_alpha_with_subject_simple(y_pp, y_ss, y_true)

            oracle_alpha_list.append(oracle_alpha)

    # Create oracle dataset
    oracle_dataset = {
        'y_pp': torch.cat(y_pp_list),
        'V_pp': torch.cat(V_pp_list),
        'y_ss': torch.cat(y_ss_list),
        'V_ss': torch.cat(V_ss_list),
        'T_obs': torch.cat(T_obs_list),
        'oracle_alpha': torch.tensor(oracle_alpha_list)
    }

    print('Oracle Dataset is created!')
    sys.exit(0)

    shrinkage = AdaptiveShrinkage()
    shrinkage.fit(oracle_dataset)
    
    # Save model
    shrinkage.save_model(model_save_path)
    
    return shrinkage

def evaluate_personalization(
    pop_model: PopulationDKGP,
    shrinkage: AdaptiveShrinkage,
    data: Dict[str, torch.Tensor],
    test_ids: List[str],
    test_subject_indices: Dict[str, Tuple[int, int]],
    max_history: int = 10
) -> pd.DataFrame:
    """Evaluate personalization on test subjects.
    
    Args:
        pop_model: Trained population model
        shrinkage: Trained adaptive shrinkage model
        data: Dictionary containing data tensors
        test_ids: List of test subject IDs
        test_subject_indices: Dictionary mapping test subject IDs to their start and end indices
        max_history: Maximum number of history points to consider
        
    Returns:
        DataFrame with evaluation results
    """
    print("Evaluating personalization on test subjects...")

    results = {
        'subject_id': [],
        'n_observations': [],
        'T_obs': [],
        'mse_population': [],
        'mse_subject_specific': [],
        'mse_personalized': []
    }
    
    # Store alpha values for plotting
    alpha_values = []

    for subject_id in test_ids:
        print(f"Processing subject {subject_id}...")
        
        # Get subject data using indices
        start_idx, end_idx = test_subject_indices[subject_id]
        x_subject = data['test_x'][start_idx:end_idx+1]
        y_subject = data['test_y'][start_idx:end_idx+1]
        

        # iterate over the number of observations
        for n_obs in range(1, min(len(x_subject), max_history + 1)):

            x_subject_history = x_subject[:n_obs]
            y_subject_history = y_subject[:n_obs]

            T_obs = x_subject_history[n_obs-1, -1]

            # Get population predictions
            y_pp, _, _, V_pp = pop_model.predict(x_subject)
            
            # Get subject-specific predictions
            ss_model = SubjectSpecificDKGP(
                train_x=x_subject_history,
                train_y=y_subject_history,
                input_dim=pop_model.input_dim,
                latent_dim=pop_model.latent_dim,
                population_params=pop_model.get_deep_params()
            )
            ss_model.fit(x_subject_history, y_subject_history, verbose=False)
            
            # Get predictions
            y_ss,_, _, V_ss = ss_model.predict( )
            
            # Get combined predictions
            combined_pred, combined_var, alpha = shrinkage.predict(
                pop_pred=y_pp,
                ss_pred=y_ss,
                n_obs=torch.tensor([n_obs] * len(x_subject)),
                pop_var=V_pp,
                ss_var=V_ss ,
                T_obs=T_obs,
                return_alpha=True  # Ensure the function returns alpha
            )
     
            # Calculate MSE
            mse_pop = ((y_pp - y_subject) ** 2).mean().item()
            mse_ss = ((y_ss - y_subject) ** 2).mean().item()
            mse_combined = ((combined_pred - y_subject) ** 2).mean().item()
            
            # Store results
            results['subject_id'].append(subject_id)
            results['n_observations'].append(n_obs)
            results['mse_population'].append(mse_pop)
            results['mse_subject_specific'].append(mse_ss)
            results['mse_personalized'].append(mse_combined)

            # Store alpha values
            alpha_values.extend(alpha.cpu().numpy())

            # Generate visuals for the current subject's predicted trajectory
            plt.figure(figsize=(10, 6))
            true_trajectory = y_subject.cpu().numpy()
            predicted_trajectory = combined_pred.cpu().numpy()
            prediction_var = combined_var.cpu().numpy()
            prediction_std = np.sqrt(prediction_var)

            plt.plot(range(len(true_trajectory)), true_trajectory, label='True Trajectory')
            plt.plot(range(len(predicted_trajectory)), predicted_trajectory, label='Predicted Trajectory')
            plt.fill_between(range(len(predicted_trajectory)),
                             predicted_trajectory - 1.96 * prediction_std,
                             predicted_trajectory + 1.96 * prediction_std,
                             alpha=0.2, label='Prediction Std')
            plt.title(f'Subject {subject_id} Predicted Trajectory')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.savefig(f'results/subject_{subject_id}_trajectory_obs_{n_obs}.png')
            plt.close()
    
    # Plot the distribution of alpha with the number of observations
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=results['n_observations'], y=alpha_values)
    plt.title('Distribution of Alpha with Number of Observations')
    plt.xlabel('Number of Observations')
    plt.ylabel('Alpha')
    plt.grid(True)
    plt.savefig('results/alpha_distribution.png')
    plt.close()

    return pd.DataFrame(results)

def evaluate_population_model(
    pop_model: PopulationDKGP,
    data: Dict[str, torch.Tensor],
    test_ids: List[str],
    test_subject_indices: Dict[str, Tuple[int, int]]
) -> pd.DataFrame:
    """Evaluate the population model on test subjects.
    
    Args:
        pop_model: Trained population model
        data: Dictionary containing data tensors
        test_ids: List of test subject IDs
        test_subject_indices: Dictionary mapping test subject IDs to their start and end indices
        
    Returns:
        DataFrame with evaluation results
    """
    results = {
        'subject_id': [],
        'mse_population': [],
        'mae_population': []
    }
    
    # Set publication-quality plot parameters
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.figsize': (10, 6),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })
    
    for subject_id in test_ids:
        # Get subject data using indices
        start_idx, end_idx = test_subject_indices[subject_id]
        x_subject = data['test_x'][start_idx:end_idx+1]
        y_subject = data['test_y'][start_idx:end_idx+1]
        
        # Get population predictions with uncertainty
        y_pp, _, _, V_pp = pop_model.predict(x_subject)
        
        # Convert y_pp to Tensor if it's a numpy array
        if isinstance(y_pp, np.ndarray):
            y_pp = torch.from_numpy(y_pp).to(y_subject.device)
        
        # Calculate MSE and MAE
        mse_pop = ((y_pp - y_subject) ** 2).mean().item()
        mae_pop = torch.abs(y_pp - y_subject).mean().item()
        
        # Store results
        results['subject_id'].append(subject_id)
        results['mse_population'].append(mse_pop)
        results['mae_population'].append(mae_pop)

        # Convert to numpy for plotting
        y_subject_np = y_subject.cpu().numpy()
        y_pp_np = y_pp.cpu().numpy()
        
        # Calculate confidence intervals (2 standard deviations)
        if isinstance(V_pp, np.ndarray):
            std_pp = np.sqrt(V_pp)
        else:
            std_pp = torch.sqrt(V_pp).cpu().numpy()
            
        upper_bound = y_pp_np + 1.96 * std_pp
        lower_bound = y_pp_np - 1.96 * std_pp
        

        time = x_subject[:, -1].cpu().numpy()


        # Plot trajectories with confidence intervals
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot confidence interval
        ax.fill_between(time, lower_bound.flatten(), upper_bound.flatten(), 
                        alpha=0.3, color='orange', label='95% Confidence Interval')
        
        # Plot ground truth and predictions
        ax.scatter(time, y_subject_np, s=80, marker='o', color='blue', 
                   label='Ground Truth', zorder=3, edgecolors='black', linewidths=0.5)
        ax.plot(time, y_pp_np, linewidth=2.5, color='orange', 
                label='Population Prediction', zorder=2)
        
        # Customize plot
        ax.set_title(f'Subject {subject_id} Trajectory Prediction')
        ax.set_xlabel('Time from baseline (in months)')
        ax.set_ylabel('Hippocampal Volume (standardized)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add legend with shadow
        legend = ax.legend(frameon=True, fancybox=True, shadow=True, loc='best')
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_alpha(0.9)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f'results/subject_{subject_id}_population_trajectory.png')
        plt.close()

    return pd.DataFrame(results)

def get_deep_params(self):
    """Get deep kernel parameters excluding feature extractor."""
    population_hyperparams = {}
    for param_name, param in self.model.named_parameters():
        if not param_name.startswith('feature'):
            population_hyperparams[param_name] = param
        else:
            pass 
    return population_hyperparams

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

    # Extract and print feature importance
    weights = self.model.feature_extractor.final_linear.weight.cpu().detach()
    print("Feature Importance Weights:", weights)


def main():
    # Parameters
    data_path = "data/biomarker_data.csv"  # Path to your data file
    model_dir = "models"
    results_dir = "results"
    
    # Create output directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load and preprocess data
    data, train_ids, val_ids, test_ids, train_subject_indices, val_subject_indices, test_subject_indices = load_and_preprocess_data(data_path)

    # Set dimensions based on data
    input_dim = data['train_x'].shape[1]
    latent_dim = input_dim // 2
    
    # Train population model
    print("Model save path:", os.path.join(model_dir, "population_dkgp.pt"))
    print("Weights save path:", os.path.join(model_dir, "population_dkgp_weights.pt"))
    
    # Train population model
    print(type(data))
    print('Training population model...')
    pop_model = train_population_model(
        data,
        input_dim,
        latent_dim,
        model_save_path=os.path.join(model_dir, "population_dkgp.pt"),
        weights_save_path=os.path.join(model_dir, "population_dkgp_weights.pt"))
    
    # Evaluate the population model
    # print('Evaluating population model...')
    # results = evaluate_population_model(
    #     pop_model,
    #     data,
    #     test_ids,
    #     test_subject_indices
    # )

    # Train adaptive shrinkage
    print('Training adaptive shrinkage...')
    adaptive_shrinkage_estimator = train_adaptive_shrinkage(
        pop_model,
        data,
        os.path.join(model_dir, "adaptive_shrinkage.json"),
        os.path.join(model_dir, "adaptive_shrinkage_weights.json"), 
        val_ids,
        val_subject_indices
    )
    
    # Evaluate personalization
    results = evaluate_personalization(
        pop_model,
        adaptive_shrinkage_estimator,
        data,
        test_ids,
        test_subject_indices
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

    print('Done!')

if __name__ == "__main__":
    main() 