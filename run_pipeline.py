"""
Complete pipeline for biomarker prediction using Deep Kernel Regression with Adaptive Shrinkage Estimation
"""
import os
import sys
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List, Optional, Union
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import xgboost as xgb
from scipy.optimize import minimize_scalar
import time

from src.adaptive_shrinkage_dkgp.models.population_dkgp import PopulationDKGP
from src.adaptive_shrinkage_dkgp.models.ss_dkgp import SubjectSpecificDKGP
from src.adaptive_shrinkage_dkgp.models.adaptive_shrinkage import AdaptiveShrinkage
from src.adaptive_shrinkage_dkgp.models.alpha_models import optimize_alpha_with_subject_simple

def load_and_preprocess_data(
    data_path: str,
    train_size: float = 0.7,
    val_size: float = 0.02,
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
    
    print(f'Train IDs: {len(train_ids)}')
    print(f'Val IDs: {len(val_ids)}')
    print(f'Test IDs: {len(test_ids)}')
    
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
        lr=0.01844,
        weight_decay=0.01
    )
    
    # Save model
    print("\nSaving model and weights...")
    model.save_model(model_save_path, weights_save_path)
    
    return model

def create_oracle_dataset(
    pop_model: PopulationDKGP,
    data: Dict[str, torch.Tensor],
    val_ids: List[str],
    val_subject_indices: Dict[str, Tuple[int, int]],
    output_path: str,
    target: str = 'ROI_48'
) -> Dict[str, torch.Tensor]:
    """Create oracle dataset for adaptive shrinkage training.
    
    Args:
        pop_model: Trained population model
        data: Dictionary containing data tensors
        val_ids: List of validation subject IDs
        val_subject_indices: Dictionary mapping validation subject IDs to their start and end indices
        output_path: Path to save the oracle dataset
        target: Target biomarker name
        
    Returns:
        Dictionary containing oracle dataset tensors
    """
    
    print("Creating oracle dataset...")
     
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
    
    # Check if oracle dataset already exists
    oracle_dataset_path = os.path.join(output_path, f'oracle_dataset_{target}.pkl')
    oracle_csv_path = os.path.join(output_path, f'oracle_dataset_{target}.csv')
    
    if os.path.exists(oracle_dataset_path):
        print(f"Loading oracle dataset from {oracle_dataset_path}")
        # Φόρτωση του oracle dataset από το αρχείο pickle
        with open(oracle_dataset_path, 'rb') as f:
            oracle_dataset = pickle.load(f)
        print("Oracle dataset loaded successfully.")
        return oracle_dataset
    
    # Initialize and train subject-specific models for validation subjects
    y_ss_list, V_ss_list, y_pp_list, V_pp_list, n_obs_list, T_obs_list, oracle_alpha_list = [], [], [], [], [], [], []
    
    # Δημιουργία καταλόγου για τα αποτελέσματα αν δεν υπάρχει
    os.makedirs(os.path.join(output_path, 'plots'), exist_ok=True)
    
    # Δημιουργία DataFrame για την αποθήκευση των αποτελεσμάτων
    results_data = []
    
    for subject_id in val_ids:
        print(f'Training subject-specific model for subject {subject_id}...')
        start_idx = val_subject_indices[subject_id][0]
        end_idx = val_subject_indices[subject_id][1]

        subject_data_x = data['val_x'][start_idx:end_idx+1]
        subject_data_y = data['val_y'][start_idx:end_idx+1]
    
        print(f'Subject data x: {subject_data_x.shape}')
        print(f'Subject data y: {subject_data_y.shape}')

        for i in range(1, subject_data_x.shape[0] - 1):  # Start from the second observation to the second to last
            
            print(f'Training subject-specific model for subject {subject_id} with {i+1} observations...')
            # Get subject data up to current observation for training the subject-specific model
            x_sub_observed = subject_data_x[:i+1]  # Χρησιμοποιούμε τα δεδομένα του συγκεκριμένου υποκειμένου
            y_sub_observed = subject_data_y[:i+1]  # Χρησιμοποιούμε τα δεδομένα του συγκεκριμένου υποκειμένου

            print(f'x_sub_observed: {x_sub_observed.shape}')
            print(f'y_sub_observed: {y_sub_observed.shape}')

            # Train subject-specific model
            ss_model = SubjectSpecificDKGP(
                train_x=x_sub_observed,
                train_y=y_sub_observed,
                input_dim=pop_model.input_dim,
                latent_dim=pop_model.latent_dim,
                population_params=pop_model.get_deep_params(),
                learning_rate=0.01844,  # Ακριβής ρυθμός μάθησης
                weight_decay=0.01,      # Weight decay
                n_epochs=400            # Αριθμός εποχών
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
            y_ss, _, _, V_ss = ss_model.predict(x_sub_observed)
            
            # Εκτύπωση στατιστικών για να βεβαιωθούμε ότι το μοντέλο αλλάζει
            print(f"Predictions summary for subject {subject_id} with {i+1} observations:")
            print(f"  Population model - Mean: {y_pp.mean().item():.4f}, Std: {y_pp.std().item():.4f}")
            print(f"  Subject-specific model - Mean: {y_ss.mean().item():.4f}, Std: {y_ss.std().item():.4f}")
            print(f"  Ground truth - Mean: {y_true.mean().item():.4f}, Std: {y_true.std().item():.4f}")
            
            # Μετατροπή σε numpy arrays για υπολογισμούς
            y_pp_np = y_pp.cpu().numpy() if isinstance(y_pp, torch.Tensor) and y_pp.is_cuda else y_pp.numpy() if isinstance(y_pp, torch.Tensor) else y_pp
            y_true_np = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) and y_true.is_cuda else y_true.numpy() if isinstance(y_true, torch.Tensor) else y_true
            y_ss_np = y_ss.cpu().numpy() if isinstance(y_ss, torch.Tensor) and y_ss.is_cuda else y_ss.numpy() if isinstance(y_ss, torch.Tensor) else y_ss
            
            # Υπολογισμός των MSE
            mse_pop = ((y_pp_np - y_true_np) ** 2).mean()
            mse_ss = ((y_ss_np - y_true_np) ** 2).mean()
            
            # Εκτύπωση των σφαλμάτων
            print(f"  MSE Population: {mse_pop:.4f}")
            print(f"  MSE Subject-Specific: {mse_ss:.4f}")
            print(f"  Improvement: {(mse_pop - mse_ss) / mse_pop * 100:.2f}%")
            
            # Convert to numpy for plotting
            # Calculate confidence intervals (2 standard deviations)
            if isinstance(V_ss, np.ndarray):
                std_ss = np.sqrt(V_ss)
            elif isinstance(V_ss, torch.Tensor):
                std_ss = torch.sqrt(V_ss)
                std_ss = std_ss.cpu().numpy() if std_ss.is_cuda else std_ss.numpy()
            else:
                std_ss = np.sqrt(V_ss)
                
            upper_bound = y_ss_np + 1.96 * std_ss
            lower_bound = y_ss_np - 1.96 * std_ss
            
            # Get time values
            if isinstance(x_sub_observed, torch.Tensor):
                time = x_sub_observed[:, -1].cpu().numpy() if x_sub_observed.is_cuda else x_sub_observed[:, -1].numpy()
            else:
                time = x_sub_observed[:, -1]
            
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
            plt.savefig(os.path.join(output_path, 'plots', f'subject_{subject_id}_trajectory_obs_{i}.png'))
            plt.close()
            
            # Αποθήκευση των μέσων τιμών των προβλέψεων και των διακυμάνσεων
            # Μετατροπή σε scalar τιμές
            if isinstance(y_ss, torch.Tensor):
                y_ss_mean = y_ss.mean().item()
                V_ss_mean = V_ss.mean().item() if isinstance(V_ss, torch.Tensor) else float(V_ss.mean())
            else:
                y_ss_mean = float(y_ss.mean())
                V_ss_mean = float(V_ss.mean())
                
            if isinstance(y_pp, torch.Tensor):
                y_pp_mean = y_pp.mean().item()
                V_pp_mean = V_pp.mean().item() if isinstance(V_pp, torch.Tensor) else float(V_pp.mean())
            else:
                y_pp_mean = float(y_pp.mean())
                V_pp_mean = float(V_pp.mean())
            
            # Προσθήκη των scalar τιμών στις λίστες
            y_ss_list.append(y_ss_mean)
            V_ss_list.append(V_ss_mean)
            y_pp_list.append(y_pp_mean)
            V_pp_list.append(V_pp_mean)
            n_obs_list.append(i + 1)
            # Διασφάλιση ότι προσθέτουμε έναν αριθμό και όχι έναν πίνακα
            if isinstance(x_sub_observed, torch.Tensor):
                time_value = x_sub_observed[-1, -1].item()
            else:
                time_value = x_sub_observed[-1, -1]
            T_obs_list.append(time_value)

            # Υπολογισμός του oracle alpha με βάση τις μέσες τιμές
            # Μετατροπή σε numpy arrays για υπολογισμούς
            if isinstance(y_true, torch.Tensor):
                y_true_mean = y_true.mean().item()
            else:
                y_true_mean = float(y_true.mean())
                
            # Υπολογισμός του oracle alpha με τις μέσες τιμές
            oracle_alpha = optimize_alpha_with_subject_simple(
                np.array([y_pp_mean]), 
                np.array([y_ss_mean]), 
                np.array([y_true_mean])
            )
            
            # Επαλήθευση της συμπεριφοράς του oracle alpha
            print(f"Subject {subject_id}, Obs {i+1}:")
            print(f"  MSE Population: {mse_pop:.4f}")
            print(f"  MSE Subject-Specific: {mse_ss:.4f}")
            print(f"  Oracle Alpha: {oracle_alpha:.4f}")
            
            oracle_alpha_list.append(oracle_alpha)
            
            # Αποθήκευση των αποτελεσμάτων στο DataFrame
            results_data.append({
                'subject_id': subject_id,
                'n_observations': i + 1,
                'mse_population': mse_pop,
                'mse_subject_specific': mse_ss,
                'oracle_alpha': oracle_alpha,
                'time_point': x_sub_observed[-1, -1].item() if isinstance(x_sub_observed, torch.Tensor) else x_sub_observed[-1, -1]
            })

    print(f"y_pp_list: {len(y_pp_list)}")
    print(f"V_pp_list: {len(V_pp_list)}")
    print(f"y_ss_list: {len(y_ss_list)}")
    print(f"V_ss_list: {len(V_ss_list)}")
    print(f"T_obs_list: {len(T_obs_list)}")
    print(f"oracle_alpha_list: {len(oracle_alpha_list)}")
    
    # Μετατροπή των λιστών σε numpy arrays
    # Τώρα που όλα τα στοιχεία είναι scalar, η μετατροπή είναι απλή
    y_pp_array = np.array(y_pp_list)
    V_pp_array = np.array(V_pp_list)
    y_ss_array = np.array(y_ss_list)
    V_ss_array = np.array(V_ss_list)
    
    # Για T_obs_list και oracle_alpha_list, που είναι πιο απλά
    T_obs_array = np.array(T_obs_list)
    oracle_alpha_array = np.array(oracle_alpha_list)
    
    # Συνδυασμός όλων των arrays σε έναν πίνακα Nx6
    # Όπου το Nx5 είναι τα features και το Nx1 είναι το target
    features = np.column_stack((y_pp_array, V_pp_array, y_ss_array, V_ss_array, T_obs_array))
    
    # Δημιουργία του oracle dataset ως numpy arrays
    X = features  # Nx5 array με τα features
    y = oracle_alpha_array  # Nx1 array με τα oracle alpha
    
    # Αποθήκευση του oracle dataset με pickle
    oracle_dataset_path_np = os.path.join(output_path, f'oracle_dataset_{target}.pkl')
    with open(oracle_dataset_path_np, 'wb') as f:
        pickle.dump({'X': X, 'y': y}, f)
    
    # Αποθήκευση των αποτελεσμάτων σε CSV
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(oracle_csv_path, index=False)
    
    print(f"Oracle dataset saved to {oracle_dataset_path_np}")
    print(f"Oracle dataset results saved to {oracle_csv_path}")
    
    # Επιστροφή του oracle dataset ως dictionary για συμβατότητα
    oracle_dataset = {
        'X': X,
        'y': y
    }
    
    return oracle_dataset

def train_adaptive_shrinkage(
    pop_model: PopulationDKGP,
    data: Dict[str, torch.Tensor],
    model_save_path: str,
    weights_save_path: str,
    val_ids: List[str],
    val_subject_indices: Dict[str, Tuple[int, int]],
    output_path: str = 'results',
    target: str = 'ROI_48'
) -> AdaptiveShrinkage:
    """Train the adaptive shrinkage estimator.
    
    Args:
        pop_model: Trained population model
        data: Dictionary containing data tensors
        model_save_path: Path to save the trained model
        weights_save_path: Path to save the model weights
        val_ids: List of validation subject IDs
        val_subject_indices: Dictionary mapping validation subject IDs to their start and end indices
        output_path: Path to save the oracle dataset and results
        target: Target biomarker name
        
    Returns:
        Trained adaptive shrinkage estimator
    """
    print("Training adaptive shrinkage estimator...")
    
    # Έλεγχος αν το oracle dataset υπάρχει ήδη
    oracle_dataset_path = os.path.join(output_path, f'oracle_dataset_{target}.pkl')
    
    if os.path.exists(oracle_dataset_path):
        print(f"Loading oracle dataset from {oracle_dataset_path}")
        # Φόρτωση του oracle dataset από το αρχείο pickle
        with open(oracle_dataset_path, 'rb') as f:
            oracle_dataset = pickle.load(f)
    else:
        # Δημιουργία του oracle dataset
        oracle_dataset = create_oracle_dataset(
            pop_model=pop_model,
            data=data,
            val_ids=val_ids,
            val_subject_indices=val_subject_indices,
            output_path=output_path,
            target=target
        )
    
    # Split the oracle dataset into training and validation sets
    print("Splitting oracle dataset into training and validation sets...")
    
    # Λήψη των X και y από το oracle dataset
    X = oracle_dataset['X']  # Nx5 array με τα features
    y = oracle_dataset['y']  # Nx1 array με τα oracle alpha
    
    # Διαχωρισμός σε σύνολα εκπαίδευσης και επικύρωσης (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {X_train.shape[0]}, Validation set size: {X_val.shape[0]}")
    
    # Hyperparameter tuning
    print("Performing hyperparameter tuning...")
    
    # Ορισμός των υπερπαραμέτρων για βελτιστοποίηση
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'subsample': [0.7, 0.8, 0.9]
    }
    
    # Δημιουργία του μοντέλου XGBoost για βελτιστοποίηση
    xgb_model = xgb.XGBRegressor(random_state=42)
    
    # Εκτέλεση της βελτιστοποίησης υπερπαραμέτρων με 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Εμφάνιση των βέλτιστων υπερπαραμέτρων
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {-grid_search.best_score_:.4f} MSE")
    
    # Αξιολόγηση του μοντέλου στο σύνολο επικύρωσης
    best_model = grid_search.best_estimator_
    val_predictions = best_model.predict(X_val)
    
    # Υπολογισμός μετρικών αξιολόγησης
    mse = mean_squared_error(y_val, val_predictions)
    mae = mean_absolute_error(y_val, val_predictions)
    r2 = r2_score(y_val, val_predictions)
    
    print(f"Validation MSE: {mse:.4f}")
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation R²: {r2:.4f}")
    
    # Οπτικοποίηση των προβλέψεων έναντι των πραγματικών τιμών
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, val_predictions, alpha=0.7)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
    plt.xlabel('True Alpha Values')
    plt.ylabel('Predicted Alpha Values')
    plt.title('Adaptive Shrinkage Model Validation')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_path, 'adaptive_shrinkage_validation.png'))
    plt.close()
    
    # Οπτικοποίηση της σημαντικότητας των χαρακτηριστικών
    feature_importance = best_model.feature_importances_
    feature_names = ['Population Prediction', 'Population Variance', 
                     'Subject-Specific Prediction', 'Subject-Specific Variance', 
                     'Time Observation']
    
    plt.figure(figsize=(12, 6))
    plt.barh(feature_names, feature_importance)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance in Adaptive Shrinkage Model')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'adaptive_shrinkage_feature_importance.png'))
    plt.close()
    
    # Δημιουργία και εκπαίδευση του τελικού μοντέλου με τις βέλτιστες υπερπαραμέτρους
    print("Training final adaptive shrinkage model...")
    shrinkage = AdaptiveShrinkage(
        n_estimators=grid_search.best_params_['n_estimators'],
        learning_rate=grid_search.best_params_['learning_rate'],
        max_depth=grid_search.best_params_['max_depth'],
        subsample=grid_search.best_params_['subsample']
    )
    
    # Εκπαίδευση του μοντέλου με όλα τα δεδομένα
    shrinkage.model = best_model
    
    # Αποθήκευση του μοντέλου
    print("Saving adaptive shrinkage model...")
    shrinkage.save_model(model_save_path)
    
    # Αποθήκευση των βέλτιστων υπερπαραμέτρων
    with open(weights_save_path, 'w') as f:
        json.dump(grid_search.best_params_, f)
    
    return shrinkage

def evaluate_personalization(
    pop_model: PopulationDKGP,
    shrinkage: AdaptiveShrinkage,
    data: Dict[str, torch.Tensor],
    test_ids: List[str],
    test_subject_indices: Dict[str, Tuple[int, int]],
    max_history: int = 10
) -> pd.DataFrame:
    """Evaluate personalization performance on test subjects.
    
    Args:
        pop_model: Trained population model
        shrinkage: Trained adaptive shrinkage model
        data: Dictionary containing data tensors
        test_ids: List of test subject IDs
        test_subject_indices: Dictionary mapping subject IDs to indices in the test data
        max_history: Maximum number of observations to use for personalization
        
    Returns:
        DataFrame containing evaluation results
    """
    # Create results dictionary
    results = {
        'subject_id': [],
        'n_observations': [],
        'mse_population': [],
        'mse_subject_specific': [],
        'mse_personalized': []
    }
    
    # Create dictionary to store predictions
    predictions_data = {
        'subject_id': [],
        'n_observations': [],
        'time_point': [],
        'true_value': [],
        'population_pred': [],
        'subject_specific_pred': [],
        'personalized_pred': [],
        'upper': [],  # Upper bound for confidence interval
        'lower': [],  # Lower bound for confidence interval
        'alpha': [],
        'Model': [],  # For Plotly grouping
        'History': [],  # For Plotly grouping
        'time': []  # For Plotly x-axis
    }
    
    # Store alpha values for analysis
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

            # create the Tobs_list
            Tobs_list = torch.tensor([T_obs] * len(x_subject))
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
            y_ss, _, _, V_ss = ss_model.predict(x_subject)
            
            # Get combined predictions
            combined_pred, combined_var, alpha = shrinkage.predict(
                pop_pred=y_pp,
                ss_pred=y_ss,
                pop_var=V_pp,
                ss_var=V_ss,
                Tobs=Tobs_list
            )
     
            # Ensure all variables are tensors for MSE calculation
            if isinstance(y_pp, np.ndarray):
                y_pp = torch.tensor(y_pp, device=y_subject.device)
            if isinstance(y_ss, np.ndarray):
                y_ss = torch.tensor(y_ss, device=y_subject.device)
            if isinstance(combined_pred, np.ndarray):
                combined_pred = torch.tensor(combined_pred, device=y_subject.device)
            if isinstance(combined_var, np.ndarray):
                combined_var = torch.tensor(combined_var, device=y_subject.device)
            if isinstance(alpha, np.ndarray):
                alpha = torch.tensor(alpha, device=y_subject.device)
            
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
            if alpha.ndim == 0:  # Handle scalar alpha
                alpha_values.append(alpha.item())
            else:
                alpha_values.extend(alpha.cpu().numpy())

            # Store predictions for later plotting
            for i in range(len(y_subject)):
                # Store population model predictions
                predictions_data['subject_id'].append(subject_id)
                predictions_data['n_observations'].append(n_obs)
                predictions_data['time_point'].append(i)
                predictions_data['true_value'].append(y_subject[i].item())
                predictions_data['population_pred'].append(y_pp[i].item())
                predictions_data['subject_specific_pred'].append(y_ss[i].item())
                predictions_data['personalized_pred'].append(combined_pred[i].item())
                
                # Calculate confidence intervals (95%)
                std_dev = torch.sqrt(combined_var[i]).item()
                predictions_data['upper'].append(combined_pred[i].item() + 1.96 * std_dev)
                predictions_data['lower'].append(combined_pred[i].item() - 1.96 * std_dev)
                
                # Store alpha
                if alpha.ndim == 0:  # Handle scalar alpha
                    predictions_data['alpha'].append(alpha.item())
                else:
                    predictions_data['alpha'].append(alpha[i].item())
                
                # Add fields for Plotly
                predictions_data['Model'].append('pers-DKGP (Alpha Simple)')
                predictions_data['History'].append(n_obs)
                predictions_data['time'].append(i)  # Use time_point as time for now

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Also create a ground truth DataFrame for Plotly
    groundtruth_data = {
        'subject_id': [],
        'time_point': [],
        'time': [],
        'y': []
    }
    
    # Add ground truth data points
    for subject_id in test_ids:
        start_idx, end_idx = test_subject_indices[subject_id]
        y_subject = data['test_y'][start_idx:end_idx+1]
        
        for i in range(len(y_subject)):
            groundtruth_data['subject_id'].append(subject_id)
            groundtruth_data['time_point'].append(i)
            groundtruth_data['time'].append(i)  # Use time_point as time for now
            groundtruth_data['y'].append(y_subject[i].item())
    
    groundtruth_df = pd.DataFrame(groundtruth_data)
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame(predictions_data)
    
    # Save both DataFrames
    os.makedirs('results', exist_ok=True)
    predictions_df.to_csv('results/predictions_for_plotting.csv', index=False)
    groundtruth_df.to_csv('results/groundtruth_for_plotting.csv', index=False)
    print(f"Predictions saved to results/predictions_for_plotting.csv")
    print(f"Ground truth saved to results/groundtruth_for_plotting.csv")
    
    # Calculate average MSE for each number of observations
    avg_results = results_df.groupby('n_observations').mean().reset_index()
    
    # Plot average MSE vs number of observations
    plt.figure(figsize=(10, 6))
    plt.plot(avg_results['n_observations'], avg_results['mse_population'], label='Population Model')
    plt.plot(avg_results['n_observations'], avg_results['mse_subject_specific'], label='Subject-Specific Model')
    plt.plot(avg_results['n_observations'], avg_results['mse_personalized'], label='Personalized Model')
    plt.xlabel('Number of Observations')
    plt.ylabel('Mean Squared Error')
    plt.title('Average MSE vs Number of Observations')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/average_mse.png')
    plt.close()
    
    # Plot alpha distribution
    plt.figure(figsize=(10, 6))
    plt.hist(alpha_values, bins=20)
    plt.xlabel('Alpha')
    plt.ylabel('Frequency')
    plt.title('Distribution of Alpha Values')
    plt.grid(True)
    plt.savefig('results/alpha_distribution.png')
    plt.close()
    
    return results_df

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

def predict(
    self,
    pop_pred: Union[np.ndarray, torch.Tensor],
    ss_pred: Union[np.ndarray, torch.Tensor],
    n_obs: Union[np.ndarray, torch.Tensor],
    pop_var: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ss_var: Optional[Union[np.ndarray, torch.Tensor]] = None
) -> Union[np.ndarray, torch.Tensor]:
    """Predict optimal shrinkage weights and combine predictions.
    
    Args:
        pop_pred: Population model predictions
        ss_pred: Subject-specific model predictions
        n_obs: Number of observations
        pop_var: Optional population model prediction uncertainties
        ss_var: Optional subject-specific model prediction uncertainties
        
    Returns:
        Combined predictions using learned shrinkage weights
    """
    # Convert to numpy if needed
    is_tensor = torch.is_tensor(pop_pred)
    if is_tensor:
        pop_pred = pop_pred.cpu().numpy() if pop_pred.is_cuda else pop_pred.numpy()
        ss_pred = ss_pred.cpu().numpy() if ss_pred.is_cuda else ss_pred.numpy()
        n_obs = n_obs.cpu().numpy() if n_obs.is_cuda else n_obs.numpy()
        if pop_var is not None:
            pop_var = pop_var.cpu().numpy() if pop_var.is_cuda else pop_var.numpy()
        if ss_var is not None:
            ss_var = ss_var.cpu().numpy() if ss_var.is_cuda else ss_var.numpy()
    
    # Prepare features
    # Διασφάλιση ότι τα δεδομένα έχουν τη σωστή μορφή
    if isinstance(pop_pred, np.ndarray) and pop_pred.ndim > 1:
        # Αν έχουμε πολυδιάστατα δεδομένα, παίρνουμε την πρώτη διάσταση
        pop_pred_val = pop_pred[:, 0].reshape(-1, 1)
        ss_pred_val = ss_pred[:, 0].reshape(-1, 1)
        
        if pop_var is not None:
            pop_var_val = pop_var[:, 0].reshape(-1, 1)
        if ss_var is not None:
            ss_var_val = ss_var[:, 0].reshape(-1, 1)
    else:
        # Αν έχουμε μονοδιάστατα δεδομένα
        pop_pred_val = pop_pred.reshape(-1, 1)
        ss_pred_val = ss_pred.reshape(-1, 1)
        
        if pop_var is not None:
            pop_var_val = pop_var.reshape(-1, 1)
        if ss_var is not None:
            ss_var_val = ss_var.reshape(-1, 1)
    
    # Δημιουργία του feature vector
    features = []
    features.append(pop_pred_val)
    
    if pop_var is not None:
        features.append(pop_var_val)
    else:
        # Αν δεν έχουμε διακύμανση, προσθέτουμε μηδενικά
        features.append(np.zeros_like(pop_pred_val))
        
    features.append(ss_pred_val)
    
    if ss_var is not None:
        features.append(ss_var_val)
    else:
        # Αν δεν έχουμε διακύμανση, προσθέτουμε μηδενικά
        features.append(np.zeros_like(ss_pred_val))
        
    # Προσθήκη του αριθμού παρατηρήσεων
    n_obs_reshaped = n_obs.reshape(-1, 1) if isinstance(n_obs, np.ndarray) else np.array([[n_obs]])
    features.append(n_obs_reshaped)
    
    # Συνένωση των features
    X = np.hstack(features)
    
    # Πρόβλεψη των βαρών
    adaptive_shrinkage_alpha = self.model.predict(X)
    
    # Διασφάλιση ότι το alpha είναι στο διάστημα [0, 1]
    adaptive_shrinkage_alpha = np.clip(adaptive_shrinkage_alpha, 0, 1)
    
    # Συνδυασμός των προβλέψεων
    if isinstance(pop_pred, np.ndarray) and pop_pred.ndim > 1:
        # Αν έχουμε πολλαπλές προβλέψεις, εφαρμόζουμε το alpha σε κάθε πρόβλεψη
        alpha_expanded = adaptive_shrinkage_alpha.reshape(-1, 1)
        personalized_pred = alpha_expanded * pop_pred + (1 - alpha_expanded) * ss_pred
        
        if pop_var is not None and ss_var is not None:
            personalized_pred_var = alpha_expanded**2 * pop_var + (1 - alpha_expanded)**2 * ss_var
        else:
            personalized_pred_var = None
    else:
        # Αν έχουμε μονοδιάστατα δεδομένα
        personalized_pred = adaptive_shrinkage_alpha * pop_pred + (1 - adaptive_shrinkage_alpha) * ss_pred
        
        if pop_var is not None and ss_var is not None:
            personalized_pred_var = adaptive_shrinkage_alpha**2 * pop_var + (1 - adaptive_shrinkage_alpha)**2 * ss_var
        else:
            personalized_pred_var = None
    
    # Επιστροφή των αποτελεσμάτων ως tensor αν τα δεδομένα εισόδου ήταν tensors
    if is_tensor:
        personalized_pred = torch.from_numpy(personalized_pred)
        if personalized_pred_var is not None:
            personalized_pred_var = torch.from_numpy(personalized_pred_var)
        adaptive_shrinkage_alpha = torch.from_numpy(adaptive_shrinkage_alpha)
    
    return personalized_pred, personalized_pred_var, adaptive_shrinkage_alpha

def optimize_alpha_with_subject_simple(y_p, y_t, y_g):
    """Optimize alpha for a subject using a simple approach.
    
    Args:
        y_p: Population model predictions
        y_t: Subject-specific model predictions
        y_g: Ground truth values
        
    Returns:
        Optimal alpha value
    """
    # Ensure all inputs are numpy arrays
    if isinstance(y_p, torch.Tensor):
        y_p = y_p.cpu().numpy() if y_p.is_cuda else y_p.numpy()
    if isinstance(y_t, torch.Tensor):
        y_t = y_t.cpu().numpy() if y_t.is_cuda else y_t.numpy()
    if isinstance(y_g, torch.Tensor):
        y_g = y_g.cpu().numpy() if y_g.is_cuda else y_g.numpy()
    
    # Define the objective function to minimize
    def objective(alpha):
        y_combined = alpha * y_p + (1 - alpha) * y_t
        return np.mean((y_combined - y_g) ** 2)
    
    # Optimize alpha in the range [0, 1]
    result = minimize_scalar(objective, bounds=(0, 1), method='bounded')
    
    return result.x

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
        model_save_path=os.path.join(model_dir, "adaptive_shrinkage.json"),
        weights_save_path=os.path.join(model_dir, "adaptive_shrinkage_weights.json"), 
        val_ids=val_ids,
        val_subject_indices=val_subject_indices,
        output_path=results_dir,
        target='ROI_48'  # Προσαρμόστε αυτό ανάλογα με το biomarker που χρησιμοποιείτε
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
        'mse_personalized': ['mean', 'std']
    })
    print(summary)

    print('Done!')

if __name__ == "__main__":
    main() 