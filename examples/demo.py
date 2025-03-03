import numpy as np
import torch
from adaptive_shrinkage_dkgp.models.population_dkgp import PopulationDKGP
from adaptive_shrinkage_dkgp.models.ss_dkgp import SubjectSpecificDKGP
from adaptive_shrinkage_dkgp.models.adaptive_shrinkage import AdaptiveShrinkage

# Load and preprocess data
def load_temp_data():
    # TODO: Replace with actual data loading
    n_subjects = 100
    n_timepoints = 50
    X = np.random.randn(n_subjects * n_timepoints, 5)  # Features
    y = np.random.randn(n_subjects * n_timepoints, 1)  # Outcomes
    subject_ids = np.repeat(np.arange(n_subjects), n_timepoints)
    return X, y, subject_ids

def main():
    # Load data
    X, y, subject_ids = load_temp_data()
    
    # Split data into population, validation, and test sets
    unique_subjects = np.unique(subject_ids)
    np.random.shuffle(unique_subjects)
    
    n_train = int(0.6 * len(unique_subjects))
    n_val = int(0.2 * len(unique_subjects))
    
    train_subjects = unique_subjects[:n_train]
    val_subjects = unique_subjects[n_train:n_train+n_val]
    test_subjects = unique_subjects[n_train+n_val:]
    
    # Create train/val/test masks
    train_mask = np.isin(subject_ids, train_subjects)
    val_mask = np.isin(subject_ids, val_subjects)
    test_mask = np.isin(subject_ids, test_subjects)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    
    # Train population DKGP model
    print("Training population DKGP model...")
    p_dkgp = PopulationDKGP()
    p_dkgp.fit(X_train, y_train)
    
    # Train adaptive shrinkage estimator
    print("Training adaptive shrinkage estimator...")
    shrinkage = AdaptiveShrinkage()
    shrinkage.fit(X_val, y_val)
    
    # Personalization demo for test subjects
    print("\nDemonstrating personalization for test subjects...")
    for subject_id in test_subjects[:20]:  # Demo with first 20 test subjects
        subject_mask = subject_ids == subject_id
        X_subject = X[subject_mask]
        y_subject = y[subject_mask]
        
        # Simulate increasing number of observations
        for n_obs in range(1, len(X_subject) + 1):
            X_sub = X_subject[:n_obs]
            y_sub = y_subject[:n_obs]
            
            # Train subject-specific DKGP
            ss_dkgp = SubjectSpecificDKGP(p_dkgp.get_deep_params())
            ss_dkgp.fit(X_sub, y_sub)
            
            # Get predictions from both models
            p_pred = p_dkgp.predict(X_sub)
            ss_pred = ss_dkgp.predict(X_sub)
            
            # Apply adaptive shrinkage
            final_pred = shrinkage.predict(p_pred, ss_pred, n_obs)
            
            print(f"Subject {subject_id}, Observations {n_obs}: "
                  f"MSE = {np.mean((final_pred - y_sub) ** 2):.4f}")

if __name__ == "__main__":
    main() 