import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import argparse
from sklearn.metrics import mean_absolute_error
from src.adaptive_shrinkage_dkgp.models.alpha_models import optimize_alpha_with_subject_simple

"""
Adaptive Shrinkage Estimator for combining population and subject-specific predictions.
"""
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

class AdaptiveShrinkage:
    """Adaptive Shrinkage Estimator.
    
    This class implements an adaptive shrinkage estimator that learns to
    optimally combine predictions from population and subject-specific models
    based on the number of observations and prediction uncertainties.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        subsample: float = 0.8,
        random_state: int = 42
    ) -> None:
        """Initialize the adaptive shrinkage estimator.
        
        Args:
            n_estimators: Number of trees in XGBoost
            learning_rate: Learning rate for XGBoost
            max_depth: Maximum tree depth
            subsample: Subsample ratio
            random_state: Random seed
        """
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=random_state
        )
    
    def _calculate_oracle_shrinkage(
        self,
        pop_pred: np.ndarray,
        ss_pred: np.ndarray,
        true_values: np.ndarray
    ) -> np.ndarray:
        """Calculate oracle shrinkage weights.
        
        Args:
            pop_pred: Population model predictions
            ss_pred: Subject-specific model predictions
            true_values: True target values
            
        Returns:
            Optimal shrinkage weights
        """
        def mse_loss(w: float) -> float:
            combined = w * pop_pred + (1 - w) * ss_pred
            return mean_squared_error(true_values, combined)
        
        weights = np.zeros(len(pop_pred))
        for i in range(len(pop_pred)):
            result = minimize(
                mse_loss,
                x0=0.5,
                bounds=[(0, 1)],
                method='L-BFGS-B'
            )
            weights[i] = result.x[0]
        
        return weights
    
    def fit(self, oracle_dataset: Dict[str, torch.Tensor]):
        """Fit the adaptive shrinkage model using the oracle dataset."""
        # Extract variables from the oracle dataset
        y_pp = oracle_dataset['y_pp']
        V_pp = oracle_dataset['V_pp']
        y_ss = oracle_dataset['y_ss']
        V_ss = oracle_dataset['V_ss']
        T_obs = oracle_dataset['T_obs']
        oracle_alpha = oracle_dataset['oracle_alpha']

        # Use the XGBRegressor model initialized in the constructor
        X = torch.cat([y_pp, V_pp, y_ss, V_ss, T_obs], dim=1).numpy()
        y = oracle_alpha.numpy()
        self.model.fit(X, y)
        print("Adaptive shrinkage model fitted using oracle dataset.")
    
    def predict(
        self,
        pop_pred: Union[np.ndarray, torch.Tensor],
        ss_pred: Union[np.ndarray, torch.Tensor],
        pop_var: Optional[Union[np.ndarray, torch.Tensor]] = None,
        ss_var: Optional[Union[np.ndarray, torch.Tensor]] = None,
        Tobs: Optional[Union[np.ndarray, torch.Tensor, float]] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """Predict optimal shrinkage weights and combine predictions.
        
        Args:
            pop_pred: Population model predictions
            ss_pred: Subject-specific model predictions
            pop_var: Optional population model prediction uncertainties
            ss_var: Optional subject-specific model prediction uncertainties
            Tobs: Optional time of last observation
            
        Returns:
            Combined predictions using learned shrinkage weights
        """
        # Check if inputs are tensors and convert to numpy if needed
        is_tensor = False
        device = None
        
        # Handle pop_pred
        if torch.is_tensor(pop_pred):
            is_tensor = True
            device = pop_pred.device
            pop_pred = pop_pred.detach().cpu().numpy()
            
        # Handle ss_pred
        if torch.is_tensor(ss_pred):
            is_tensor = True
            if device is None and hasattr(ss_pred, 'device'):
                device = ss_pred.device
            ss_pred = ss_pred.detach().cpu().numpy()
            
        # Handle pop_var
        if pop_var is not None and torch.is_tensor(pop_var):
            if device is None and hasattr(pop_var, 'device'):
                device = pop_var.device
            pop_var = pop_var.detach().cpu().numpy()
            
        # Handle ss_var
        if ss_var is not None and torch.is_tensor(ss_var):
            if device is None and hasattr(ss_var, 'device'):
                device = ss_var.device
            ss_var = ss_var.detach().cpu().numpy()
            
        # Handle Tobs
        if Tobs is not None and torch.is_tensor(Tobs):
            if device is None and hasattr(Tobs, 'device'):
                device = Tobs.device
            Tobs = Tobs.detach().cpu().numpy()
        
        # Prepare features
        features = []
        
        # Add population prediction (y_pp)
        features.append(pop_pred.reshape(-1, 1))
        
        # Add subject-specific prediction (y_ss)
        features.append(ss_pred.reshape(-1, 1))
        
        # Add population variance (V_pp) if available
        if pop_var is not None:
            features.append(pop_var.reshape(-1, 1))
        else:
            # If no variance, add zeros
            features.append(np.zeros_like(pop_pred.reshape(-1, 1)))
        
        # Add subject-specific variance (V_ss) if available
        if ss_var is not None:
            features.append(ss_var.reshape(-1, 1))
        else:
            # If no variance, add zeros
            features.append(np.zeros_like(ss_pred.reshape(-1, 1)))
            
        # Add time of last observation (Tobs) if available
        if Tobs is not None:
            # Convert scalar to array if needed
            if isinstance(Tobs, (int, float)):
                Tobs = np.array([Tobs] * len(pop_pred))
            features.append(Tobs.reshape(-1, 1))
        else:
            # If no Tobs, add zeros
            features.append(np.zeros_like(pop_pred.reshape(-1, 1)))
        
        # We use 5 features in this sequence: y_pp, y_ss, V_pp, V_ss, Tobs
        
        # features sequence should be y_pp, y_ss, V_pp, V_ss, Tobs
        X = np.hstack(features)
        print(f"Feature matrix shape: {X.shape}")
        
        # Predict weights
        adaptive_shrinkage_alpha = self.model.predict(X)
        print(f"Alpha shape: {adaptive_shrinkage_alpha.shape}")

        # take the mean of the adaptive_shrinkage_alpha
        adaptive_shrinkage_alpha = np.mean(adaptive_shrinkage_alpha)

        # Combine predictions
        personalized_pred = adaptive_shrinkage_alpha * pop_pred + (1 - adaptive_shrinkage_alpha) * ss_pred
        
        # Calculate combined variance
        if pop_var is not None and ss_var is not None:
            personalized_var = adaptive_shrinkage_alpha**2 * pop_var + (1 - adaptive_shrinkage_alpha)**2 * ss_var
        else:
            personalized_var = np.zeros_like(personalized_pred)
        
        # Convert back to tensor if input was tensor
        if is_tensor and device is not None:
            personalized_pred = torch.tensor(personalized_pred, device=device)
            personalized_var = torch.tensor(personalized_var, device=device)
            adaptive_shrinkage_alpha = torch.tensor(adaptive_shrinkage_alpha, device=device)
        
        return personalized_pred, personalized_var, adaptive_shrinkage_alpha
    
    def save_model(self, path: str) -> None:
        """Save the XGBoost model to disk.
        
        Args:
            path: Path to save the model
        """
        self.model.save_model(path)
    
    @classmethod
    def load_model(cls, path: str) -> 'AdaptiveShrinkage':
        """Load a saved XGBoost model from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded AdaptiveShrinkage model
        """
        model = cls()
        model.model.load_model(path)
        return model