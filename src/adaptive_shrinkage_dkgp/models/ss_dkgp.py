"""
Subject-Specific Deep Kernel Gaussian Process model.
"""
from typing import Dict, Optional, Tuple, Union
import torch
import gpytorch
from .base import BaseDeepKernel, LargeFeatureExtractor

class SubjectSpecificDKGP(BaseDeepKernel):
    """Subject-Specific Deep Kernel Gaussian Process model.
    
    This model is initialized with parameters from a population model
    and fine-tuned on subject-specific data.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        input_dim: int,
        latent_dim: int,
        population_params: Optional[Dict[str, torch.Tensor]] = None,
        depth: int = 2,
        dropout: float = 0.1,
        activation: str = 'relu',
        kernel: str = 'RBF',
        mean: str = 'constant',
        learning_rate: float = 0.0184,
        weight_decay: float = 0.01,
        n_epochs: int = 400,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
        """Initialize the subject-specific DKGP model.
        
        Args:
            train_x: Training input data
            train_y: Training target data
            input_dim: Dimension of input features
            latent_dim: Dimension of latent space
            population_params: Optional parameters from population model
            depth: Number of layers in feature extractor
            dropout: Dropout rate
            activation: Activation function ('relu', 'tanh', etc.)
            kernel: Kernel function ('RBF' or 'Matern')
            mean: Mean function ('constant' or 'linear')
            learning_rate: Learning rate for optimization
            n_epochs: Number of training epochs
            device: Device to run the model on
        """
        # Initialize likelihood
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        
        super().__init__(
            train_x=train_x,
            train_y=train_y,
            likelihood=likelihood,
            input_dim=input_dim,
            latent_dim=latent_dim,
            depth=depth,
            dropout=dropout,
            activation=activation,
            kernel=kernel,
            mean=mean
        )
        # Ensure consistent layer naming
        self.feature_extractor = LargeFeatureExtractor(
            datadim=input_dim,
            depth=[(input_dim, int(input_dim/2))],  
            dr=dropout,
            activ=activation
        )
        
        if population_params is not None:
            adjusted_population_params = {k.replace('feature_extractor.', ''): v for k, v in population_params.items()}
            self.feature_extractor.load_state_dict(adjusted_population_params)
            # Freeze feature extractor parameters
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.device = device
        self.weight_decay = weight_decay
        self.to(device)
    
    def fit(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        verbose: bool = True,
        patience: int = 10,  # Παράμετρος για early stopping
        min_delta: float = 1e-4  # Ελάχιστη βελτίωση για να θεωρηθεί πρόοδος
    ) -> Dict[str, list]:
        """Train the subject-specific DKGP model.
        
        Args:
            train_x: Training input data
            train_y: Training target data
            verbose: Whether to print training progress
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change in loss to qualify as improvement
            
        Returns:
            Dictionary containing training history
        """
        self.train()
        
        # Ensure data is on the correct device
        train_x = train_x.to(self.device)
        train_y = train_y.to(self.device)
        
        # Only optimize GP parameters, feature extractor remains frozen
        optimizer = torch.optim.Adam([
            {'params': self.covar_module.parameters(), 'lr': self.learning_rate},
            {'params': self.mean_module.parameters(), 'lr': self.learning_rate},
            {'params': self.likelihood.parameters(), 'lr': self.learning_rate}
        ], weight_decay=self.weight_decay)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=verbose
        )
        
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        
        history = {
            'train_loss': [],
        }
        
        # Early stopping variables
        best_loss = float('inf')
        no_improve_epochs = 0
        
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            output = self(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            current_loss = loss.item()
            history['train_loss'].append(current_loss)
            
            # Update learning rate based on loss
            scheduler.step(current_loss)
            
            # Early stopping check
            if current_loss < best_loss - min_delta:
                best_loss = current_loss
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                
            if no_improve_epochs >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f'Epoch {epoch+1}/{self.n_epochs} - Training Loss: {current_loss:.4f}')

        return history
    
    def unfreeze_feature_extractor(self) -> None:
        """Unfreeze the feature extractor parameters for fine-tuning."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
    
    def save_model(self, path: str) -> None:
        """Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'likelihood_state_dict': self.likelihood.state_dict(),
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim
        }, path)
    
    @classmethod
    def load_model(
        cls,
        path: str,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> 'SubjectSpecificDKGP':
        """Load a saved model from disk.
        
        Args:
            path: Path to the saved model
            train_x: Training input data
            train_y: Training target data
            device: Device to load the model on
            
        Returns:
            Loaded SubjectSpecificDKGP model
        """
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            train_x=train_x,
            train_y=train_y,
            input_dim=checkpoint['input_dim'],
            latent_dim=checkpoint['latent_dim'],
            device=device
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        
        return model



