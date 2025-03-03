"""
Base classes for Deep Kernel Gaussian Process models.
"""
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import gpytorch
import math

from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution, LMCVariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood

from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL

from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import MultitaskGaussianLikelihood

from gpytorch.models import ExactGP

class BaseDeepKernel(ExactGP):
    """Base class for Deep Kernel Gaussian Process models.
    
    This class implements the basic structure of a Deep Kernel GP model,
    combining a neural network feature extractor with a GP layer.
    
    Attributes:
        feature_extractor: Neural network for feature extraction
        mean_module: GP mean function
        covar_module: GP covariance function
        likelihood: GP likelihood function
    """
    
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        input_dim: int,
        latent_dim: int,
        depth: int = 2,
        dropout: float = 0.1,
        activation: str = 'relu',
        kernel: str = 'RBF',
        mean: str = 'constant',
        pretrained_extractor: Optional[nn.Module] = None,
        gp_hyperparams: Optional[Dict] = None
    ) -> None:
        """Initialize the Deep Kernel GP model.
        
        Args:
            train_x: Training input data
            train_y: Training target data
            likelihood: GP likelihood function
            input_dim: Dimension of input features
            latent_dim: Dimension of latent space
            depth: Number of layers in feature extractor
            dropout: Dropout rate
            activation: Activation function ('relu', 'tanh', etc.)
            kernel: Kernel function ('RBF' or 'Matern')
            mean: Mean function ('constant' or 'linear')
            pretrained_extractor: Optional pretrained feature extractor
            gp_hyperparams: Optional GP hyperparameters
        """
        super().__init__(train_x, train_y, likelihood)
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Feature extractor
        if pretrained_extractor is not None:
            self.feature_extractor = pretrained_extractor
        else:
            self.feature_extractor = self._build_feature_extractor(
                input_dim, latent_dim, depth, dropout, activation
            )
        
        # GP components
        self.mean_module = (
            LinearMean(input_size=latent_dim)
            if mean.lower() == 'linear'
            else ConstantMean()
        )
        
        kernel_cls = MaternKernel if kernel.upper() == 'MATERN' else RBFKernel
        self.covar_module = ScaleKernel(
            kernel_cls(ard_num_dims=latent_dim)
        )
        
        # Scale neural network outputs
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)
        
        # Initialize GP hyperparameters if provided
        if gp_hyperparams is not None:
            self.initialize(**gp_hyperparams)
    
    def _build_feature_extractor(
        self,
        input_dim: int,
        latent_dim: int,
        depth: int,
        dropout: float,
        activation: str
    ) -> nn.Module:
        """Build the neural network feature extractor.
        
        Args:
            input_dim: Input dimension
            latent_dim: Output dimension
            depth: Number of hidden layers
            dropout: Dropout rate
            activation: Activation function name
        
        Returns:
            Neural network module
        """
        layers = []
        dims = [input_dim] + [max(latent_dim, input_dim // (2**i)) for i in range(depth)]
        
        for i in range(len(dims)-1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                getattr(nn, activation.upper())(),
                nn.Dropout(dropout)
            ])
        
        # Final layer to latent dimension
        layers.append(nn.Linear(dims[-1], latent_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            GP posterior distribution
        """
        # Extract features
        features = self.feature_extractor(x)
        features = self.scale_to_bounds(features)
        
        # GP layer
        mean_x = self.mean_module(features)
        covar_x = self.covar_module(features)
        
        return MultivariateNormal(mean_x, covar_x)
    
    def predict(
        self,
        x: torch.Tensor,
        return_std: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Make predictions with the model.
        
        Args:
            x: Input tensor
            return_std: Whether to return predictive standard deviation
            
        Returns:
            Mean predictions and optionally standard deviations
        """
        self.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self(x)
            mean = posterior.mean
            
            if return_std:
                std = posterior.stddev
                return mean, std
            return mean


