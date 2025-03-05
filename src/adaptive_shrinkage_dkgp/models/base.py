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
    def __init__(self, input_dim, hidden_dim=64, feature_dim=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(PopulationDKGP, self).__init__(
            train_x=None,  # Placeholder, should be set during training
            train_y=None,  # Placeholder, should be set during training
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
            train_x=None,  # Placeholder, should be set during training
            train_y=None,  # Placeholder, should be set during training
            likelihood=self.likelihood,
            depth=2,  # Adjust depth as needed
            dropout=0.1,  # Adjust dropout as needed
            activation='relu',  # Adjust activation as needed
            pretrained=False,  # Adjust based on your needs
            latent_dim=feature_dim,
            feature_extractor=None,  # Adjust based on your needs
            gphyper=None  # Adjust based on your needs
        ).to(device)


class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, datadim, depth, dr, activ):
        super(LargeFeatureExtractor, self).__init__()

        # Deep Kernel Architecture
        self.datadim = datadim
        self.depth = depth
        self.activation_function = activ
        self.droupout_rate = dr
        print('depth', depth)
        final_layer = depth[-1]
        print('final layer', final_layer)

        for i, d in enumerate(self.depth[:-1]):
            dim1, dim2 = d
            self.add_module(f'linear{i+1}', torch.nn.Linear(dim1, dim2))
            if self.activation_function == 'relu':
                self.add_module(f'activ{i+1}', torch.nn.ReLU())
            elif self.activation_function == 'leakyr':
                self.add_module(f'activ{i+1}', torch.nn.LeakyReLU())
            elif self.activation_function == 'prelu':
                self.add_module(f'activ{i+1}', torch.nn.PReLU())
            elif self.activation_function == 'selu':
                self.add_module(f'activ{i+1}', torch.nn.SELU())

        print('Final Layer', final_layer[0], final_layer[1])
        self.add_module('final_linear', torch.nn.Linear(int(final_layer[0]), int(final_layer[1])))
        self.add_module('dr1', torch.nn.Dropout(self.droupout_rate))


