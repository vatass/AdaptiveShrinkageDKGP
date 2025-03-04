import torch
import gpytorch
from torch import nn

class SingleTaskDeepKernel(gpytorch.models.ExactGP):
    def __init__(self, input_dim, hidden_dim=64, output_dim=32):
        # Initialize likelihood
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super(SingleTaskDeepKernel, self).__init__(None, None, likelihood)
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # GP mean and covariance
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=output_dim)
        )

    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # GP layer
        mean = self.mean_module(features)
        covar = self.covar_module(features)
        
        return gpytorch.distributions.MultivariateNormal(mean, covar) 