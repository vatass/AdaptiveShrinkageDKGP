import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import pickle
import sys, os 
from os.path import exists
from operator import add
import argparse
import torch
import gpytorch 
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as pltmv 
import torch
import torch.nn as nn
import torch.optim as optim


def objective_within_subject_simple(alpha, y_g, y_p, y_t):
    combined_predictions = alpha * y_p + (1 - alpha) * y_t
    
    # Calculate squared errors
    errors = y_g - combined_predictions
    squared_errors = np.sum(errors**2)
    
    return squared_errors

def optimize_alpha_with_subject_simple(y, y_p, y_t):
    initial_alpha = 0.5  # Starting guess for alpha
    bounds = [(0, 1)]  # Alpha should be between 0 and 1

    result = minimize(
        objective_within_subject_simple,
        x0=initial_alpha,
        args=(y, y_p, y_t),
        bounds=bounds
    )

    optimal_alpha = result.x[0]
    print("Optimal alpha:", optimal_alpha)
    return optimal_alpha
