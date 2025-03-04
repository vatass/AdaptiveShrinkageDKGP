import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import argparse
from sklearn.metrics import mean_absolute_error
from src.adaptive_shrinkage_dkgp.models.alpha_models import optimize_alpha_with_subject_simple

# '''
# In this script we use the validation subjects to train the XGBoost functions 
# '''

# ### CUDA-PYTORCH CHECK ###
# import torch
# print("PyTorch Version:", torch.__version__)
# print("CUDA Version PyTorch Sees:", torch.version.cuda)
# print("cuDNN Version:", torch.backends.cudnn.version())

# ### CUDA CHECK ###
# import torch
# print("CUDA Available:", torch.cuda.is_available())
# print("Number of CUDA Devices:", torch.cuda.device_count())
# # print("CUDA Device Name:", torch.cuda.get_device_name(0))


# gpu_id = 0
# fontsize = 19

# parser = argparse.ArgumentParser(description='Alpha-GP Modeling')
# parser.add_argument("--learning_rate", help='Learning Rate', type=float, default=0.01844)   
# parser.add_argument("--datasets", help="GPUs", default='1adniblsa')
# parser.add_argument("--alpha_optim", help='Alpha Optimization Option', default='alpha_simple') # choices are alpha_simple, alpha_aug, alpha_aug2
# parser.add_argument("--alpha_function", help='Alpha Input Space Function', default='predvar') # choices are predvartobs, deviations of pred var with Tobs


# # wandb.init(project="AlphaModeling", entity="vtassop", save_code=True)

# ## Visualization script for the mixed effects idea for personalization ## 
# resultdir = '/home/cbica/Desktop/LongGPClustering'

# args = parser.parse_args()
# lr = args.learning_rate
# datasets = args.datasets
# alpha_optim = args.alpha_optim
# alpha_function = args.alpha_function 

# roi_names = ['Right Hippocampus','Right Thalamus Proper', 'Right Lateral Ventricle', 'Left Hippocampus', 'Right Amygdala', 'Left Amygdala', 'Right PHG']
# roi_idxs = [ 13, 23, 17, 14, 4, 5, 109]
# rois_numbers = [ 47, 59, 51]

# longitudinal_covariates = pd.read_csv(resultdir + '/data2/longitudinal_covariates_subjectsamples_longclean_hmuse_convs_adniblsa.csv')
# print(longitudinal_covariates.head(10))
# for c in longitudinal_covariates.columns: 
#     print(c, longitudinal_covariates[c].unique())

# for idx, task in enumerate(roi_idxs): 

#     print('Task', roi_names[idx])
#     pers_s = pd.read_csv('./newneuripsresults/person_ss_singletask_' + str(roi_idxs[idx]) + '_dkgp_population_'+ datasets +'.csv')
#     pers_ss = pers_s[pers_s['model']!='Population']
#     pop = pd.read_csv( './neuripsresults/singletask_' + str(roi_idxs[idx]) + '_dkgp_population_'+ datasets +'.csv')
#     newpop = {'id': [], 'kfold': [], 'score': [], 'lower': [], 'upper': [], 'variance': [], 'y': [], 'history': []}
#     for h in [4, 5, 6, 7]:
#         newpop['id'].extend(pop['id'].tolist())
#         newpop['kfold'].extend(pop['kfold'].tolist())
#         newpop['score'].extend(pop['score'].tolist())
#         newpop['lower'].extend(pop['lower'].tolist())
#         newpop['upper'].extend(pop['upper'].tolist())
#         newpop['variance'].extend(pop['variance'].tolist())
#         newpop['y'].extend(pop['y'].tolist())
#         newpop['history'].extend([h for k in range(len(pop['y'].tolist()))])
    
#     pop = pd.DataFrame(data=newpop)
#     unique_subjects_ss = list(pers_ss['id'].unique())

#     #  split unique_subjects_ss into train and test 
#     # This gives me 72 Train Subjects and the rest are for Test the DME-GP
#     validation_subjects = unique_subjects_ss[:int(0.3*len(unique_subjects_ss))]
#     test_subjects = unique_subjects_ss[int(0.3*len(unique_subjects_ss)):]
#     print('Train Subjects', len(validation_subjects))
#     print('Test Subjects', len(test_subjects))

#     #### Alphas Dataset Creation ####
#     ### Create it on the unseen information ###
#     y_ss_list, y_p_list, V_ss_list,V_pp_list, Tobs_list, corr_ids_list = [], [], [], [], [], [] 
#     y_list = [] 

#     X, Y_simple, Y_aug, Y_aug2 = [], [], [], []

#     df_alphas_within = {'id': [], 'alpha_simple': [], 'alpha_aug': [], 'alpha_aug2': [], 'y_ss': [], 'y_pp': [], 'y': [], 'V_ss': [], 'V_pp': [], 'Tobs': []}

#     for val_subject in validation_subjects: 

#         pers_ss_sub = pers_ss[pers_ss['id'] == val_subject]
#         pop_sub = pop[pop['id'] == val_subject]

#         for h in [4,5,6,7]:
#             pers_ss_sub_h = pers_ss_sub[pers_ss_sub['history_points'] == h]
#             pop_sub_h = pop_sub[pop_sub['history'] == h]
#             y_ss_list.extend(pers_ss_sub_h['score'].tolist())
#             y_p_list.extend(pop_sub_h['score'].tolist())
#             y_list.extend(pers_ss_sub_h['y'].tolist())
#             V_ss_list.extend(pers_ss_sub_h['variance'].tolist())
#             V_pp_list.extend(pop_sub_h['variance'].tolist())
            
#             ### Time of observation ###
#             Tobs = pers_ss_sub_h['time'].tolist()[h-1]
#             ## Populate this for all the samples in the trajectory 
#             Tobs_list_within = [Tobs for k in range(len(pers_ss_sub_h['score'].tolist()))]
#             Tobs_list.extend(Tobs_list_within)
#             # store the validation subject id 
#             corr_ids_list.extend(pers_ss_sub_h['id'].tolist()) 

#             y_ss = pers_ss_sub_h['score'].tolist()
#             y_p = pop_sub_h['score'].tolist()
#             y = pers_ss_sub_h['y'].tolist()
#             V_ss = pers_ss_sub_h['variance'].tolist()
#             V_pp = pop_sub_h['variance'].tolist()

#             y_arr = np.array(y)
#             y_ss_arr = np.array(y_ss)
#             y_pp_arr = np.array(y_p)
#             V_ss_arr = np.array(V_ss)
#             V_pp_arr = np.array(V_pp)
#             Tobs = np.array(Tobs_list)

#             ## I want to stack in the X array the y_ss, y_pp, V_ss, V_pp, Tobs
#             Xarr = np.array([y_ss_arr, y_pp_arr, V_ss_arr, V_pp_arr, Tobs]).T
#             X.append(Xarr)

#             alphas_simple =  optimize_alpha_with_subject_simple(y_arr, y_pp_arr, y_ss_arr)
  
#             Y_simple.append(alphas_simple)


#             df_alphas_within['id'].extend([val_subject for k in range(len(y_ss_arr))])
#             df_alphas_within['alpha_simple'].extend([alphas_simple for k in range(len(y_ss_arr))])
#             df_alphas_within['y_ss'].extend(y_ss_arr)
#             df_alphas_within['y_pp'].extend(y_pp_arr)
#             df_alphas_within['y'].extend(y_arr)
#             df_alphas_within['V_ss'].extend(V_ss_arr)
#             df_alphas_within['V_pp'].extend(V_pp_arr)
#             df_alphas_within['Tobs'].extend(Tobs_list_within)

#     X_within = np.array(X)
#     Y_simple = np.array(Y_simple)

#     print(X_within.shape)

#     df_alphas_within = pd.DataFrame(df_alphas_within)
#     # store it to csv
#     df_alphas_within.to_csv('./neuripsresults/alphas_comparison_within_'+ str(roi_idxs[idx]) + '_'+ datasets +'.csv')

#     X = X_within
#     Y = Y_simple
   
#     # Define the features and target
#     features = ['y_ss', 'y_pp', 'V_ss', 'V_pp', 'Tobs']
#     target = alpha_optim  # assuming the target column name is the same as t_type

#     X = df_alphas_within[features]
#     Y = df_alphas_within[target]

#     ### Keep Held Out Rows for  Alpha Model Testing ###
#     X_train = X[:int(0.9*len(X))]
#     y_train = Y[:int(0.9*len(Y))]
#     X_test = X[int(0.9*len(X)):]
#     y_test = Y[int(0.9*len(Y)):]
#     print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


#     xgb_model = XGBRegressor(n_estimators=100, random_state=42)
#     xgb_model.fit(X_train, y_train)

#         # Save the model to a file
#     xgb_model.save_model('xgb_model.json')

#     # Create a new model instance and load the model from the file
#     loaded_model = XGBRegressor()
#     loaded_model.load_model('xgb_model.json')

#     # Use the loaded model for predictions on the test set
#     test_predictions = loaded_model.predict(X_test)
#     test_mae = mean_absolute_error(y_test, test_predictions)
#     print('XGBOOST MAE:', test_mae)

#     xgb_model.save_model('./xgb_model_'+ str(alpha_optim) + '_' + str(roi_idxs[idx]) + '.json')


# print('Alpha XGB Training Completed!')

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
    
    def fit(
        self,
        pop_pred: Union[np.ndarray, torch.Tensor],
        ss_pred: Union[np.ndarray, torch.Tensor],
        true_values: Union[np.ndarray, torch.Tensor],
        n_obs: Union[np.ndarray, torch.Tensor],
        pop_std: Optional[Union[np.ndarray, torch.Tensor]] = None,
        ss_std: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> None:
        """Train the adaptive shrinkage estimator.
        
        Args:
            pop_pred: Population model predictions
            ss_pred: Subject-specific model predictions
            true_values: True target values
            n_obs: Number of observations for each prediction
            pop_std: Optional population model prediction uncertainties
            ss_std: Optional subject-specific model prediction uncertainties
        """
        # Convert to numpy if needed
        if torch.is_tensor(pop_pred):
            pop_pred = pop_pred.cpu().numpy()
        if torch.is_tensor(ss_pred):
            ss_pred = ss_pred.cpu().numpy()
        if torch.is_tensor(true_values):
            true_values = true_values.cpu().numpy()
        if torch.is_tensor(n_obs):
            n_obs = n_obs.cpu().numpy()
        if pop_std is not None and torch.is_tensor(pop_std):
            pop_std = pop_std.cpu().numpy()
        if ss_std is not None and torch.is_tensor(ss_std):
            ss_std = ss_std.cpu().numpy()
        
        # Calculate oracle shrinkage weights
        oracle_weights = self._calculate_oracle_shrinkage(
            pop_pred, ss_pred, true_values
        )
        
        # Prepare features for XGBoost
        features = [n_obs.reshape(-1, 1)]
        if pop_std is not None:
            features.append(pop_std.reshape(-1, 1))
        if ss_std is not None:
            features.append(ss_std.reshape(-1, 1))
        
        X = np.hstack(features)
        
        # Train XGBoost model
        self.model.fit(X, oracle_weights)
    
    def predict(
        self,
        pop_pred: Union[np.ndarray, torch.Tensor],
        ss_pred: Union[np.ndarray, torch.Tensor],
        n_obs: Union[np.ndarray, torch.Tensor],
        pop_std: Optional[Union[np.ndarray, torch.Tensor]] = None,
        ss_std: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """Predict optimal shrinkage weights and combine predictions.
        
        Args:
            pop_pred: Population model predictions
            ss_pred: Subject-specific model predictions
            n_obs: Number of observations
            pop_std: Optional population model prediction uncertainties
            ss_std: Optional subject-specific model prediction uncertainties
            
        Returns:
            Combined predictions using learned shrinkage weights
        """
        # Convert to numpy if needed
        is_tensor = torch.is_tensor(pop_pred)
        if is_tensor:
            pop_pred = pop_pred.cpu().numpy()
            ss_pred = ss_pred.cpu().numpy()
            n_obs = n_obs.cpu().numpy()
            if pop_std is not None:
                pop_std = pop_std.cpu().numpy()
            if ss_std is not None:
                ss_std = ss_std.cpu().numpy()
        
        # Prepare features
        features = [n_obs.reshape(-1, 1)]
        if pop_std is not None:
            features.append(pop_std.reshape(-1, 1))
        if ss_std is not None:
            features.append(ss_std.reshape(-1, 1))
        
        X = np.hstack(features)
        
        # Predict weights
        weights = self.model.predict(X)
        
        # Combine predictions
        combined = weights * pop_pred + (1 - weights) * ss_pred
        
        # Convert back to tensor if input was tensor
        if is_tensor:
            combined = torch.tensor(combined, device=pop_pred.device)
        
        return combined
    
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