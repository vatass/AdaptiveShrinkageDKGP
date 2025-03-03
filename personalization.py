import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import argparse
from functions import mae, mse, R2, calc_coverage

'''

Personalization of Predicted Trajectories Through Adaptive Shrinkage Estimation
This script loads the population predictions, the subject-specific predictions and the XGBoost model 
to infer the optimal alpha parameter and produces the results for the final personalization product
'''

parser = argparse.ArgumentParser(description='Personalization on Biomarkers')
parser.add_argument("--biomarker", help='Biomarker I want to exctrapolate', type=str, default='Right Hippocampus')   
parser.add_argument("--roi_idx", help='Index of the ROI', type=str, default=13)   
parser.add_argument("--dataset", help='What is the dataset', type=str, default='1adniblsa')   
parser.add_argument("--alpha_model", help='Path to alpha model. This is where the data are too', type=str, default='./neuripsresults/xgb_alpha_simple_13.json')
parser.add_argument("--alpha_optim", help='Alpha Optimization Choice', type=str, default='alpha_simple') # alpha_simple or alpha_aug

args = parser.parse_args()
biomarker = args.biomarker
roi_idx = args.roi_idx
datasets = args.dataset
alpha_model = args.alpha_model
alpha_optim = args.alpha_optim


### Load the XGBoost Model ###
# Create a new model instance
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.load_model('./neuripsresults/xgb_model_'+ alpha_optim + '_' + roi_idx +'.json')

df_predictions = {'time': [], 'y': [], 'score': [],  'upper': [], 'lower': [], 'History': [], 'Model': [], 'alpha': [], 'Alpha-Model': [], 'id': []}
df_metrics_history = {'id': [], 'time': [], 'history': [], 'ae': [], 'se': [], 'model': [], 'interval': [], 'coverage': [] , 'Alpha-Model': [], 'Tobs': [] } 
df_mean_metrics = {'id': [], 'history': [], 'mse': [], 'mae': [], 'rmse': [], 'r2': [], 'model': [], 'coverage': [], 'interval': [], 'Alpha-Model': []}     

print('Task', biomarker)
# 4,5,6,7
pers_s = pd.read_csv('./newneuripsresults/person_ss_singletask_' + str(roi_idx) + '_dkgp_population_'+ datasets +'.csv')
pers_ss = pers_s[pers_s['model']!='Population']
pop = pd.read_csv( './neuripsresults/singletask_' + str(roi_idx) + '_dkgp_population_'+ datasets +'.csv')
newpop = {'id': [], 'kfold': [], 'score': [], 'lower': [], 'upper': [], 'variance': [], 'y': [], 'history': []}
for h in [3, 4, 5, 6,7]:
    newpop['id'].extend(pop['id'].tolist())
    newpop['kfold'].extend(pop['kfold'].tolist())
    newpop['score'].extend(pop['score'].tolist())
    newpop['lower'].extend(pop['lower'].tolist())
    newpop['upper'].extend(pop['upper'].tolist())
    newpop['variance'].extend(pop['variance'].tolist())
    newpop['y'].extend(pop['y'].tolist())
    newpop['history'].extend([h for k in range(len(pop['y'].tolist()))])

pop = pd.DataFrame(data=newpop)

test_subjects = pers_ss['id'].unique()
alpha_exploration_dict  = {'alpha': [], 'y_ss': [], 'y_pp': [], 'V_ss': [], 'V_pp': [], 'Tobs': []}

### For the test subjects infer the alpha
for t in test_subjects: 

    ### I need to store the variance ### 
    pers_subject = pers_ss[pers_ss['id'] == t]
    pop_subject = pop[pop['id'] == t]

    for h in [4, 5,6,7]:

        pers_subject_h = pers_subject[pers_subject['history_points'] == h]
        pop_subject_h = pop_subject[pop_subject['history'] == h]  

        y_ss = pers_subject_h[pers_subject_h['id'] == t]['score'].tolist()
        y = pers_subject_h[pers_subject_h['id'] == t]['y'].tolist() 
        y_pp = pop_subject_h[pop_subject_h['id'] == t]['score'].tolist()
        V_pp = pop_subject_h[pop_subject_h['id'] == t]['variance'].tolist()
        V_ss = pers_subject_h[pers_subject_h['id'] == t]['variance'].tolist()
        time_ = pers_subject_h[pers_subject_h['id'] == t]['time'].tolist()
        time_obs = time_[:h]
        Tobs = time_[h-1]
        time_unseen = time_[h:]

        ## Whole Trajectory ##
        y_ss_unobs = pers_subject_h[pers_subject_h['id'] == t]['score'].tolist()
        y_pp_unobs = pop_subject_h[pop_subject_h['id'] == t]['score'].tolist()
        V_pp_unobs = pop_subject_h[pop_subject_h['id'] == t]['variance'].tolist()
        V_ss_unobs = pers_subject_h[pers_subject_h['id'] == t]['variance'].tolist()
        Tobs_list = [Tobs for k in range(len(y_ss_unobs))]
        
        print(len(y_ss_unobs), len(y_pp_unobs), len(V_pp_unobs), len(V_ss_unobs), len(Tobs_list))

        history = pers_ss[pers_ss['id'] == t]['history_points'].tolist()[:h]

        # calculate y_pp_unobs - y_ss_unobs
        y_diff = [a-b for a,b in zip(y_pp_unobs,y_ss_unobs)]
        v_diff = [a-b for a,b in zip(V_pp_unobs,V_ss_unobs)]

        Tobs_np = np.array([Tobs]*len(y_ss_unobs), dtype=np.float32)

        x_test = np.array([y_ss_unobs, y_pp_unobs, V_ss_unobs, V_pp_unobs, Tobs_list], dtype=np.float32).T
        # elif alpha_function == 'dev':
        #     x_test = np.array([y_diff, v_diff,Tobs_list], dtype=np.float32).T

        print('Input to the Oracle', x_test.shape)

        alpha_pred = xgb_model.predict(x_test)

        ### Alpha Exploration for XGBoost function Explainability ###
        alpha_exploration_dict['alpha'].extend(alpha_pred.tolist())
        alpha_exploration_dict['y_ss'].extend(y_ss_unobs)
        alpha_exploration_dict['y_pp'].extend(y_pp_unobs)
        alpha_exploration_dict['V_ss'].extend(V_ss_unobs)
        alpha_exploration_dict['V_pp'].extend(V_pp_unobs)
        alpha_exploration_dict['Tobs'].extend(Tobs_list)


        alpha_pred = np.mean(alpha_pred)

        alpha_pred_mean = np.mean(alpha_pred)


        #### FORMULA ::: y_dme = alpha * y_pp + (1 - alpha) * y_ss ####
        ### Calculate the combined prediction and variance ###
        alpha_pred = np.broadcast_to(alpha_pred, len(y_pp_unobs))
        y_dme = y_pp_unobs * alpha_pred + y_ss_unobs * (1 - alpha_pred)

        # Calculate combined variance
        alpha_pred_squared = alpha_pred ** 2
        V_c = V_pp_unobs * alpha_pred_squared + V_ss_unobs * (1 - alpha_pred) ** 2

        std_c = np.sqrt(V_c)

        # Calculate 95% confidence intervals
        lower_dme= y_dme - 1.96 * std_c
        upper_dme = y_dme + 1.96 * std_c

        ### Convert all the trajectory !!!!
        df_predictions['id'].extend([t for k in range(len(y_dme))])
        df_predictions['time'].extend(time_)
        df_predictions['y'].extend(y)
        df_predictions['score'].extend(y_dme.tolist())
        df_predictions['History'].extend([h for k in range(len(y_dme))])
        df_predictions['upper'].extend(upper_dme.tolist())
        df_predictions['lower'].extend(lower_dme.tolist())
        df_predictions['Model'].extend(['DME' for k in range(len(y_dme))])
        df_predictions['alpha'].extend([alpha_pred_mean for k in range(len(y_dme))])
        df_predictions['Alpha-Model'].extend(['Alpha-XGB' for k in range(len(y_dme))])
        
        ### Calculate the Evaluation Metrics on the Unseen Trajectory 
        y_unseen = y[h:]
        y_dme_pred = y_dme[h:]
        upper_dme_pred = upper_dme[h:]
        lower_dme_pred = lower_dme[h:]

        mae_pers, ae_pers = mae(np.array(y_unseen), np.array(y_dme_pred))

        print('MAE')
        print(mae_pers)
        print('AE')
        print(ae_pers)


        mse_pers, rmse_pers, se_pers = mse(np.array(y_unseen), np.array(y_dme_pred))  
        rsq = R2(np.array(y_unseen),  np.array(y_dme_pred)) 

        coverage, interval_width, mean_coverage, mean_interval_width  = calc_coverage(predictions=np.array(y_dme_pred), groundtruth=np.array(y_unseen),
        intervals=[np.array(lower_dme_pred), np.array(upper_dme_pred)])  

        coverage, interval_width, mean_coverage, mean_interval_width = coverage.numpy().astype(int), interval_width.numpy(), mean_coverage.numpy(), mean_interval_width.numpy() 

        ####
        df_metrics_history['id'].extend([t for p in range(ae_pers.shape[0])])
        df_metrics_history['time'].extend(time_unseen)
        df_metrics_history['history'].extend([h for p in range(ae_pers.shape[0])])
        df_metrics_history['Tobs'].extend([Tobs for p in range(ae_pers.shape[0])])
        df_metrics_history['ae'].extend(ae_pers.tolist())
        df_metrics_history['se'].extend(se_pers.tolist())
        df_metrics_history['model'].extend(['DME-GP' for p in range(ae_pers.shape[0])])
        df_metrics_history['interval'].extend(interval_width.tolist())
        df_metrics_history['coverage'].extend(coverage.tolist())
        df_metrics_history['Alpha-Model'].extend(['Alpha-XGB' for k in range(len(time_unseen))])

        df_mean_metrics['id'].append(t)
        df_mean_metrics['history'].append(h)
        df_mean_metrics['mse'].append(mse_pers)
        df_mean_metrics['mae'].append(mae_pers)
        df_mean_metrics['rmse'].append(rmse_pers)
        df_mean_metrics['r2'].append(rsq)
        df_mean_metrics['model'].append('DME-GP')
        df_mean_metrics['coverage'].append(np.mean(coverage))
        df_mean_metrics['interval'].append(np.mean(interval_width))
        df_mean_metrics['Alpha-Model'].append(['Alpha-XGB' for k in range(len(y_dme_pred))])

df_predictions = pd.DataFrame(df_predictions)
df_metrics_history = pd.DataFrame(df_metrics_history)
df_mean_metrics = pd.DataFrame(df_mean_metrics)
alpha_exploration_df = pd.DataFrame(alpha_exploration_dict)

alpha_exploration_df.to_csv('./neuripsrebut/deepmixedeffects_xgb_alpha_exploration_'+alpha_optim+'_dkgp_' + str(roi_idx) + '_'+ datasets +'.csv')
# df_predictions.to_csv('./neuripsrebut/deepmixedeffects_xgb_predictions_'+alpha_optim+'_dkgp_' + str(roi_idx) + '_'+ datasets +'.csv')
# df_metrics_history.to_csv('./neuripsrebut/deepmixedeffects_xgb_metrics_history_'+alpha_optim+'_dkgp_' + str(roi_idx) + '_'+ datasets +'.csv')
# df_mean_metrics.to_csv('./neuripsrebut/deepmixedeffects_xgb_mean_metrics_'+alpha_optim+'_dkgp_' + str(roi_idx) + '_'+ datasets +'.csv')
