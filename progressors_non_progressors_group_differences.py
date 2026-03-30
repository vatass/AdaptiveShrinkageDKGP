# ==============================================================================
# Experiment: Predicted Trajectories Highlight Consistent Group Differences
#             Between Progressors and Non-Progressors
# ==============================================================================
#
# OVERVIEW
# --------
# This script tests whether GP-predicted longitudinal brain ROI trajectories
# preserve the group-level signal that separates subjects who convert to
# Alzheimer's Disease (Progressors) from those who remain stable
# (Non-Progressors). The analysis is run in parallel on real (observed) and
# predicted volumes, so that results can be compared directly.
#
# MOTIVATION
# ----------
# The DKGP model predicts future brain trajectories from a subject's observed
# history. A key validation question is: do predicted trajectories retain
# biologically meaningful group differences? If the Status × Time interaction
# is significant in both the real and the predicted data, this supports the
# claim that the model captures clinically relevant longitudinal dynamics —
# not just average volume levels, but the diverging rates of atrophy between
# groups over time.
#
# EXPERIMENTAL DESIGN
# -------------------
# For each of 7 AD-sensitive brain ROIs, a Linear Mixed Model (LMM) is fit
# separately to real and predicted volumes:
#
#   ROI ~ age_baseline + sex + Status + time + Status × time
#
# where Status is binary (Progressor / Non-Progressor) and subjects are
# treated as the random-effects grouping variable. The key term of interest
# is the Status × Time interaction: a significant interaction indicates that
# the two groups diverge at different rates over the follow-up period.
#
# The same model is fit twice per ROI — once on real volumes (column "y")
# and once on DKGP-predicted volumes (column "score") — so that we can
# assess whether the predicted trajectories mirror the real signal.
#
# DATA
# ----
# - Subjects        : ADNI cohort, filtered to those with DKGP predictions
# - Progressors     : subjects who converted from MCI to AD during follow-up
# - Non-Progressors : subjects who remained MCI-stable (non-converters)
# - ROI features    : 7 AD-sensitive MUSE volumetric regions
#                     (hippocampus, amygdala, OFuG, PHG, thalamus)
# - Covariates      : baseline age, sex (fixed effects); subject (random)
#
# VISUALISATIONS
# --------------
# For each ROI, four spaghetti plots are produced:
#   1. Real volumes      vs time — individual trajectories + LMM fit
#   2. Real volumes      vs age  — individual trajectories + LMM fit
#   3. Predicted volumes vs time — individual trajectories + LMM fit
#   4. Predicted volumes vs age  — individual trajectories + LMM fit
#
# MULTIPLE COMPARISONS
# --------------------
# Because the Status × Time interaction is tested across 7 ROIs, Bonferroni
# correction is applied (α_corrected = 0.05 / 7 ≈ 0.0071). Results are
# reported both before and after correction.
#
# OUTPUTS
# -------
# - ./manuscript1/LMM_results_bonferroni.txt
#       Full LMM summaries for every ROI × data type, plus the Bonferroni
#       correction table and manuscript-ready summary
# - ./manuscript1/real_staging_plot_{ROI}.{png,svg}
#       Real volume spaghetti + LMM trajectory (time axis)
# - ./nataging/real_age_staging_plot_{ROI}.{png,svg}
#       Real volume spaghetti + LMM trajectory (age axis)
# - ./manuscript1/predicted_staging_plot_{ROI}.{png,svg}
#       Predicted volume spaghetti + LMM trajectory (time axis)
# - ./manuscript1/predicted_age_staging_plot_{ROI}.{png,svg}
#       Predicted volume spaghetti + LMM trajectory (age axis)
# ==============================================================================
 

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import pickle
import sys, os 
from os.path import exists
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from operator import add
import argparse
from functions import process_temporal_singletask_data
import statsmodels.formula.api as smf
from io import StringIO
import contextlib
from statsmodels.stats.multitest import multipletests

resultsdir = '/home/cbica/Desktop/LongGPClustering/'

font_size = 20
sns.set_theme(context="paper",style="whitegrid", rc={'axes.grid' : False})
font = {'weight': 'normal', 'size': font_size}
plt.rc('font', **font)

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
plt.rcParams['font.family'] = 'Arial'

# Set the text color to black
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'

parser = argparse.ArgumentParser(description='Plots for Staging Analysis')
## Data Parameters 
parser.add_argument("--datasets", help="GPUs", default='')

args = parser.parse_args()
datasets = args.datasets

# Function to capture model summary output
def capture_model_summary(model_fit):
    """Capture the model summary output as a string"""
    f = StringIO()
    with contextlib.redirect_stdout(f):
        print(model_fit.summary())
    return f.getvalue()

# Initialize results file
results_file = './manuscript1/LMM_results_bonferroni.txt'
with open(results_file, 'w') as f:
    f.write("LMM Model Results for Staging Analysis (with Bonferroni Correction)\n")
    f.write("=" * 70 + "\n\n")

# Initialize lists to store p-values for multiple comparisons correction
all_real_p_values = []
all_pred_p_values = []
roi_names_list = []

roi_names = [ 'Right Hippocampus','Left Hippocampus', 'Right Amygdala', 'Left Amygdala', 'Right OFuG', 'Right PHG','Right Thalamus Proper']
roi_idxs = [13, 14, 4, 5, 99, 109, 23]

for i, r in enumerate(roi_idxs):
    print('ROI', roi_names[i])
    
    with open(results_file, 'a') as f:
        f.write(f"\nROI: {roi_names[i]} (Index: {r})\n")
        f.write("-" * 30 + "\n")

    data_predicted = pd.read_csv('./manuscript1/singletask_MUSE_'+ str(r) + '_dkgp_population_allstudies.csv')

    for c in data_predicted.columns: 
        print(c) 

    data_nonprogressor =  pd.read_csv('/home/cbica/Desktop/LongGPClustering/ADNINonConverters.csv')
    data_progressor = pd.read_csv('/home/cbica/Desktop/LongGPClustering/ADNIConverters.csv')

    predicted_data_ids = list(data_predicted['id'].unique()) 

    longitudinal_covariates = pd.read_csv('longitudinal_covariates_subjectsamples_longclean_hmuse_convs_allstudies.csv')

    print('ADNI Progressors', len(list(data_progressor['PTID'].unique())))
    print('ADNI Non Progressor', len(list(data_nonprogressor['PTID'].unique())))
    print('Keep the real data that I do have the predictions of so as to have 1-1 correspondense')

    predicted_data_ids = list(data_predicted['id'].unique()) 
    data_nonprogressor = data_nonprogressor[data_nonprogressor['PTID'].isin(predicted_data_ids)]
    data_progressor = data_progressor[data_progressor['PTID'].isin(predicted_data_ids)]

    print('Progressors and Non-Progressors for the predictions that we do have')
    print('ADNI Progressors', len(list(data_progressor['PTID'].unique())))
    print('ADNI Non Progressor', len(list(data_nonprogressor['PTID'].unique())))

    progressor_ids = list(data_progressor['PTID'].unique())
    nonprogressor_ids = list(data_nonprogressor['PTID'].unique())

    # Create the Progressor Dataframe.
    subject_data_df = {
        'time': [],
        'ROI': [],
        'subject': [],
        'age_baseline': [], 
        'sex': [],
        'Status': []    }
    for progressor_id in progressor_ids: 

        subject_progressor_data = data_predicted[data_predicted['id']==progressor_id]

        longitudinal_covariates_progressor = longitudinal_covariates[longitudinal_covariates['PTID']==progressor_id]
        print(longitudinal_covariates_progressor.shape[0], subject_progressor_data.shape[0])
        assert subject_progressor_data.shape[0] == longitudinal_covariates_progressor.shape[0]

        time = subject_progressor_data['time'].tolist()
        value = subject_progressor_data['y'].tolist()
        age_progr = longitudinal_covariates_progressor['Age'].tolist()[0]
        sex_progr = longitudinal_covariates_progressor['Sex'].tolist()[0]

        subject_data_df['time'].extend(time)
        subject_data_df['ROI'].extend(value)
        subject_data_df['subject'].extend([progressor_id for l in range(len(time))])
        subject_data_df['Status'].extend(['Progressor' for l in range(len(time))])
        subject_data_df['age_baseline'].extend([age_progr for l in range(len(time))])
        subject_data_df['sex'].extend([sex_progr for l in range(len(time))])

    print('Non Progressor Data')
    for nonprogressor_id in nonprogressor_ids: 
        subject_nonprogressor_data = data_predicted[data_predicted['id']==nonprogressor_id]

        longitudinal_covariates_nonprogressor = longitudinal_covariates[longitudinal_covariates['PTID']==nonprogressor_id]
        print(longitudinal_covariates_nonprogressor.shape[0], subject_nonprogressor_data.shape[0])

        if subject_nonprogressor_data.shape[0] != longitudinal_covariates_nonprogressor.shape[0]: 
            print('Discrepancy in the number of rows between the predicted data and the longitudinal covariates')
            print(subject_nonprogressor_data.shape[0], longitudinal_covariates_nonprogressor.shape[0])
            print(subject_nonprogressor_data.head(10))
            print(longitudinal_covariates_nonprogressor.head(10))


        # assert subject_nonprogressor_data.shape[0] == longitudinal_covariates_nonprogressor.shape[0]

        time = subject_nonprogressor_data['time'].tolist()
        value = subject_nonprogressor_data['y'].tolist()
        age_nonprogr = longitudinal_covariates_nonprogressor['Age'].tolist()[0]
        sex_nonprogr = longitudinal_covariates_nonprogressor['Sex'].tolist()[0]

        subject_data_df['time'].extend(time)
        subject_data_df['ROI'].extend(value)
        subject_data_df['subject'].extend([nonprogressor_id for l in range(len(time))])
        subject_data_df['Status'].extend(['Non-Progressor' for l in range(len(time))])
        subject_data_df['age_baseline'].extend([age_nonprogr for l in range(len(time))])
        subject_data_df['sex'].extend([sex_nonprogr for l in range(len(time))])


    subject_data_df = pd.DataFrame(data=subject_data_df)
    # print(subject_data_df.head(10))

    ### Visualization of Real Data: Progressor vs Non-Progressor
    # Plotting the modified spaghetti plot with LaTeX-like fonts
    plt.figure(figsize=(10,7), dpi=400)
    subject_all = []
    for subject in subject_data_df['subject'].unique():
        subject_data = subject_data_df[subject_data_df['subject'] == subject]
        longitudinal_age = longitudinal_covariates[longitudinal_covariates['PTID']==subject]['Age']

        if len(longitudinal_age) != subject_data.shape[0]: 
            continue

        subject_data['Age'] = longitudinal_age.tolist()
        group = subject_data['Status'].iloc[0]
        color = '#8B0000B2' if group == 'Progressor' else '#4299E1B2'  # Wine red and very light blue with transparency

        plt.plot(subject_data['time'], subject_data['ROI'], color=color, alpha=0.3, marker='o', label=subject_data.iloc[0]['Status'])
        plt.scatter(subject_data['time'], subject_data['ROI'], color=color, alpha=0.3, marker='o')
        subject_all.append(subject_data)
        
    subject_data_df = pd.concat(subject_all, axis=0)

    # Avoiding duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    print(subject_data_df.head(10))

    # Fit a linear model for each group and plot the lm trajectory
    for group, vivid_color, light_color in [('Progressor', '#8B0000', '#8B0000B2'), ('Non-Progressor', '#4299E1', '#4299E1B2')]:
        group_data = subject_data_df[subject_data_df['Status'] == group]

        # fit a lmm for group data and calculate the intervals
        model = smf.mixedlm("ROI ~ time + Status * time", groups=group_data['subject'], data=group_data, re_formula="~time")
        result =  model.fit()
        time_values = group_data['time'].unique().tolist()

        time_values_sorted = sorted(time_values)
        time_values = np.array(time_values_sorted)
        time_values = time_values.tolist()
        status_values = [group for l in range(len(time_values))]

        test_data = pd.DataFrame({'time': time_values, 'Status': status_values})
        mean_predicted = result.predict(exog=test_data)
        pred_se = np.sqrt(result.scale)
        ci_multiplier = 1.96  # Z-score multiplier for 95% CI
        lower_ci = mean_predicted - ci_multiplier * pred_se
        upper_ci = mean_predicted + ci_multiplier * pred_se


        gfg = sns.lineplot(x='time', y='ROI', data=group_data, ci=None, estimator=None, markers='o', units="subject", color=light_color, alpha=0.6, label=group)
        plt.plot(time_values, mean_predicted, 'b-', label='Fitted Line', color=vivid_color, alpha=0.9, lw=2)
        plt.fill_between(time_values, lower_ci, upper_ci, color=vivid_color, alpha=0.3, label='95% CI')
    
    sns.despine(bottom=True, left=True)
    plt.xlabel('Time (in months) from baseline', fontdict=font)
    plt.ylabel(roi_names[i], fontdict=font)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.title('Noisy Measurements', fontdict=font)
    plt.legend(by_label.values(), by_label.keys(), loc='upper left',fontsize=19)
    plt.savefig('./manuscript1/real_staging_plot_' + roi_names[i]  + '.png')
    plt.savefig('./manuscript1/real_staging_plot_' + roi_names[i]  + '.svg', format='svg')


    plt.figure(figsize=(10,7), dpi=400)
    for group, vivid_color, light_color in [('Progressor', '#8B0000', '#8B0000B2'), ('Non-Progressor', '#4299E1', '#4299E1B2')]:
        group_data = subject_data_df[subject_data_df['Status'] == group]

        model = smf.mixedlm("ROI ~ Age + Status * Age", groups=group_data['subject'], data=group_data, re_formula="~Age")
        result =  model.fit()
        time_values = group_data['Age'].unique().tolist()

        time_values_sorted = sorted(time_values)
        time_values = np.array(time_values_sorted)
        time_values = time_values.tolist()
        status_values = [group for l in range(len(time_values))]

        test_data = pd.DataFrame({'Age': time_values, 'Status': status_values})
        mean_predicted = result.predict(exog=test_data)
        pred_se = np.sqrt(result.scale)
        ci_multiplier = 1.96  # Z-score multiplier for 95% CI
        lower_ci = mean_predicted - ci_multiplier * pred_se
        upper_ci = mean_predicted + ci_multiplier * pred_se

        gfg = sns.lineplot(x='Age', y='ROI', data=group_data, ci=None, estimator=None, markers='o', units="subject", color=light_color, alpha=0.5, label=group)
        plt.scatter(group_data['Age'], group_data['ROI'], color=light_color, alpha=0.5, marker='o')

        plt.plot(time_values, mean_predicted, 'b-', label='Fitted Line', color=vivid_color, alpha=0.9, lw=2)
        plt.fill_between(time_values, lower_ci, upper_ci, color=vivid_color, alpha=0.3, label='95% CI')
    
    sns.despine(bottom=True, left=True)
    plt.xlabel('Age', fontdict=font)
    plt.ylabel(roi_names[i], fontdict=font)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.title('Noisy Measurements', fontdict=font)
    plt.legend().remove()
    plt.savefig('./nataging/real_age_staging_plot_' + roi_names[i]  + '.png')
    plt.savefig('./nataging/real_age_staging_plot_' + roi_names[i]  + '.svg', format='svg')

    fixed_effects_formula = 'ROI ~ age_baseline + sex + Status + time + Status * time' 
    md = smf.mixedlm(fixed_effects_formula, groups=subject_data_df['subject'], data=subject_data_df)
    mdf = md.fit()
    print('Real')
    print(mdf.summary()) 
    
    real_p_value = mdf.pvalues['Status[T.Progressor]:time']
    all_real_p_values.append(real_p_value)
    roi_names_list.append(roi_names[i])
    
    with open(results_file, 'a') as f:
        f.write("\nREAL DATA LMM RESULTS:\n")
        f.write("Model Formula: " + fixed_effects_formula + "\n")
        f.write(capture_model_summary(mdf))
        f.write("\n" + "="*50 + "\n")


    ### Visualization of Predicted Data: Progressor vs Non-Progressor 
    subject_data = {'time': [], 'ROI': [], 'subject': [], 'Status': [], 'age_baseline': [], 'sex': []}

    for progressor_id in progressor_ids: 
        subject_progressor_data = data_predicted[data_predicted['id']==progressor_id]
        longitudinal_covariates_progressor = longitudinal_covariates[longitudinal_covariates['PTID']==progressor_id]
        print(longitudinal_covariates_progressor.shape[0], subject_progressor_data.shape[0])
        assert subject_progressor_data.shape[0] == longitudinal_covariates_progressor.shape[0]

        time = subject_progressor_data['time'].tolist()
        value = subject_progressor_data['score'].tolist()
        age_progr = longitudinal_covariates_progressor['Age'].tolist()[0]
        sex_progr = longitudinal_covariates_progressor['Sex'].tolist()[0]

        subject_data['time'].extend(time)
        subject_data['ROI'].extend(value)
        subject_data['subject'].extend([progressor_id for l in range(len(time))])
        subject_data['Status'].extend(['Progressor' for l in range(len(time))])
        subject_data['age_baseline'].extend([age_progr for l in range(len(time))])
        subject_data['sex'].extend([sex_progr for l in range(len(time))])

    print('Non Progressor Data')
    for nonprogressor_id in nonprogressor_ids: 
        subject_nonprogressor_data = data_predicted[data_predicted['id']==nonprogressor_id]
        longitudinal_covariates_nonprogressor = longitudinal_covariates[longitudinal_covariates['PTID']==nonprogressor_id]
        print(longitudinal_covariates_nonprogressor.shape[0], subject_nonprogressor_data.shape[0])

        if subject_nonprogressor_data.shape[0] != longitudinal_covariates_nonprogressor.shape[0]: 
            print('Discrepancy in the number of rows between the predicted data and the longitudinal covariates')
            print(subject_nonprogressor_data.shape[0], longitudinal_covariates_nonprogressor.shape[0])
            print(subject_nonprogressor_data.head(10))
            print(longitudinal_covariates_nonprogressor.head(10))

        time = subject_nonprogressor_data['time'].tolist()
        value = subject_nonprogressor_data['score'].tolist()
        age_nonprogr = longitudinal_covariates_nonprogressor['Age'].tolist()[0]
        sex_nonprogr = longitudinal_covariates_nonprogressor['Sex'].tolist()[0]

        subject_data['time'].extend(time)
        subject_data['ROI'].extend(value)
        subject_data['subject'].extend([nonprogressor_id for l in range(len(time))])
        subject_data['Status'].extend(['Non-Progressor' for l in range(len(time))])
        subject_data['age_baseline'].extend([age_nonprogr for l in range(len(time))])
        subject_data['sex'].extend([sex_nonprogr for l in range(len(time))])

    subject_data_df = pd.DataFrame(data=subject_data)

    ### Visualization of Predicted Data: Progressor vs Non-Progressor
    subject_all = [] 
    plt.figure(figsize=(10,7), dpi=400)
    for subject in subject_data_df['subject'].unique():
        subject_data = subject_data_df[subject_data_df['subject'] == subject]
        longitudinal_age = longitudinal_covariates[longitudinal_covariates['PTID']==subject]['Age']
    
        if len(longitudinal_age)!= subject_data.shape[0]: 
            continue 

        subject_data['Age'] = longitudinal_age.tolist()
        group = subject_data['Status'].iloc[0]
        color = '#8B0000B2' if group == 'Progressor' else '#4299E1B2'  # Wine red and very light blue with transparency
        plt.plot(subject_data['time'], subject_data['ROI'], color=color, alpha=0.3, marker='o', label=subject_data.iloc[0]['Status'])
        plt.scatter(subject_data['time'], subject_data['ROI'], color=color, alpha=0.3, marker='o')
        subject_all.append(subject_data)

    subject_data_df = pd.concat(subject_all, axis=0)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    for group, vivid_color, light_color in [('Progressor', '#8B0000', '#8B0000B2'), ('Non-Progressor', '#4299E1', '#4299E1B2')]:
        group_data = subject_data_df[subject_data_df['Status'] == group]

        # fit a lmm for group data and calculate the intervals
        model = smf.mixedlm("ROI ~ time + Status * time", groups=group_data['subject'], data=group_data)
        result =  model.fit()
        time_values = group_data['time'].unique().tolist()

        time_values_sorted = sorted(time_values)
        time_values = np.array(time_values_sorted)
        time_values = time_values.tolist()
        status_values = [group for l in range(len(time_values))]

        test_data = pd.DataFrame({'time': time_values, 'Status': status_values})
        mean_predicted = result.predict(exog=test_data)
        pred_se = np.sqrt(result.scale)
        ci_multiplier = 1.96  # Z-score multiplier for 95% CI
        lower_ci = mean_predicted - ci_multiplier * pred_se
        upper_ci = mean_predicted + ci_multiplier * pred_se


        gfg = sns.lineplot(x='time', y='ROI', data=group_data, ci=None, estimator=None, markers='o', units="subject", color=light_color, alpha=0.6, label=group)
        plt.plot(time_values, mean_predicted, label='Fitted Line', color=vivid_color, alpha=0.9, lw=2)
        plt.fill_between(time_values, lower_ci, upper_ci, color=vivid_color, alpha=0.3, label='95% CI')
    
    sns.despine(bottom=True, left=True)
    plt.xlabel('Time (in months) from baseline', fontdict=font)
    plt.ylabel(roi_names[i], fontdict=font)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.title('Predicted Trajectory', fontdict=font)
    plt.legend().remove()
    plt.savefig('./manuscript1/predicted_staging_plot_' + roi_names[i]  + '.png')
    plt.savefig('./manuscript1/predicted_staging_plot_' + roi_names[i]  + '.svg', format='svg')



    ## Predicted Age Curves ###
    plt.figure(figsize=(10,7), dpi=400)
    ### Plot the Age-Status Trend ###
    # Fit a linear model for each group and plot the lm trajectory
    for group, vivid_color, light_color in [('Progressor', '#8B0000', '#8B0000B2'), ('Non-Progressor', '#4299E1', '#4299E1B2')]:
        group_data = subject_data_df[subject_data_df['Status'] == group]

        # fit a lmm for group data and calculate the intervals
        model = smf.mixedlm("ROI ~ Age + Status * Age", groups=group_data['subject'], data=group_data)
        result =  model.fit()
        time_values = group_data['Age'].unique().tolist()

        time_values_sorted = sorted(time_values)
        time_values = np.array(time_values_sorted)
        time_values = time_values.tolist()
        status_values = [group for l in range(len(time_values))]

        test_data = pd.DataFrame({'Age': time_values, 'Status': status_values})
        mean_predicted = result.predict(exog=test_data)
        pred_se = np.sqrt(result.scale)
        ci_multiplier = 1.96  # Z-score multiplier for 95% CI
        lower_ci = mean_predicted - ci_multiplier * pred_se
        upper_ci = mean_predicted + ci_multiplier * pred_se


        gfg = sns.lineplot(x='Age', y='ROI', data=group_data, ci=None, estimator=None, units="subject", color=light_color, alpha=0.5, label=group)
        plt.scatter(group_data['Age'], group_data['ROI'], color=light_color, alpha=0.5, marker='o')
        plt.plot(time_values, mean_predicted, label='Fitted Line', color=vivid_color, alpha=0.9, lw=2)
        plt.fill_between(time_values, lower_ci, upper_ci, color=vivid_color, alpha=0.3, label='95% CI')
    
    sns.despine(bottom=True, left=True)
    plt.xlabel('Age', fontdict=font)
    plt.ylabel(roi_names[i], fontdict=font)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.title('Predicted Trejectory', fontdict=font)
    plt.legend().remove()
    plt.savefig('./manuscript1/predicted_age_staging_plot_' + roi_names[i]  + '.png')
    plt.savefig('./manuscript1/predicted_age_staging_plot_' + roi_names[i]  + '.svg', format='svg')


    fixed_effects_formula = 'ROI ~ age_baseline + sex + Status + time + Status * time' 
    md = smf.mixedlm(fixed_effects_formula, groups=subject_data_df['subject'], data=subject_data_df)
    mdf = md.fit()
    print('Predicted')
    print(mdf.summary()) 
    
    pred_p_value = mdf.pvalues['Status[T.Progressor]:time']
    all_pred_p_values.append(pred_p_value)
    
    with open(results_file, 'a') as f:
        f.write("\nPREDICTED DATA LMM RESULTS:\n")
        f.write("Model Formula: " + fixed_effects_formula + "\n")
        f.write(capture_model_summary(mdf))
        f.write("\n" + "="*50 + "\n")


# Apply Bonferroni correction for multiple comparisons
print("\n" + "="*80)
print("MULTIPLE COMPARISONS CORRECTION (BONFERRONI)")
print("="*80)

n_tests = len(roi_names)
alpha = 0.05
bonferroni_threshold = alpha / n_tests

print(f"Number of ROIs tested: {n_tests}")
print(f"Original alpha level: {alpha}")
print(f"Bonferroni corrected alpha: {bonferroni_threshold:.6f}")
print(f"Bonferroni threshold (α/{n_tests}): {bonferroni_threshold:.6f}")
print()

real_corrected = multipletests(all_real_p_values, alpha=alpha, method='bonferroni')
pred_corrected = multipletests(all_pred_p_values, alpha=alpha, method='bonferroni')

print("REAL DATA RESULTS:")
print("-" * 40)
real_sig_original = sum(1 for p in all_real_p_values if p < alpha)
real_sig_corrected = sum(real_corrected[0])
print(f"Significant at α = {alpha}: {real_sig_original}/{n_tests} ROIs")
print(f"Significant after Bonferroni correction: {real_sig_corrected}/{n_tests} ROIs")
print()

print("PREDICTED DATA RESULTS:")
print("-" * 40)
pred_sig_original = sum(1 for p in all_pred_p_values if p < alpha)
pred_sig_corrected = sum(pred_corrected[0])
print(f"Significant at α = {alpha}: {pred_sig_original}/{n_tests} ROIs")
print(f"Significant after Bonferroni correction: {pred_sig_corrected}/{n_tests} ROIs")
print()

print("DETAILED RESULTS BY ROI:")
print("-" * 80)
print(f"{'ROI':<25} {'Real P':<10} {'Real P*':<10} {'Real Sig*':<10} {'Pred P':<10} {'Pred P*':<10} {'Pred Sig*':<10}")
print("-" * 80)

for i, roi in enumerate(roi_names_list):
    real_sig = "Yes" if real_corrected[0][i] else "No"
    pred_sig = "Yes" if pred_corrected[0][i] else "No"
    
    print(f"{roi:<25} {all_real_p_values[i]:<10.6f} {real_corrected[1][i]:<10.6f} {real_sig:<10} "
          f"{all_pred_p_values[i]:<10.6f} {pred_corrected[1][i]:<10.6f} {pred_sig:<10}")

print("-" * 80)
print("P* = Bonferroni corrected p-value")
print("Sig* = Significant after Bonferroni correction")
print()

# Summary for manuscript
print("MANUSCRIPT SUMMARY:")
print("-" * 40)
print(f"After applying Bonferroni correction for {n_tests} ROIs (α = {alpha/n_tests:.6f}):")
print(f"• Real data: {real_sig_corrected}/{n_tests} ROIs show significant Status×Time interaction")
print(f"• Predicted data: {pred_sig_corrected}/{n_tests} ROIs show significant Status×Time interaction")

if real_sig_corrected > 0:
    sig_real_rois = [roi_names_list[i] for i, sig in enumerate(real_corrected[0]) if sig]
    print(f"• Significant ROIs in real data: {', '.join(sig_real_rois)}")

if pred_sig_corrected > 0:
    sig_pred_rois = [roi_names_list[i] for i, sig in enumerate(pred_corrected[0]) if sig]
    print(f"• Significant ROIs in predicted data: {', '.join(sig_pred_rois)}")

# Save corrected results to file
with open(results_file, 'a') as f:
    f.write("\n\n" + "="*70 + "\n")
    f.write("MULTIPLE COMPARISONS CORRECTION (BONFERRONI)\n")
    f.write("="*70 + "\n")
    f.write(f"Number of ROIs tested: {n_tests}\n")
    f.write(f"Original alpha level: {alpha}\n")
    f.write(f"Bonferroni corrected alpha: {bonferroni_threshold:.6f}\n")
    f.write(f"Bonferroni threshold (α/{n_tests}): {bonferroni_threshold:.6f}\n\n")
    
    f.write("REAL DATA RESULTS:\n")
    f.write("-" * 40 + "\n")
    f.write(f"Significant at α = {alpha}: {real_sig_original}/{n_tests} ROIs\n")
    f.write(f"Significant after Bonferroni correction: {real_sig_corrected}/{n_tests} ROIs\n\n")
    
    f.write("PREDICTED DATA RESULTS:\n")
    f.write("-" * 40 + "\n")
    f.write(f"Significant at α = {alpha}: {pred_sig_original}/{n_tests} ROIs\n")
    f.write(f"Significant after Bonferroni correction: {pred_sig_corrected}/{n_tests} ROIs\n\n")
    
    f.write("DETAILED RESULTS BY ROI:\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'ROI':<25} {'Real P':<10} {'Real P*':<10} {'Real Sig*':<10} {'Pred P':<10} {'Pred P*':<10} {'Pred Sig*':<10}\n")
    f.write("-" * 80 + "\n")
    
    for i, roi in enumerate(roi_names_list):
        real_sig = "Yes" if real_corrected[0][i] else "No"
        pred_sig = "Yes" if pred_corrected[0][i] else "No"
        
        f.write(f"{roi:<25} {all_real_p_values[i]:<10.6f} {real_corrected[1][i]:<10.6f} {real_sig:<10} "
                f"{all_pred_p_values[i]:<10.6f} {pred_corrected[1][i]:<10.6f} {pred_sig:<10}\n")
    
    f.write("-" * 80 + "\n")
    f.write("P* = Bonferroni corrected p-value\n")
    f.write("Sig* = Significant after Bonferroni correction\n\n")
    
    f.write("MANUSCRIPT SUMMARY:\n")
    f.write("-" * 40 + "\n")
    f.write(f"After applying Bonferroni correction for {n_tests} ROIs (α = {alpha/n_tests:.6f}):\n")
    f.write(f"• Real data: {real_sig_corrected}/{n_tests} ROIs show significant Status×Time interaction\n")
    f.write(f"• Predicted data: {pred_sig_corrected}/{n_tests} ROIs show significant Status×Time interaction\n")
    
    if real_sig_corrected > 0:
        sig_real_rois = [roi_names_list[i] for i, sig in enumerate(real_corrected[0]) if sig]
        f.write(f"• Significant ROIs in real data: {', '.join(sig_real_rois)}\n")
    
    if pred_sig_corrected > 0:
        sig_pred_rois = [roi_names_list[i] for i, sig in enumerate(pred_corrected[0]) if sig]
        f.write(f"• Significant ROIs in predicted data: {', '.join(sig_pred_rois)}\n")

print(f"\nLMM results with Bonferroni correction have been saved to: {results_file}")
