'''
Manuscript1: SPARE-AD Longitudinal Analysis for Nature Aging

This script is used to analyze the SPARE-AD longitudinal data and generate the plots for the manuscript.

- Error with history 
- Discriminative Analysis (ROC Curves + PR Curves)

Updated: Added PR-AUC, F1, Precision, Recall metrics for consistency with other results sections.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    f1_score
)
import sys
import os
from datetime import datetime

# Set up logging to capture all output
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Create log file with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f'./miccai26/nature_aging_spare_ad_results_{timestamp}.txt'
os.makedirs('./miccai26', exist_ok=True)
sys.stdout = Logger(log_filename)

import seaborn as sns
from scipy import stats
import pickle
from os.path import exists
from matplotlib.pyplot import figure
from operator import add
import argparse
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

##### Functions ########

def calculate_rate_of_change(group, column='y'):
    slope, intercept = np.polyfit(group['time'], group[column], 1)
    return slope

def calculate_rate_of_change_prediction(group, column='prediction'):
    slope, intercept = np.polyfit(group['time'], group[column], 1)
    return slope

def youden_operating_point(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = tpr - fpr
    i = np.argmax(j)
    return {
        "threshold": thr[i],
        "TPR": float(tpr[i]),
        "FPR": float(fpr[i]),
        "specificity": float(1 - fpr[i]),
        "youden_J": float(j[i]),
    }

def ppv_fdr_at_threshold(y_true, y_score, thr):
    y_pred = (y_score >= thr).astype(int)
    TP = int(((y_pred==1) & (y_true==1)).sum())
    FP = int(((y_pred==1) & (y_true==0)).sum())
    PPV = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    FDR = 1 - PPV if not np.isnan(PPV) else np.nan
    return float(PPV), float(FDR)

# -------------------------------
# NEW: PR-AUC and F1 utilities
# -------------------------------

def calculate_prauc(y_true, y_score):
    """
    Compute PR-AUC (average precision) and its bootstrap SE.
    Uses sklearn's average_precision_score which computes the
    area under the precision-recall curve using the trapezoidal rule.
    Bootstrap SE is estimated over 1000 resamples.
    """
    ap = average_precision_score(y_true, y_score)
    
    # Bootstrap SE
    n_bootstrap = 1000
    ap_boot = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        ap_boot.append(average_precision_score(y_true[idx], y_score[idx]))
    se = float(np.std(ap_boot)) if ap_boot else np.nan
    return float(ap), se

def calculate_f1_at_threshold(y_true, y_score, threshold=None):
    """
    Compute F1 score. If threshold is None, finds the threshold
    that maximises F1 on the precision-recall curve (optimal F1).
    Returns F1, precision, recall, and the threshold used.
    """
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_score)
    
    # precision_recall_curve appends a final point (precision=1, recall=0)
    # with no corresponding threshold; align arrays
    f1_vals = np.where(
        (precision_vals[:-1] + recall_vals[:-1]) > 0,
        2 * precision_vals[:-1] * recall_vals[:-1] / (precision_vals[:-1] + recall_vals[:-1]),
        0.0
    )
    
    if threshold is None:
        best_idx = np.argmax(f1_vals)
        best_thr = thresholds[best_idx]
    else:
        best_thr = threshold
        diffs = np.abs(thresholds - best_thr)
        best_idx = np.argmin(diffs)
    
    return {
        "F1":        float(f1_vals[best_idx]),
        "Precision": float(precision_vals[best_idx]),
        "Recall":    float(recall_vals[best_idx]),
        "threshold": float(best_thr),
    }

def calculate_auc_se(y_true, y_scores):
    """Hanley-McNeil SE for ROC-AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_val = auc(fpr, tpr)
    n1 = np.sum(y_true)
    n2 = len(y_true) - n1
    Q1 = auc_val / (2 - auc_val)
    Q2 = 2 * auc_val**2 / (1 + auc_val)
    se = np.sqrt((auc_val * (1 - auc_val) +
                  (n1 - 1) * (Q1 - auc_val**2) +
                  (n2 - 1) * (Q2 - auc_val**2)) / (n1 * n2))
    return auc_val, se

# -------------------------------
font_size = 19
resultdir = '/home/cbica/Desktop/LongGPClustering'

parser = argparse.ArgumentParser(description='Updated Plots')
parser.add_argument("--datasets", help="GPUs", default='allstudies')
args = parser.parse_args()
datasets = args.datasets

print("="*80)
print("SPARE-AD LONGITUDINAL ANALYSIS FOR NATURE AGING MANUSCRIPT")
print("="*80)
print(f"Analysis datasets: {datasets}")
print(f"Result directory: {resultdir}")
print()

#### STRATIFY METRICS COVARIATES #####
print("Loading longitudinal covariates...")
longitudinal_covariates = pd.read_csv(
    '/home/cbica/Desktop/LongGPRegressionBaseline/'
    'longitudinal_covariates_subjectsamples_longclean_hmuse_convs_allstudies.csv'
)
longitudinal_covariates['Diagnosis'].replace(
    [-1.0, 0.0, 1.0, 2.0], ['UKN', 'CN', 'MCI', 'AD'], inplace=True
)

print(f"Total subjects in longitudinal covariates: {len(longitudinal_covariates['PTID'].unique())}")
print(f"Total observations: {len(longitudinal_covariates)}")
print(f"Diagnosis distribution:")
print(longitudinal_covariates['Diagnosis'].value_counts())
print()

race_mapping = {
    'White': 'White', 'Black': 'Black', 'Asian': 'Asian',
    'Chinese': 'Asian', 'Japanese': 'Asian', 'Filipino': 'Asian',
    'Other Asian or Other Pacific Islander': 'Asian',
    'Hawaiian/Other PI': 'Pacific Islander',
    'Am Indian/Alaskan': 'American Indian',
    'American Indian or Alaska Native': 'American Indian',
    'More than one': 'More than one',
    'Unknown': 'Unknown', 'Not Classifiable': 'Unknown',
    'Other NonWhite': 'Other NonWhite'
}
longitudinal_covariates = longitudinal_covariates.replace(race_mapping)

##### PLOT METRICS ####
roi_names = ['SPARE_AD']
roi_idxs  = [0]

df_metrics_over_time = {
    'biomarker':[], 'id': [], 'time': [], 'Tobs': [],
    'y': [], 'model': [], 'history': [], 'prediction': [],
    'status': [], 'sex': [], 'race': [], 'education': [], 'apoe4': []
}
df_rate_of_change_over_time = {
    'biomarker':[], 'id': [], 'Tobs': [], 'model': [],
    'history': [], 'prediction': [], 'real': [], 'error': [], 'status':[]
}

for roi_idx, task in enumerate(roi_names):
    print(f"Processing {task} biomarker...")

    tempdkgp = pd.read_csv(
        './manuscript1/adaptive_shrinkage_predictions_alpha_simple_dkgp_SPARE_AD_allstudies.csv'
    )
    print(f"Loaded predictions file with {len(tempdkgp)} total observations")
    print(f"Unique subjects in predictions: {len(tempdkgp['id'].unique())}")

    total_ids = tempdkgp['id'].unique().tolist()

    subject_counts = {
        'Healthy Control': 0, 'CN Progressor to MCI': 0, 'AD Progressor': 0,
        'MCI Stable': 0, 'AD': 0, 'MCI Progressor': 0, 'UKN': 0
    }
    total_observations      = 0
    subjects_with_covariates    = 0
    subjects_without_covariates = 0

    for s in total_ids:
        tempdkgp_s = tempdkgp[tempdkgp['id'] == s]
        longitudinal_covariates_u = longitudinal_covariates[longitudinal_covariates['PTID'] == s]

        if longitudinal_covariates_u.shape[0] == 0:
            subjects_without_covariates += 1
            continue
        subjects_with_covariates += 1

        race      = longitudinal_covariates_u['Race'].to_list()[0]
        sex       = longitudinal_covariates_u['Sex'].to_list()[0]
        sex       = 'M' if sex == 0 else 'F'
        apoe4     = longitudinal_covariates_u['APOE4_Alleles'].to_list()[0]
        education = longitudinal_covariates_u['Education_Years'].to_list()[0]
        education = '0-16 years' if education == 0 else '16 or more years'
        if apoe4 == -1:
            apoe4 = 'UNK'

        diagnosis = longitudinal_covariates_u['Diagnosis'].tolist()
        if   diagnosis[0]=='CN'  and diagnosis[-1]=='CN':  status = 'Healthy Control'
        elif diagnosis[0]=='CN'  and diagnosis[-1]=='MCI': status = 'CN Progressor to MCI'
        elif diagnosis[0]=='CN'  and diagnosis[-1]=='AD':  status = 'AD Progressor'
        elif diagnosis[0]=='MCI' and diagnosis[-1]=='MCI': status = 'MCI Stable'
        elif diagnosis[0]=='AD'  and diagnosis[-1]=='AD':  status = 'AD'
        elif diagnosis[0]=='MCI' and diagnosis[-1]=='AD':  status = 'MCI Progressor'
        else:                                               status = 'UKN'

        subject_counts[status] += 1
        total_observations += len(tempdkgp_s['y'].to_list())

        n = len(tempdkgp_s['y'].to_list())
        df_metrics_over_time['time'].extend(tempdkgp_s['time'].to_list())
        df_metrics_over_time['id'].extend(tempdkgp_s['id'].to_list())
        df_metrics_over_time['y'].extend(tempdkgp_s['y'].to_list())
        df_metrics_over_time['history'].extend(tempdkgp_s['History'].to_list())
        df_metrics_over_time['Tobs'].extend(tempdkgp_s['time'].to_list())
        df_metrics_over_time['prediction'].extend(tempdkgp_s['score'].to_list())
        df_metrics_over_time['status'].extend([status]*n)
        df_metrics_over_time['sex'].extend([sex]*n)
        df_metrics_over_time['race'].extend([race]*n)
        df_metrics_over_time['education'].extend([education]*n)
        df_metrics_over_time['apoe4'].extend([apoe4]*n)
        df_metrics_over_time['biomarker'].extend([roi_names[roi_idx]]*n)
        df_metrics_over_time['model'].extend(['TempDKGP']*n)

        history_points = tempdkgp_s['History'].unique()
        for h in history_points:
            tempdkgp_s_h = tempdkgp_s[tempdkgp_s['History'] == h]
            slope,      _ = np.polyfit(tempdkgp_s_h['time'], tempdkgp_s_h['score'], 1)
            real_slope, _ = np.polyfit(tempdkgp_s_h['time'], tempdkgp_s_h['y'], 1)
            error = abs(slope - real_slope)

            df_rate_of_change_over_time['id'].append(s)
            df_rate_of_change_over_time['Tobs'].append(tempdkgp_s_h['time'].to_list()[h-1])
            df_rate_of_change_over_time['biomarker'].append(roi_names[roi_idx])
            df_rate_of_change_over_time['model'].append('TempDKGP')
            df_rate_of_change_over_time['history'].append(h)
            df_rate_of_change_over_time['prediction'].append(slope)
            df_rate_of_change_over_time['real'].append(real_slope)
            df_rate_of_change_over_time['error'].append(error)
            df_rate_of_change_over_time['status'].append(status)

    df = pd.DataFrame(data=df_metrics_over_time)
    df_roc = pd.DataFrame(data=df_rate_of_change_over_time)
    df_roc.to_csv('./miccai26/SPAREAD_Experiment_Section2_RateOfChange_'+datasets+'_'+roi_names[roi_idx]+'.csv')
    df.to_csv('./miccai26/SPAREAD_Experiment_Section2_AdaptedPredictions_'+datasets+'_'+roi_names[roi_idx]+'.csv')


# ================================================================
# DISCRIMINATION ANALYSIS — ROC + PR metrics
# ================================================================
print("\n" + "="*80)
print("DISCRIMINATION ANALYSIS (ROC-AUC + PR-AUC + F1)")
print("="*80)

df = pd.read_csv("./manuscript1/SPAREAD_Experiment_Section2_RateOfChange_allstudies_SPARE_AD.csv")

# Convert Tobs to years and bin
df['Tobs_years'] = df['Tobs'] / 12.0
df['year_bin'] = pd.cut(
    df['Tobs_years'],
    bins=[0, 1, 2, 3, 4, 5, float('inf')],
    labels=['0-1', '1-2', '2-3', '3-4', '4-5', '5+'],
    include_lowest=True
)

comparison_pairs = [
    ("Healthy Control",  "CN Progressor to MCI"),
    ("MCI Stable",       "MCI Progressor"),
]

year_bins = ['0-1', '1-2', '2-3', '3-4', '4-5', '5+']
results_tobs = []

for year_bin in year_bins:
    df_bin = df[df["year_bin"] == year_bin]
    print(f"\nYear bin {year_bin}: {len(df_bin)} obs, {len(df_bin['id'].unique())} subjects")

    for g1, g2 in comparison_pairs:
        sub_df = df_bin[df_bin["status"].isin([g1, g2])].copy()
        sub_df["label"] = (sub_df["status"] == g2).astype(int)

        grouped_real  = sub_df.groupby("id")["real"].mean()
        grouped_pred  = sub_df.groupby("id")["prediction"].mean()
        grouped_label = sub_df.groupby("id")["label"].first()

        if len(grouped_label.unique()) < 2 or len(grouped_label) < 10:
            print(f"  {g1} vs {g2}: Insufficient data (n={len(grouped_label)})")
            continue

        y_true  = grouped_label.values.astype(int)
        y_real  = grouped_real.values
        y_pred  = grouped_pred.values

        # ---- ROC-AUC ----
        auc_real, se_real = calculate_auc_se(y_true, y_real)
        auc_pred, se_pred = calculate_auc_se(y_true, y_pred)

        # ---- Youden operating point ----
        op_real = youden_operating_point(y_true, y_real)
        op_pred = youden_operating_point(y_true, y_pred)
        ppv_real, fdr_real = ppv_fdr_at_threshold(y_true, y_real, op_real["threshold"])
        ppv_pred, fdr_pred = ppv_fdr_at_threshold(y_true, y_pred, op_pred["threshold"])

        # ---- PR-AUC (average precision) ----
        prauc_real, prase_real = calculate_prauc(y_true, y_real)
        prauc_pred, prase_pred = calculate_prauc(y_true, y_pred)

        # ---- Optimal F1 (maximised over PR threshold) ----
        f1_real = calculate_f1_at_threshold(y_true, y_real)
        f1_pred = calculate_f1_at_threshold(y_true, y_pred)

        # ---- Baseline PR-AUC (prevalence = random classifier) ----
        prevalence = y_true.mean()

        print(f"  {g1} vs {g2}:")
        print(f"    n={len(y_true)}, prevalence={prevalence:.3f}")
        print(f"    ROC-AUC  — Real: {auc_real:.3f}±{se_real:.3f} | Pred: {auc_pred:.3f}±{se_pred:.3f}")
        print(f"    PR-AUC   — Real: {prauc_real:.3f}±{prase_real:.3f} | Pred: {prauc_pred:.3f}±{prase_pred:.3f} (baseline={prevalence:.3f})")
        print(f"    Best F1  — Real: {f1_real['F1']:.3f} (P={f1_real['Precision']:.3f}, R={f1_real['Recall']:.3f}) | "
              f"Pred: {f1_pred['F1']:.3f} (P={f1_pred['Precision']:.3f}, R={f1_pred['Recall']:.3f})")
        print(f"    Youden   — Real: TPR={op_real['TPR']:.3f}, PPV={ppv_real:.3f} | "
              f"Pred: TPR={op_pred['TPR']:.3f}, PPV={ppv_pred:.3f}")

        results_tobs.append({
            "year_bin": year_bin,
            "comparison": f"{g1} vs {g2}",
            "N": int(len(y_true)),
            "Prevalence": float(prevalence),

            # ROC-AUC
            "ROCAUC_Real":    float(auc_real),
            "ROCAUC_SE_Real": float(se_real),
            "ROCAUC_Pred":    float(auc_pred),
            "ROCAUC_SE_Pred": float(se_pred),

            # PR-AUC
            "PRAUC_Real":    float(prauc_real),
            "PRAUC_SE_Real": float(prase_real),
            "PRAUC_Pred":    float(prauc_pred),
            "PRAUC_SE_Pred": float(prase_pred),
            "PRAUC_Baseline": float(prevalence),

            # Optimal F1 — Real
            "F1_Real":        f1_real["F1"],
            "Precision_Real": f1_real["Precision"],
            "Recall_Real":    f1_real["Recall"],

            # Optimal F1 — Predicted
            "F1_Pred":        f1_pred["F1"],
            "Precision_Pred": f1_pred["Precision"],
            "Recall_Pred":    f1_pred["Recall"],

            # Youden — Real
            "TPR_Real_Youden": op_real["TPR"],
            "FPR_Real_Youden": op_real["FPR"],
            "PPV_Real_Youden": ppv_real,

            # Youden — Predicted
            "TPR_Pred_Youden": op_pred["TPR"],
            "FPR_Pred_Youden": op_pred["FPR"],
            "PPV_Pred_Youden": ppv_pred,
        })

results_tobs_df = pd.DataFrame(results_tobs)
results_tobs_df.to_csv("./miccai26/SPAREAD_Experiment_Discrimination_AllMetrics_by_Time.csv", index=False)

# ================================================================
# FIGURES — ROC + PR curves side by side, per year bin
# ================================================================
print("\nGenerating publication-quality ROC + PR curve figures...")

plt.rcParams.update({
    'font.size': 12,
    'axes.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'legend.frameon': False,
})

for year_bin in ['0-1', '2-3']:          # manuscript focal bins
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f'SPARE-AD Rate of Change Discrimination — {year_bin} Years Since Baseline',
        fontsize=16, y=0.98
    )

    for comp_idx, (g1, g2) in enumerate(comparison_pairs):
        ax_roc = axes[comp_idx, 0]
        ax_pr  = axes[comp_idx, 1]

        df_bin = df[df["year_bin"] == year_bin]
        sub_df = df_bin[df_bin["status"].isin([g1, g2])].copy()
        sub_df["label"] = (sub_df["status"] == g2).astype(int)

        grouped_real  = sub_df.groupby("id")["real"].mean()
        grouped_pred  = sub_df.groupby("id")["prediction"].mean()
        grouped_label = sub_df.groupby("id")["label"].first()

        if len(grouped_label.unique()) < 2 or len(grouped_label) < 10:
            continue

        y_true = grouped_label.values.astype(int)
        y_real = grouped_real.values
        y_pred = grouped_pred.values

        prevalence = y_true.mean()

        # -- ROC curves --
        fpr_r, tpr_r, _ = roc_curve(y_true, y_real)
        fpr_p, tpr_p, _ = roc_curve(y_true, y_pred)
        auc_r = auc(fpr_r, tpr_r)
        auc_p = auc(fpr_p, tpr_p)

        ax_roc.plot(fpr_r, tpr_r, color='red',  linestyle='--', linewidth=2.5,
                    label=f'Real (AUC={auc_r:.3f})')
        ax_roc.plot(fpr_p, tpr_p, color='blue', linestyle='-',  linewidth=2.5,
                    label=f'Predicted (AUC={auc_p:.3f})')
        ax_roc.plot([0,1],[0,1], 'k--', alpha=0.4, linewidth=1)
        ax_roc.set_xlabel('False Positive Rate', fontsize=12)
        ax_roc.set_ylabel('True Positive Rate', fontsize=12)
        ax_roc.set_title(f'ROC — {g1} vs {g2}\nn={len(y_true)}', fontsize=12)
        ax_roc.set_xlim([0,1]); ax_roc.set_ylim([0,1])
        ax_roc.legend(fontsize=10)
        ax_roc.set_aspect('equal')

        # -- PR curves --
        prec_r, rec_r, _ = precision_recall_curve(y_true, y_real)
        prec_p, rec_p, _ = precision_recall_curve(y_true, y_pred)
        ap_r = average_precision_score(y_true, y_real)
        ap_p = average_precision_score(y_true, y_pred)

        ax_pr.plot(rec_r, prec_r, color='red',  linestyle='--', linewidth=2.5,
                   label=f'Real (AP={ap_r:.3f})')
        ax_pr.plot(rec_p, prec_p, color='blue', linestyle='-',  linewidth=2.5,
                   label=f'Predicted (AP={ap_p:.3f})')
        # Baseline: random classifier
        ax_pr.axhline(y=prevalence, color='gray', linestyle=':', linewidth=1.5,
                      label=f'Baseline (prev={prevalence:.3f})')
        ax_pr.set_xlabel('Recall', fontsize=12)
        ax_pr.set_ylabel('Precision', fontsize=12)
        ax_pr.set_title(f'PR Curve — {g1} vs {g2}\nn={len(y_true)}', fontsize=12)
        ax_pr.set_xlim([0,1]); ax_pr.set_ylim([0,1])
        ax_pr.legend(fontsize=10)
        ax_pr.set_aspect('equal')

    plt.tight_layout()
    bin_label = year_bin.replace('+','plus').replace('-','to')
    plt.savefig(f"./miccai26/SPAREAD_ROC_PR_Curves_{bin_label}yr.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"./miccai26/SPAREAD_ROC_PR_Curves_{bin_label}yr.svg", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved ROC+PR figure for {year_bin} year bin")

# ================================================================
# SUMMARY TABLE — manuscript focal bins only (0-1 and 2-3 years)
# ================================================================
print("\n" + "="*80)
print("MANUSCRIPT SUMMARY — 0-1 and 2-3 year bins")
print("="*80)

focal = results_tobs_df[results_tobs_df["year_bin"].isin(["0-1","2-3"])]
cols  = [
    "year_bin","comparison","N","Prevalence",
    "ROCAUC_Pred","ROCAUC_SE_Pred",
    "PRAUC_Pred","PRAUC_SE_Pred","PRAUC_Baseline",
    "F1_Pred","Precision_Pred","Recall_Pred",
    "TPR_Pred_Youden","PPV_Pred_Youden",
    "ROCAUC_Real","ROCAUC_SE_Real",
    "PRAUC_Real","PRAUC_SE_Real",
    "F1_Real","Precision_Real","Recall_Real",
]
print(focal[cols].to_string(index=False))
focal[cols].to_csv("./miccai26/SPAREAD_ManuscriptFocal_AllMetrics.csv", index=False)

print("\nAll files saved to ./miccai26/")
print("Analysis complete.")