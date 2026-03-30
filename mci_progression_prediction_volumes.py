# ==============================================================================
# Experiment: MCI Stable vs MCI Progressor Classification
#             Using Rate-of-Change in 145 Brain ROI Volumes
#             DKGP Predicted Trajectories vs RNN-AD Predicted Trajectories
# ==============================================================================
#
# OVERVIEW
# --------
# This script benchmarks p-DKGP-predicted brain trajectory slopes against
# RNN-AD-predicted slopes (Nguyen et al. 2020) for the task of classifying
# MCI subjects as stable (MCI→MCI) or progressing to AD (MCI→AD). For each
# subject and each of 145 MUSE brain ROIs, the OLS slope of the longitudinal
# trajectory is computed as the feature. Four feature sets are evaluated:
#
#   1. Baseline volumes       — static first-timepoint 145 volumes
#   2. RNN-AD predicted RoC   — slopes from RNN-AD predicted trajectories using cross-sectional data
#   3. DKGP predicted RoC     — slopes from p-DKGP predicted trajectories using cross-sectional data
#   4. Real RoC               — slopes from real observed trajectories (upper bound)
#
# MOTIVATION
# ----------
# Predicting MCI-to-AD conversion is the most clinically relevant prognostic
# task in Alzheimer's research. If trajectory prediction models capture true
# longitudinal dynamics, their predicted slopes should contain more information
# than static baseline volumes alone — and should approach the discriminative
# power of real observed slopes. Comparing p-DKGP against RNN-AD provides a
# head-to-head benchmark of two fundamentally different trajectory modelling
# approaches: p-DKGP (Gaussian process) vs RNN-AD (recurrent
# neural network).
#
# CLASSIFIER SELECTION
# --------------------
# Logistic Regression is applied to all four feature sets
# (Baseline, RNN-AD, DKGP, Real). Using the same classifier across all
# conditions ensures that any differences in ROC-AUC reflect the quality
# of the trajectory features rather than differences in classifier capacity.
#
# CLASSIFICATION PROTOCOL
# -----------------------
# - Nested cross-validation:
#     Outer loop — StratifiedKFold(5): 5 held-out test folds (20% each)
#     Inner loop — GridSearchCV with StratifiedKFold(3): hyperparameter tuning
#     The best inner model is refit on the full outer train split before
#     evaluation on the held-out test fold — no information leakage.
# - Classifiers evaluated: Logistic Regression, Random Forest, Gradient Boosting
# - Feature scaling: StandardScaler fit on outer train only, applied to test
# - Metrics: ROC-AUC (primary), PR-AUC, accuracy, precision, recall, F1, FDR, FPR
# - Mean ROC curve: per-fold TPR interpolated onto a common FPR grid (300 points)
#
# SIGNIFICANCE TESTING — DELONG'S TEST
# --------------------------------------
# DeLong's test (DeLong, DeLong & Clarke-Pearson, Biometrics 1988) is used to
# compare ROC-AUC between feature sets. This is the gold standard for correlated
# AUC comparison — it exploits the full paired structure of the predictions
# (each subject contributes one probability score per model), giving far more
# power than a t-test on 5 fold AUCs.
#
# Three pairwise comparisons are run:
#   1. p-DKGP predicted   vs. Baseline volumes 
#   2. RNN-AD predicted vs. Baseline volumes
#   3. p-DKGP predicted   vs. RNN-AD predicted 
#
#
# DATA
# ----
# - RNN-AD predictions  : ../Standalone_Nguyen2020_RNNAD/nataging/
#                         RNN_AD_Consolidated.csv
#                         (columns: id, time, y_H_MUSE_Volume_*, score_H_MUSE_Volume_*)
# - DKGP predictions    : ./nataging/OldHarmonizedMUSEROIs.csv
#                         (same column structure as RNN-AD)
# - Longitudinal covars : longitudinal_covariates_subjectsamples_longclean_
#                         hmuse_convs_allstudies.csv
#                         (Diagnosis per visit, used to derive progression labels)
# - Baseline volumes    : ./nataging/HarmonizedROIVolumes.csv
#                         (first timepoint y_H_MUSE_Volume_* per subject)
# - Progression labels  : derived from first/last Diagnosis per subject:
#                         MCI→MCI = MCI Stable (label 0)
#                         MCI→AD  = MCI Progressor (label 1)
#
#
# OUTPUTS
# -------
# Figures (all saved to ./miccai26/):
#   RNNAD_DKGP_mci_stable_vs_progressor_comprehensive_roc_curves.{png,pdf,svg}
#       4-curve ROC figure: Baseline [GBM], RNN-AD [LR worst], DKGP [LR best],
#       Real UB [RF], with AUC ± std in legend and asymmetric classifier footnote
#   RNNAD_DKGP_slope_quality_comparison.{png,svg}
#       Histogram comparing real vs predicted slope distributions (both models)
#
# CSVs / text files:
#   ./nataging/rate_of_change_per_roi_rnnad.csv
#       Per-subject per-ROI OLS slopes (real and RNN-AD predicted)
#   ./nataging/rate_of_change_per_roi_dkgp.csv
#       Per-subject per-ROI OLS slopes (real and DKGP predicted)
#   ./nataging/RNNAD_DKGP_rate_of_change_mci_stable_mci_progressor.txt
#       Full console output: dataset info, slope quality, CV results,
#       DeLong test results, comprehensive metrics table
#
# Cache:
#   ./nataging/mci_classification_results_cache.pkl
#       Pickled dict of all four evaluate_features() result dicts, used by
#       --plot-only mode to regenerate figures without rerunning classification
#
# Plot-only mode:
#   Run with --plot-only flag to regenerate figures from the cached results
#   without repeating the full nested CV (which can take 30+ minutes).
# ==============================================================================
 
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, roc_curve, roc_auc_score, classification_report,
    precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.inspection import permutation_importance
from sklearn.exceptions import ConvergenceWarning
from scipy import stats
from textwrap import wrap
from sklearn.metrics import (
    accuracy_score, roc_curve, roc_auc_score, average_precision_score,
    classification_report, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.calibration import calibration_curve
from statsmodels.stats.multitest import multipletests
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=FutureWarning, module='seaborn')
warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn.svm._base')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*overflow encountered.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered.*')
warnings.filterwarnings('ignore', message='.*findfont: Font family.*not found.*')
warnings.filterwarnings('ignore', message='.*Font family.*not found.*')
warnings.filterwarnings('ignore', message='.*seaborn styles shipped by Matplotlib.*')

# FIX 1: use the non-deprecated style name
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'DejaVu Sans',
    'font.sans-serif': ['DejaVu Sans'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16,
    'figure.dpi': 300
})


CLASSIFIER = 'Logistic Regression' 


def plot_comprehensive_roc_curves(baseline_results, rnn_pred_results, dkgp_pred_results,
                                   real_results, save_prefix='comprehensive_comparison'):
    """
    Create publication-quality 4-curve ROC figure for Results section.
    Shows: Baseline [GBM], RNN-AD [GBM worst], DKGP [LR best], Real UB [RF].
    Clean design: no grid, despined axes, AUC ± std in legend.
    """
    sns.set(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    # Ordered from bottom to top visually; colors chosen for accessibility
    methods = [
        ('Baseline volumes',          baseline_results,    CLASSIFIER, '#9B59B6', (5, 2),     1.8),
        ('RNN-AD predicted',          rnn_pred_results,    CLASSIFIER,    '#E74C3C', (4, 2, 1, 2), 2.0),
        ('DKGP predicted (ours)',     dkgp_pred_results,   CLASSIFIER,   '#1A5276', 'solid',      2.5),
        ('Real trajectories (upper bound)', real_results,  CLASSIFIER,  '#7F8C8D', (2, 2),       1.8),
    ]

    for label, results, clf, color, dashes, lw in methods:
        if results is None:
            continue
        cr  = results[clf]
        fpr = cr['fpr']
        tpr = cr['tpr']
        auc = cr['roc_auc']
        std = cr['roc_auc_std']

        line, = ax.plot(fpr, tpr, color=color, linewidth=lw,
                        label=f'{label}  (AUC = {auc:.3f} ± {std:.3f})')
        if dashes != 'solid':
            line.set_dashes(dashes)

    # Chance line
    ax.plot([0, 1], [0, 1], color='#AAAAAA', linewidth=1.2,
            linestyle='--', alpha=0.7, label='Chance level')

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel('False Positive Rate', fontsize=15)
    ax.set_ylabel('True Positive Rate',  fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=13,
                   width=1.2, length=5, direction='out')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('normal')

    legend = ax.legend(
        title='n = 882 MCI subjects',
        title_fontsize=10,
        fontsize=10,
        loc='lower right',
        frameon=True,
        framealpha=0.92,
        edgecolor='#CCCCCC',
        handlelength=2.4,
        handletextpad=0.6
    )
    legend.get_title().set_color('#555555')

    # Footnote below legend about asymmetric design
    ax.annotate(
        f'RNN-AD: {CLASSIFIER} | DKGP: {CLASSIFIER}',
        xy=(0.99, 0.01), xycoords='axes fraction',
        fontsize=8, color='#888888',
        ha='right', va='bottom', style='italic'
    )

    sns.despine()
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(f'./miccai26/RNNAD_DKGP_{save_prefix}_comprehensive_roc_curves.png',
                dpi=600, bbox_inches='tight')
    plt.savefig(f'./miccai26/RNNAD_DKGP_{save_prefix}_comprehensive_roc_curves.pdf',
                bbox_inches='tight')
    plt.savefig(f'./miccai26/RNNAD_DKGP_{save_prefix}_comprehensive_roc_curves.svg',
                bbox_inches='tight')
    plt.close()
    print(f"✓ ROC curves saved — all four conditions "
          f"(Baseline [GBM] | RNN-AD [{CLASSIFIER}] | "
          f"DKGP [{CLASSIFIER}] | Real UB [{CLASSIFIER}])")


def load_baseline_volumes():
    """Load baseline raw volumes for 145 ROIs from the first timepoint of each subject."""
    print("Loading baseline volumes...")
    try:
        df = pd.read_csv('./nataging/HarmonizedROIVolumes.csv')
        volume_cols = [col for col in df.columns if col.startswith('y_H_MUSE_Volume')]
        baseline_data = []
        for subject in df['id'].unique():
            subject_data = df[df['id'] == subject].sort_values('time')
            if len(subject_data) > 0:
                first_timepoint = subject_data.iloc[0]
                row_data = {'subject_id': subject, 'time': first_timepoint['time']}
                for col in volume_cols:
                    row_data[col] = first_timepoint[col]
                baseline_data.append(row_data)
        baseline_df = pd.DataFrame(baseline_data)
        print(f"Loaded baseline volumes for {len(baseline_df)} subjects")
        return baseline_df
    except Exception as e:
        print(f"Error loading baseline volumes: {e}")
        return None


def perform_false_discovery_analysis(y_true, y_pred, y_prob, alpha=0.05):
    """Perform false discovery rate analysis."""
    contingency_table = confusion_matrix(y_true, y_pred)
    if contingency_table.shape == (2, 2):
        chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
    else:
        p_value = 1.0

    fdr = 1 - precision_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    _, p_adjusted, _, _ = multipletests([p_value], method='fdr_bh', alpha=alpha)

    return {
        'fdr':         fdr,
        'fpr':         fpr,
        'p_value':     p_value,
        'p_adjusted':  p_adjusted[0],
        'significant': p_adjusted[0] < alpha
    }


def delong_test(y_true, y_prob_a, y_prob_b):
    """
    DeLong's test for comparing two correlated ROC curves.

    Reference: DeLong, DeLong & Clarke-Pearson (1988), Biometrics 44(3):837-845.

    This is the gold standard for comparing AUCs from two classifiers evaluated
    on the same set of subjects. It exploits the full paired structure of the
    predictions — each subject contributes one probability score per model —
    giving far more power than a t-test on 5 fold AUCs.

    Parameters
    ----------
    y_true   : array-like, shape (n,)  — binary ground truth labels
    y_prob_a : array-like, shape (n,)  — predicted probabilities from model A
    y_prob_b : array-like, shape (n,)  — predicted probabilities from model B

    Returns
    -------
    dict with keys: auc_a, auc_b, z_stat, p_value, significant
    """
    y_true   = np.asarray(y_true)
    y_prob_a = np.asarray(y_prob_a)
    y_prob_b = np.asarray(y_prob_b)

    def compute_midrank(x):
        """Compute midranks for the DeLong structural component."""
        J      = np.argsort(x)
        Z      = x[J]
        N      = len(x)
        T      = np.zeros(N, dtype=float)
        i      = 0
        while i < N:
            j = i
            while j < N - 1 and Z[j] == Z[j + 1]:
                j += 1
            T[i:j+1] = 0.5 * (i + j + 2)   # 1-based midrank
            i = j + 1
        T2 = np.empty(N, dtype=float)
        T2[J] = T
        return T2

    def structural_components(y_true, y_prob):
        """
        Compute the structural components V10 and V01 used in DeLong's variance.
        V10[i] = P(score of positive i > score of random negative)  — placement value
        V01[j] = P(score of random positive > score of negative j)  — placement value
        """
        pos_idx = np.where(y_true == 1)[0]
        neg_idx = np.where(y_true == 0)[0]
        m = len(pos_idx)   # number of positives
        n = len(neg_idx)   # number of negatives

        pos_scores = y_prob[pos_idx]
        neg_scores = y_prob[neg_idx]

        # Placement values for positives (V10)
        V10 = np.zeros(m)
        for i, ps in enumerate(pos_scores):
            V10[i] = np.mean(ps > neg_scores) + 0.5 * np.mean(ps == neg_scores)

        # Placement values for negatives (V01)
        V01 = np.zeros(n)
        for j, ns in enumerate(neg_scores):
            V01[j] = np.mean(pos_scores > ns) + 0.5 * np.mean(pos_scores == ns)

        auc = np.mean(V10)
        return auc, V10, V01, m, n

    auc_a, V10_a, V01_a, m, n = structural_components(y_true, y_prob_a)
    auc_b, V10_b, V01_b, _, _ = structural_components(y_true, y_prob_b)

    # Covariance matrix of [AUC_a, AUC_b] using DeLong's variance estimator
    # S = (1/m)*S10 + (1/n)*S01  where S10, S01 are 2x2 covariance matrices
    S10 = np.cov(np.vstack([V10_a, V10_b]))   # shape (2,2)
    S01 = np.cov(np.vstack([V01_a, V01_b]))   # shape (2,2)

    S = S10 / m + S01 / n   # variance of [AUC_a, AUC_b]

    # Variance of the difference AUC_a - AUC_b
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]

    if var_diff <= 0:
        return {
            'auc_a': auc_a, 'auc_b': auc_b,
            'z_stat': 0.0, 'p_value': 1.0, 'significant': False,
            'note': 'var_diff <= 0 — cannot compute z-statistic'
        }

    z_stat  = (auc_a - auc_b) / np.sqrt(var_diff)
    p_value = 2 * stats.norm.sf(abs(z_stat))   # two-sided

    return {
        'auc_a':       auc_a,
        'auc_b':       auc_b,
        'z_stat':      z_stat,
        'p_value':     p_value,
        'significant': p_value < 0.05
    }


def evaluate_features(X, y, feature_type):
    """
    Evaluate features using a nested cross-validation scheme:

    Outer loop — StratifiedKFold(5):
        Each fold holds out 20% as a clean test set, never seen during training.

    Inner loop — for each outer train split (80%):
        A GridSearchCV with StratifiedKFold(3) tunes hyperparameters on train/val.
        The best estimator is then refit on the full 80% and evaluated on the 20% test.

    This gives 5 held-out test AUCs per classifier. The mean ± std across those
    5 folds is the reported performance, and the 5 fold AUCs are stored for the
    paired t-test in perform_statistical_comparison.

    The mean ROC curve is built by interpolating each fold's TPR onto a common
    FPR grid — the standard publication approach.
    """
    from sklearn.model_selection import StratifiedKFold, GridSearchCV

    print(f"\n{'='*80}")
    print(f"ANALYSIS USING {feature_type.upper()} FEATURES")
    print(f"{'='*80}")
    print(f"Scheme: 5-fold outer CV | inner GridSearchCV(3-fold) for hyperparameter tuning")

    X = np.array(X)
    y = np.array(y)

    # ── Outer CV: 5 stratified folds ────────────────────────────────────────
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Model search spaces ──────────────────────────────────────────────────
    model_configs = {
        'Logistic Regression': {
            'estimator': LogisticRegression(
                max_iter=2000, class_weight='balanced', random_state=42),
            'param_grid': {
                'C': [0.01, 0.1, 1.0, 10.0]
            }
        },
        'Random Forest': {
            'estimator': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'param_grid': {
                'n_estimators': [100, 200],
                'max_depth':    [None, 5, 10]
            }
        },
        'Gradient Boosting': {
            'estimator': GradientBoostingClassifier(random_state=42),
            'param_grid': {
                'n_estimators':  [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth':     [3, 5]
            }
        }
    }

    # Common FPR grid for mean ROC construction
    mean_fpr = np.linspace(0, 1, 300)

    results = {}
    for name, config in model_configs.items():
        print(f"\n{name}:")
        print("-" * len(name))

        # Per-fold accumulators
        fold_aucs       = []
        fold_pr_aucs    = []
        fold_accuracies = []
        fold_precisions = []
        fold_recalls    = []
        fold_f1s        = []
        fold_fdrs       = []
        fold_fprs_stat  = []
        tprs_interp     = []
        best_params_log = []
        # Store raw predictions for DeLong's test (pooled across folds)
        all_y_true      = []
        all_y_prob      = []

        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
            X_train_outer, X_test = X[train_idx], X[test_idx]
            y_train_outer, y_test = y[train_idx], y[test_idx]

            # Scale: fit only on outer train, apply to both
            scaler         = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_outer)
            X_test_scaled  = scaler.transform(X_test)

            # Inner GridSearchCV on the outer train split
            gs = GridSearchCV(
                estimator  = config['estimator'],
                param_grid = config['param_grid'],
                cv         = inner_cv,
                scoring    = 'roc_auc',
                n_jobs     = -1,
                refit      = True   # refit best estimator on full outer train
            )
            gs.fit(X_train_scaled, y_train_outer)
            best_params_log.append(gs.best_params_)

            # Evaluate best model on held-out test set
            best_model = gs.best_estimator_
            y_pred = best_model.predict(X_test_scaled)
            y_prob = (best_model.predict_proba(X_test_scaled)[:, 1]
                      if hasattr(best_model, "predict_proba")
                      else best_model.decision_function(X_test_scaled))

            # Accumulate metrics
            fold_aucs.append(roc_auc_score(y_test, y_prob))
            fold_pr_aucs.append(average_precision_score(y_test, y_prob))
            fold_accuracies.append(accuracy_score(y_test, y_pred))
            fold_precisions.append(precision_score(y_test, y_pred, zero_division=0))
            fold_recalls.append(recall_score(y_test, y_pred, zero_division=0))
            fold_f1s.append(f1_score(y_test, y_pred, zero_division=0))

            fdr_res = perform_false_discovery_analysis(y_test, y_pred, y_prob)
            fold_fdrs.append(fdr_res['fdr'])
            fold_fprs_stat.append(fdr_res['fpr'])

            # Store raw predictions for DeLong (pooled across folds)
            all_y_true.extend(y_test.tolist())
            all_y_prob.extend(y_prob.tolist())

            # Interpolate TPR onto common FPR grid
            fpr_fold, tpr_fold, _ = roc_curve(y_test, y_prob)
            tpr_i = np.interp(mean_fpr, fpr_fold, tpr_fold)
            tpr_i[0] = 0.0
            tprs_interp.append(tpr_i)

            print(f"  Fold {fold_idx}: AUC = {fold_aucs[-1]:.3f}  "
                  f"best params = {gs.best_params_}")

        # Mean ROC curve across 5 folds
        mean_tpr     = np.mean(tprs_interp, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr      = np.std(tprs_interp, axis=0)

        mean_auc = np.mean(fold_aucs)
        std_auc  = np.std(fold_aucs)

        print(f"\n  → Mean ROC-AUC : {mean_auc:.3f} ± {std_auc:.3f}  [5 held-out test folds]")
        print(f"  → Mean PR-AUC  : {np.mean(fold_pr_aucs):.3f} ± {np.std(fold_pr_aucs):.3f}")
        print(f"  → Mean Accuracy: {np.mean(fold_accuracies):.3f} ± {np.std(fold_accuracies):.3f}")
        print(f"  → Mean Precision: {np.mean(fold_precisions):.3f} ± {np.std(fold_precisions):.3f}")
        print(f"  → Mean Recall  : {np.mean(fold_recalls):.3f} ± {np.std(fold_recalls):.3f}")
        print(f"  → Mean F1      : {np.mean(fold_f1s):.3f} ± {np.std(fold_f1s):.3f}")
        print(f"\n  False Discovery Analysis (mean across folds):")
        print(f"    FDR: {np.mean(fold_fdrs):.3f} ± {np.std(fold_fdrs):.3f}")
        print(f"    FPR: {np.mean(fold_fprs_stat):.3f} ± {np.std(fold_fprs_stat):.3f}")

        results[name] = {
            'roc_auc':      mean_auc,
            'roc_auc_std':  std_auc,
            'pr_auc':       np.mean(fold_pr_aucs),
            'pr_auc_std':   np.std(fold_pr_aucs),
            'accuracy':     np.mean(fold_accuracies),
            'precision':    np.mean(fold_precisions),
            'recall':       np.mean(fold_recalls),
            'f1':           np.mean(fold_f1s),
            'fdr':          np.mean(fold_fdrs),
            'fpr_stat':     np.mean(fold_fprs_stat),
            'fold_aucs':    np.array(fold_aucs),    # 5 values for paired t-test
            'fpr':          mean_fpr,               # mean ROC x-axis
            'tpr':          mean_tpr,               # mean ROC y-axis
            'tpr_std':      std_tpr,                # ± band
            'best_params':  best_params_log,        # hyperparams per fold
            'all_y_true':   np.array(all_y_true),   # pooled labels  — for DeLong
            'all_y_prob':   np.array(all_y_prob),   # pooled probs   — for DeLong
        }

    return results


def compute_95ci(fold_aucs):
    """
    Compute 95% CI for mean AUC from k-fold CV results using
    the t-distribution with df = k - 1.

    Parameters
    ----------
    fold_aucs : array-like, shape (k,) — per-fold AUC values

    Returns
    -------
    mean, ci_lower, ci_upper
    """
    from scipy import stats
    fold_aucs = np.asarray(fold_aucs)
    k         = len(fold_aucs)
    mean      = np.mean(fold_aucs)
    se        = stats.sem(fold_aucs)          # std / sqrt(k)
    t_crit    = stats.t.ppf(0.975, df=k - 1) # two-sided, df = k-1 = 4
    margin    = t_crit * se
    return mean, mean - margin, mean + margin

def perform_statistical_comparison(baseline_results, rnn_results, dkgp_results,
                                    real_results,
                                    alpha=0.05):
    """
    Statistical comparison across all four feature sets using DeLong's test
    on pooled CV predictions (n ≈ 882).

    Three pairwise comparisons are run:
      1. DKGP predicted   vs. Baseline volumes  (primary claim)
      2. RNN-AD predicted vs. Baseline volumes  (secondary claim)
      3. DKGP predicted   vs. RNN-AD predicted  (head-to-head)

    All comparisons use the same classifier applied symmetrically, guaranteeing
    that differences in AUC reflect trajectory quality rather than classifier choice.

    DeLong's test requires row-level alignment of pooled prediction vectors —
    i.e., the same subjects appearing in the same order across all feature sets.
    This is enforced by the assertion below.

    Parameters
    ----------
    baseline_results : dict  — output of evaluate_features for baseline volumes
    rnn_results      : dict  — output of evaluate_features for RNN-AD predicted slopes
    dkgp_results     : dict  — output of evaluate_features for DKGP predicted slopes
    real_results     : dict  — output of evaluate_features for real trajectory slopes
    classifier       : str   — must be the same classifier used across all feature sets
    alpha            : float — significance threshold
    """
    print("\n" + "="*80)
    print("DELONG'S TEST — ALL PAIRWISE COMPARISONS")
    print(f"Classifier : {CLASSIFIER} (applied symmetrically to all feature sets)")
    print(f"n          ≈ {len(dkgp_results[CLASSIFIER]['all_y_true'])} subjects (pooled CV)")
    print("="*80)

    # ── Extract pooled prediction vectors ─────────────────────────────────────
    y_true_base = baseline_results[CLASSIFIER]['all_y_true']
    y_true_rnn  = rnn_results[CLASSIFIER]['all_y_true']
    y_true_dkgp = dkgp_results[CLASSIFIER]['all_y_true']

    y_prob_base = baseline_results[CLASSIFIER]['all_y_prob']
    y_prob_rnn  = rnn_results[CLASSIFIER]['all_y_prob']
    y_prob_dkgp = dkgp_results[CLASSIFIER]['all_y_prob']

    # ── Alignment assertion ───────────────────────────────────────────────────
    # DeLong's test is only valid when all y_true vectors are identical —
    # i.e., the same subjects appeared in the same held-out test sets
    # across all three evaluate_features calls (guaranteed by random_state=42).
    assert np.array_equal(y_true_base, y_true_rnn) and \
           np.array_equal(y_true_rnn,  y_true_dkgp), \
        "y_true vectors differ across feature sets — CV splits are misaligned. " \
        "Ensure all calls to evaluate_features use random_state=42 and receive " \
        "subjects in the same order."

    y_true = y_true_dkgp   # all identical; use any one

    # ── AUC summary ──────────────────────────────────────────────────────────
    auc_base = baseline_results[CLASSIFIER]['roc_auc']
    auc_rnn  = rnn_results[CLASSIFIER]['roc_auc']
    auc_dkgp = dkgp_results[CLASSIFIER]['roc_auc']
    auc_real = real_results[CLASSIFIER]['roc_auc']

    std_base = baseline_results[CLASSIFIER]['roc_auc_std']
    std_rnn  = rnn_results[CLASSIFIER]['roc_auc_std']
    std_dkgp = dkgp_results[CLASSIFIER]['roc_auc_std']
    std_real = real_results[CLASSIFIER]['roc_auc_std']

    # ── AUC summary with 95% CI ───────────────────────────────────────────────
    print(f"\n  AUC Summary (mean [95% CI] across 5 folds):")

    for label, results, classifier in [
        ("Real upper bound", real_results, CLASSIFIER),
        ("DKGP predicted",   dkgp_results, CLASSIFIER),
        ("RNN-AD predicted", rnn_results, CLASSIFIER),
        ("Baseline volumes", baseline_results, CLASSIFIER),
    ]:
        if results is None:
            continue
        fold_aucs          = results[classifier]['fold_aucs']
        mean, lower, upper = compute_95ci(fold_aucs)
        print(f"    {label:<22}: {mean:.4f} [95% CI: {lower:.4f}, {upper:.4f}]")



    # ── Pairwise DeLong tests ─────────────────────────────────────────────────
    comparisons = [
        ("DKGP predicted",   "Baseline volumes",  y_prob_dkgp, y_prob_base),
        ("RNN-AD predicted", "Baseline volumes",  y_prob_rnn,  y_prob_base),
        ("DKGP predicted",   "RNN-AD predicted",  y_prob_dkgp, y_prob_rnn),
    ]

    delong_results = {}
    for name_a, name_b, prob_a, prob_b in comparisons:
        key = f"{name_a} vs {name_b}"
        print(f"\n{'─'*60}")
        print(f"  {key}")
        print(f"{'─'*60}")

        res = delong_test(y_true, prob_a, prob_b)
        delong_results[key] = res

        delta = res['auc_a'] - res['auc_b']
        print(f"  AUC ({name_a:<22}): {res['auc_a']:.4f}")
        print(f"  AUC ({name_b:<22}): {res['auc_b']:.4f}")
        print(f"  ΔAUC                      : {delta:+.4f}")
        print(f"  z-statistic               : {res['z_stat']:.3f}")
        print(f"  p-value                   : {res['p_value']:.4e}")
        print(f"  Significant (α = {alpha})  : {'✓ Yes' if res['significant'] else '✗ No'}")
        if 'note' in res:
            print(f"  Note: {res['note']}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("SUMMARY TABLE")
    print(f"{'─'*60}")
    header = f"  {'Comparison':<40} {'ΔAUC':>8}  {'z':>7}  {'p':>12}  {'Sig':>5}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key, res in delong_results.items():
        delta = res['auc_a'] - res['auc_b']
        sig   = '✓' if res['significant'] else '✗'
        print(f"  {key:<40} {delta:>+8.4f}  {res['z_stat']:>7.3f}  "
              f"{res['p_value']:>12.4e}  {sig:>5}")

    # ── Manuscript-ready conclusion ───────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("MANUSCRIPT CONCLUSION")
    print(f"{'─'*60}")

    dkgp_vs_base = delong_results["DKGP predicted vs Baseline volumes"]
    rnn_vs_base  = delong_results["RNN-AD predicted vs Baseline volumes"]
    dkgp_vs_rnn  = delong_results["DKGP predicted vs RNN-AD predicted"]

    for label, res, name_a, name_b in [
        ("DKGP vs Baseline", dkgp_vs_base, "DKGP",   "Baseline"),
        ("RNN vs Baseline",  rnn_vs_base,  "RNN-AD",  "Baseline"),
        ("DKGP vs RNN-AD",   dkgp_vs_rnn,  "DKGP",   "RNN-AD"),
    ]:
        sig_str = "significantly outperforms" if res['significant'] else \
                  "does not significantly outperform"
        print(f"  {label}: {name_a} {sig_str} {name_b} "
              f"(ΔAUC = {res['auc_a'] - res['auc_b']:+.4f}, "
              f"z = {res['z_stat']:.3f}, p = {res['p_value']:.4e})")

    return {
        'delong_dkgp_vs_baseline': dkgp_vs_base,
        'delong_rnn_vs_baseline':  rnn_vs_base,
        'delong_dkgp_vs_rnn':      dkgp_vs_rnn,
        'auc_summary': {
            'baseline': (auc_base, std_base),
            'rnn':      (auc_rnn,  std_rnn),
            'dkgp':     (auc_dkgp, std_dkgp),
            'real':     (auc_real, std_real),
        }
    }

def analyze_slope_quality(rate_of_change_df):
    """Analyze the quality of predicted slopes compared to real slopes."""
    print(f"\n{'='*80}")
    print("SLOPE QUALITY ANALYSIS")
    print(f"{'='*80}")

    valid_data = rate_of_change_df.dropna(subset=['real_slope', 'pred_slope'])
    if len(valid_data) == 0:
        print("No valid slope data found for analysis")
        return None

    real_stats = valid_data['real_slope'].describe()
    pred_stats = valid_data['pred_slope'].describe()

    print("\nReal Slope Statistics:")
    for k in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
        print(f"  {k}: {real_stats[k]:.3f}")

    print("\nPredicted Slope Statistics:")
    for k in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
        print(f"  {k}: {pred_stats[k]:.3f}")

    correlation    = valid_data['real_slope'].corr(valid_data['pred_slope'])
    r_squared      = correlation ** 2
    mae            = np.mean(np.abs(valid_data['real_slope'] - valid_data['pred_slope']))
    rmse           = np.sqrt(np.mean((valid_data['real_slope'] - valid_data['pred_slope'])**2))
    variance_ratio = (pred_stats['std'] / real_stats['std'])**2

    print(f"\nCorrelation between real and predicted slopes: {correlation:.3f}")
    print(f"R-squared: {r_squared:.3f}")
    print(f"Mean Absolute Error: {mae:.3f}")
    print(f"Root Mean Squared Error: {rmse:.3f}")
    print(f"Variance Ratio (Predicted/Real): {variance_ratio:.3f}")

    for col, label in [('num_timepoints', 'Number of Timepoints'),
                       ('time_range',     'Time Range (years)')]:
        s = valid_data[col].describe()
        print(f"\n{label} Statistics:")
        for k in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
            print(f"  {k}: {s[k]:.1f}")

    print(f"\n{'='*80}")
    print("SLOPE QUALITY ASSESSMENT")
    print(f"{'='*80}")

    if   correlation > 0.8: corr_quality = "Excellent"
    elif correlation > 0.6: corr_quality = "Good"
    elif correlation > 0.4: corr_quality = "Moderate"
    elif correlation > 0.2: corr_quality = "Poor"
    else:                   corr_quality = "Very Poor"
    print(f"Correlation Quality: {corr_quality} ({correlation:.3f})")

    if   0.8 < variance_ratio < 1.2: var_quality = "Good (variance preserved)"
    elif 0.5 < variance_ratio < 2.0: var_quality = "Moderate (some variance distortion)"
    else:                            var_quality = "Poor (significant variance distortion)"
    print(f"Variance Quality: {var_quality} (ratio: {variance_ratio:.3f})")

    if   mae < 0.01: error_quality = "Excellent"
    elif mae < 0.02: error_quality = "Good"
    elif mae < 0.05: error_quality = "Moderate"
    else:            error_quality = "Poor"
    print(f"Error Quality: {error_quality} (MAE: {mae:.3f})")

    print(f"\nOverall Slope Quality Assessment:")
    if correlation > 0.6 and 0.5 < variance_ratio < 2.0 and mae < 0.02:
        overall_quality = "Good"
        print(f"  The model shows good ability to predict rate of change")
    elif correlation > 0.4 and 0.3 < variance_ratio < 3.0 and mae < 0.05:
        overall_quality = "Moderate"
        print(f"  The model shows moderate ability to predict rate of change")
    else:
        overall_quality = "Poor"
        print(f"  The model shows limited ability to predict rate of change")
    print(f"  Overall Quality: {overall_quality}")

    plt.figure(figsize=(10, 6))
    plt.hist(valid_data['real_slope'], bins=50, alpha=0.5, label='Real Slopes',      density=True)
    plt.hist(valid_data['pred_slope'], bins=50, alpha=0.5, label='Predicted Slopes', density=True)
    plt.xlabel('Slope Value', fontweight='bold')
    plt.ylabel('Density',     fontweight='bold')
    plt.title('Distribution of Real vs Predicted Slopes', pad=20, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./miccai26/RNNAD_DKGP_slope_quality_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('./miccai26/RNNAD_DKGP_slope_quality_comparison.svg',              bbox_inches='tight')
    plt.close()
    print(f"\nSlope quality comparison plot saved.")

    return {
        'correlation':     correlation,
        'r_squared':       r_squared,
        'mae':             mae,
        'rmse':            rmse,
        'variance_ratio':  variance_ratio,
        'overall_quality': overall_quality
    }


def analyze_progressor_classification(rate_of_change_df, covariates_df, baseline_df=None):
    """Analyze progressor vs non-progressor classification using real and/or predicted data."""

    with open('./nataging/RNNAD_DKGP_rate_of_change_mci_stable_mci_progressor.txt', 'w') as f:
        import sys
        original_stdout = sys.stdout

        class TeeOutput:
            def __init__(self, file):
                self.file   = file
                self.stdout = sys.stdout
            def write(self, text):
                self.stdout.write(text)
                self.file.write(text)
            def flush(self):
                self.stdout.flush()
                self.file.flush()

        sys.stdout = TeeOutput(f)

        try:
            print("\n" + "="*80)
            print("DATASET INFORMATION")
            print("="*80)

            # ── Separate by model ────────────────────────────────────────────
            rnn_data  = rate_of_change_df[rate_of_change_df['model'] == 'RNN-AD']
            dkgp_data = rate_of_change_df[rate_of_change_df['model'] == 'DKGP']
            dkgp_data = dkgp_data.dropna(subset=['real_slope', 'pred_slope'])

            # ── ROI coverage check ───────────────────────────────────────────
            dkgp_roi_counts = dkgp_data.groupby('subject_id')['roi'].nunique()
            rnn_roi_counts  = rnn_data.groupby('subject_id')['roi'].nunique()
            print(f"DKGP subjects with < 145 ROIs: {(dkgp_roi_counts < 145).sum()}")
            print(f"RNN-AD subjects with < 145 ROIs: {(rnn_roi_counts < 145).sum()}")
            print(f"DKGP ROI count distribution:\n{dkgp_roi_counts.value_counts().sort_index()}")

            # ── NaN check ────────────────────────────────────────────────────
            for label, df in [('RNN-AD', rnn_data), ('DKGP', dkgp_data)]:
                n_nan_real  = df['real_slope'].isna().sum()
                n_nan_pred  = df['pred_slope'].isna().sum()
                n_subj_real = df[df['real_slope'].isna()]['subject_id'].nunique()
                n_subj_pred = df[df['pred_slope'].isna()]['subject_id'].nunique()
                print(f"\n[{label}] NaN check:")
                print(f"  real_slope NaNs: {n_nan_real} rows across {n_subj_real} subjects")
                print(f"  pred_slope NaNs: {n_nan_pred} rows across {n_subj_pred} subjects")
                print(f"  NaN ROIs (real): {sorted(df[df['real_slope'].isna()]['roi'].unique().tolist())}")
                print(f"  NaN ROIs (pred): {sorted(df[df['pred_slope'].isna()]['roi'].unique().tolist())}")

            print(f"\nRNN-AD data: {len(rnn_data)} rows")
            print(f"DKGP data: {len(dkgp_data)} rows")
            print(f"Number of ROIs: {len(rate_of_change_df['roi'].unique())}")

            # ── Pivot predicted slopes ───────────────────────────────────────
            rnn_pred_wide = rnn_data.pivot(
                index='subject_id', columns='roi', values='pred_slope'
            ).reset_index()
            rnn_pred_wide.columns = ['subject_id'] + [
                f'pred_roc_{col}' for col in rnn_pred_wide.columns[1:]
            ]

            dkgp_pred_wide = dkgp_data.pivot(
                index='subject_id', columns='roi', values='pred_slope'
            ).reset_index()
            dkgp_pred_wide.columns = ['subject_id'] + [
                f'pred_roc_{col}' for col in dkgp_pred_wide.columns[1:]
            ]

            # Real upper bound comes from RNN-AD file — same preprocessing
            # guarantees real AUC >= predicted AUC for RNN-AD
            real_wide = rnn_data.pivot(
                index='subject_id', columns='roi', values='real_slope'
            ).reset_index()
            real_wide.columns = ['subject_id'] + [
                f'real_roc_{col}' for col in real_wide.columns[1:]
            ]

            print(f"\nRNN-AD predicted subjects: {len(rnn_pred_wide)}")
            print(f"DKGP predicted subjects:   {len(dkgp_pred_wide)}")
            print(f"Real upper-bound subjects: {len(real_wide)}")

            # ── Intersect subjects across slope feature sets ──────────────────
            common_subjects = (
                set(rnn_pred_wide['subject_id'])
                & set(dkgp_pred_wide['subject_id'])
                & set(real_wide['subject_id'])
            )
            print(f"\nCommon subjects across slope feature sets: {len(common_subjects)}")

            rnn_pred_wide  = rnn_pred_wide[rnn_pred_wide['subject_id'].isin(common_subjects)]
            dkgp_pred_wide = dkgp_pred_wide[dkgp_pred_wide['subject_id'].isin(common_subjects)]
            real_wide      = real_wide[real_wide['subject_id'].isin(common_subjects)]

            # ── Slope quality ────────────────────────────────────────────────
            print("\n" + "="*80)
            print("SLOPE QUALITY ANALYSIS - RNN-AD MODEL")
            print("="*80)
            rnn_slope_quality = analyze_slope_quality(
                rate_of_change_df[rate_of_change_df['model'] == 'RNN-AD']
            )

            print("\n" + "="*80)
            print("SLOPE QUALITY ANALYSIS - DKGP MODEL")
            print("="*80)
            dkgp_slope_quality = analyze_slope_quality(
                rate_of_change_df[rate_of_change_df['model'] == 'DKGP']
            )

            if rnn_slope_quality and dkgp_slope_quality:
                print("\n" + "="*80)
                print("SLOPE QUALITY COMPARISON")
                print("="*80)
                print(f"  RNN-AD Correlation: {rnn_slope_quality['correlation']:.3f}  |  Overall: {rnn_slope_quality['overall_quality']}")
                print(f"  DKGP   Correlation: {dkgp_slope_quality['correlation']:.3f}  |  Overall: {dkgp_slope_quality['overall_quality']}")
                print(f"  Difference (DKGP - RNN-AD): {dkgp_slope_quality['correlation'] - rnn_slope_quality['correlation']:.3f}")
                better = "DKGP" if dkgp_slope_quality['correlation'] > rnn_slope_quality['correlation'] else "RNN-AD"
                print(f"\n  {better} shows better slope prediction quality")

            # ── Build progression labels ─────────────────────────────────────
            progression_data = []
            for subject in covariates_df['PTID'].unique():
                subject_data = covariates_df[covariates_df['PTID'] == subject].sort_values('Time')
                if len(subject_data) > 1:
                    initial_dx = subject_data.iloc[0]['Diagnosis']
                    final_dx   = subject_data.iloc[-1]['Diagnosis']
                    if isinstance(initial_dx, (int, float, np.int64, np.float64)) and \
                       isinstance(final_dx,   (int, float, np.int64, np.float64)):
                        if initial_dx == 1:
                            progression_data.append({
                                'PTID':              subject,
                                'initial_diagnosis': initial_dx,
                                'final_diagnosis':   final_dx,
                                'is_progressor':     int(initial_dx == 1 and final_dx == 2)
                            })

            progression_df   = pd.DataFrame(progression_data)
            overlap_subjects = set(progression_df['PTID']) & common_subjects
            progression_df   = progression_df[progression_df['PTID'].isin(overlap_subjects)]

            print(f"\nSubjects with progression data (starting as MCI): {len(set(progression_df['PTID']))}")
            print(f"Subjects with both progression and ROC data:       {len(overlap_subjects)}")
            print(f"\nFinal analysis dataset:")
            print(f"  Total:           {len(progression_df)}")
            print(f"  Progressors:     {progression_df['is_progressor'].sum()}")
            print(f"  Non-Progressors: {(progression_df['is_progressor'] == 0).sum()}")
            print(f"  Progression rate: {progression_df['is_progressor'].mean()*100:.1f}%")

            # ── Add PTID & merge labels ──────────────────────────────────────
            for df in [rnn_pred_wide, dkgp_pred_wide, real_wide]:
                df['PTID'] = df['subject_id']

            labels = progression_df[['PTID', 'is_progressor']]

            merged_real      = pd.merge(real_wide,      labels, on='PTID', how='inner')
            merged_pred_rnn  = pd.merge(rnn_pred_wide,  labels, on='PTID', how='inner')
            merged_pred_dkgp = pd.merge(dkgp_pred_wide, labels, on='PTID', how='inner')

            # ── Baseline: intersect with slope cohort ────────────────────────
            merged_baseline  = None
            baseline_results = None
            if baseline_df is not None:
                baseline_df['PTID'] = baseline_df['subject_id']
                baseline_df_filtered = baseline_df[
                    baseline_df['PTID'].isin(set(merged_real['PTID']))
                ]
                merged_baseline = pd.merge(baseline_df_filtered, labels, on='PTID', how='inner')
                print(f"\n  Baseline subjects (after cohort intersection): {len(merged_baseline)}")
            else:
                print("\n  Note: Baseline analysis skipped — baseline volumes not available")

            # ── Canonical subject ordering ────────────────────────────────────
            # All four feature sets must contain exactly the same subjects in
            # exactly the same row order so that StratifiedKFold(random_state=42)
            # produces identical fold assignments across all evaluate_features
            # calls — a hard requirement for DeLong's test row-level alignment.
            common_ptids = (
                set(merged_real['PTID'])
                & set(merged_pred_rnn['PTID'])
                & set(merged_pred_dkgp['PTID'])
                & (set(merged_baseline['PTID']) if merged_baseline is not None
                   else set(merged_real['PTID']))
            )
            print(f"\nSubjects in all feature sets (final intersection): {len(common_ptids)}")

            merged_real      = (merged_real[merged_real['PTID'].isin(common_ptids)]
                                .sort_values('PTID').reset_index(drop=True))
            merged_pred_rnn  = (merged_pred_rnn[merged_pred_rnn['PTID'].isin(common_ptids)]
                                .sort_values('PTID').reset_index(drop=True))
            merged_pred_dkgp = (merged_pred_dkgp[merged_pred_dkgp['PTID'].isin(common_ptids)]
                                .sort_values('PTID').reset_index(drop=True))
            if merged_baseline is not None:
                merged_baseline = (merged_baseline[merged_baseline['PTID'].isin(common_ptids)]
                                   .sort_values('PTID').reset_index(drop=True))

            # ── Label alignment verification ──────────────────────────────────
            assert (merged_real['is_progressor'].values ==
                    merged_pred_rnn['is_progressor'].values).all(), \
                "Label mismatch: real vs RNN-AD after sorting."
            assert (merged_real['is_progressor'].values ==
                    merged_pred_dkgp['is_progressor'].values).all(), \
                "Label mismatch: real vs DKGP after sorting."
            if merged_baseline is not None:
                assert (merged_real['is_progressor'].values ==
                        merged_baseline['is_progressor'].values).all(), \
                    "Label mismatch: real vs baseline after sorting."
            print(f"Label alignment verified across all feature sets ✓")
            print(f"Final cohort: {len(merged_real)} subjects  "
                  f"({merged_real['is_progressor'].sum()} progressors, "
                  f"{(merged_real['is_progressor']==0).sum()} non-progressors)")

            # # ── evaluate_features ────────────────────────────────────────────
            # # Classifier is fixed to Logistic Regression across all feature sets
            # # for symmetric DeLong comparisons.
            # CLASSIFIER = 'Logistic Regression'

            if merged_baseline is not None:
                baseline_results = evaluate_features(
                    merged_baseline.drop(
                        ['subject_id', 'PTID', 'is_progressor', 'time'],
                        axis=1, errors='ignore'
                    ),
                    merged_baseline['is_progressor'],
                    'Baseline'
                )

            real_results = evaluate_features(
                merged_real.drop(['subject_id', 'PTID', 'is_progressor'], axis=1),
                merged_real['is_progressor'],
                'Real (upper bound)'
            )

            pred_rnn_results = evaluate_features(
                merged_pred_rnn.drop(['subject_id', 'PTID', 'is_progressor'], axis=1),
                merged_pred_rnn['is_progressor'],
                'RNN-AD Predicted'
            )

            pred_dkgp_results = evaluate_features(
                merged_pred_dkgp.drop(['subject_id', 'PTID', 'is_progressor'], axis=1),
                merged_pred_dkgp['is_progressor'],
                'DKGP Predicted'
            )

            # ── Cache results ────────────────────────────────────────────────
            CACHE_PATH = './nataging/mci_classification_results_cache.pkl'
            os.makedirs('./nataging', exist_ok=True)
            with open(CACHE_PATH, 'wb') as f:
                pickle.dump({
                    'baseline_results':  baseline_results,
                    'pred_rnn_results':  pred_rnn_results,
                    'pred_dkgp_results': pred_dkgp_results,
                    'real_results':      real_results,
                }, f)
            print(f"\n✓ Results cached to {CACHE_PATH}")

            # ── ROC plot ─────────────────────────────────────────────────────
            plot_comprehensive_roc_curves(
                baseline_results,
                pred_rnn_results,
                pred_dkgp_results,
                real_results,
                save_prefix='mci_stable_vs_progressor'
            )

            # ── DeLong pairwise comparisons ──────────────────────────────────
            stats_results = perform_statistical_comparison(
                baseline_results = baseline_results,
                rnn_results      = pred_rnn_results,
                dkgp_results     = pred_dkgp_results,
                real_results     = real_results,
                alpha            = 0.05
            )

            # ── Comprehensive metrics table ───────────────────────────────────
            print(f"\n" + "="*80)
            print("COMPREHENSIVE METRICS SUMMARY (mean ± std across 5 folds)")
            print(f"Classifier: {CLASSIFIER} — applied symmetrically to all feature sets")
            print("="*80)

            method_list = [
                ("Real upper bound", real_results[CLASSIFIER]),
                ("DKGP predicted",   pred_dkgp_results[CLASSIFIER]),
                ("RNN-AD predicted", pred_rnn_results[CLASSIFIER]),
            ]
            if baseline_results is not None:
                method_list.append(("Baseline volumes", baseline_results[CLASSIFIER]))

            metrics = [
                ('roc_auc',   'AUC-ROC'),
                ('pr_auc',    'AUC-PR'),
                ('accuracy',  'Accuracy'),
                ('precision', 'Precision'),
                ('recall',    'Recall'),
                ('f1',        'F1 Score'),
                ('fdr',       'FDR'),
                ('fpr_stat',  'FPR'),
            ]

            col_w = 24
            print(f"\n  {'Metric':<14}", end="")
            for name, _ in method_list:
                print(f"  {name:<{col_w}}", end="")
            print()
            print("  " + "-" * (14 + (col_w + 2) * len(method_list)))

            for key, label in metrics:
                print(f"  {label:<14}", end="")
                for _, r in method_list:
                    val     = r.get(key, float('nan'))
                    std_key = f'{key}_std'
                    cell    = (f"{val:.3f}±{r[std_key]:.3f}"
                               if std_key in r else f"{val:.3f}")
                    print(f"  {cell:<{col_w}}", end="")
                print()

            # ── Delta table: DKGP vs RNN-AD and DKGP vs Baseline ─────────────
            dkgp_r = pred_dkgp_results[CLASSIFIER]
            rnn_r  = pred_rnn_results[CLASSIFIER]

            print(f"\n  Delta (DKGP predicted − RNN-AD predicted):")
            for key, label in metrics:
                dv = dkgp_r.get(key, float('nan'))
                rv = rnn_r.get(key, float('nan'))
                if not (np.isnan(dv) or np.isnan(rv)):
                    print(f"    {label:<14}: {dv - rv:+.3f}  "
                          f"{'↑' if dv > rv else '↓'}")

            if baseline_results is not None:
                base_r = baseline_results[CLASSIFIER]
                print(f"\n  Delta (DKGP predicted − Baseline volumes):")
                for key, label in metrics:
                    dv = dkgp_r.get(key, float('nan'))
                    bv = base_r.get(key, float('nan'))
                    if not (np.isnan(dv) or np.isnan(bv)):
                        print(f"    {label:<14}: {dv - bv:+.3f}  "
                              f"{'↑' if dv > bv else '↓'}")

            print("\n" + "="*80)
            print("ANALYSIS COMPLETE")
            print("="*80)

        finally:
            sys.stdout = original_stdout

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from pathlib import Path

    print("="*80)
    print("MCI STABLE VS MCI PROGRESSOR CLASSIFICATION EXPERIMENT")
    print("="*80)
    print("Experiment Design:")
    print("  - Target: Discriminate MCI Stable (1→1) vs MCI Progressor (1→2) subjects")
    print("  - Features: Rate of change in 145 brain regions")
    print("  - Models: RNN-AD predicted RoC vs DKGP predicted RoC")
    print("  - Baseline: Static brain volumes")
    print("  - Upper bound: Real rate of change (RNN-AD file)")
    print(f"  - Stress-test: RNN-AD worst [{CLASSIFIER}] vs DKGP best [{CLASSIFIER}]")
    print("="*80)

    # ── Paths ────────────────────────────────────────────────────────────────
    RNN_IN_FILE   = Path("../Standalone_Nguyen2020_RNNAD/nataging/RNN_AD_Consolidated.csv")
    DKGP_IN_FILE  = Path("./nataging/OldHarmonizedMUSEROIs.csv")
    RNN_OUT_FILE  = Path("./nataging/rate_of_change_per_roi_rnnad.csv")
    DKGP_OUT_FILE = Path("./nataging/rate_of_change_per_roi_dkgp.csv")

    def ols_slope(t, v):
        if len(t) < 2:
            return np.nan
        return np.polyfit(t, v, 1)[0]

    # ── RNN-AD ───────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("PROCESSING RNN-AD DATA")
    print("="*80)

    rnn_df = pd.read_csv(RNN_IN_FILE)
    print(f"  Loaded: {len(rnn_df):,} rows, {rnn_df.shape[1]} columns")
    print(f"  Time range: {rnn_df['time'].min():.1f} to {rnn_df['time'].max():.1f}")
    print(f"  Unique subjects: {rnn_df['id'].nunique()}")
    print(f"  Avg timepoints/subject: {len(rnn_df)/rnn_df['id'].nunique():.1f}")

    rnn_y_cols     = sorted([c for c in rnn_df.columns if c.startswith("y_H_MUSE_Volume_")],
                             key=lambda c: int(c.split("_")[-1]))
    rnn_score_cols = sorted([c for c in rnn_df.columns if c.startswith("score_H_MUSE_Volume_")],
                             key=lambda c: int(c.split("_")[-1]))
    assert len(rnn_y_cols) == len(rnn_score_cols) == 145, \
        "Expected 145 y_H_MUSE_Volume_ and 145 score_H_MUSE_Volume_ columns for RNN-AD"

    rnn_roi_indices = [int(c.split("_")[-1]) for c in rnn_y_cols]

    rnn_out = []
    for roi_idx, y_col, s_col in zip(rnn_roi_indices, rnn_y_cols, rnn_score_cols):
        sub_df = rnn_df[["id", "time", y_col, s_col]].dropna()
        for subj, g in sub_df.groupby("id", sort=False):
            t = g["time"].values.astype(float)
            y = g[y_col].values.astype(float)
            s = g[s_col].values.astype(float)
            rnn_out.append({
                "subject_id":         subj,
                "roi":                roi_idx,
                "real_slope":         ols_slope(t, y),
                "pred_slope":         ols_slope(t, s),
                "num_timepoints":     len(t),
                "time_range":         (t.max() - t.min()) if len(t) else np.nan,
                "real_initial_value": y[0] if len(y) else np.nan,
                "pred_initial_value": s[0] if len(s) else np.nan,
                "model":              "RNN-AD"
            })

    rnn_roc_df = pd.DataFrame(rnn_out)
    rnn_roc_df.to_csv(RNN_OUT_FILE, index=False)
    print(f"  RNN-AD RoC: {len(rnn_out):,} measurements, "
          f"{rnn_roc_df['subject_id'].nunique()} subjects → {RNN_OUT_FILE}")

    # ── DKGP ─────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("PROCESSING DKGP DATA")
    print("="*80)

    dkgp_df = pd.read_csv(DKGP_IN_FILE)
    print(f"  Loaded: {len(dkgp_df):,} rows, {dkgp_df.shape[1]} columns")
    print(f"  Time range: {dkgp_df['time'].min():.1f} to {dkgp_df['time'].max():.1f}")
    print(f"  Unique subjects: {dkgp_df['id'].nunique()}")

    dkgp_y_cols     = sorted([c for c in dkgp_df.columns if c.startswith("y_H_MUSE_Volume_")],
                              key=lambda c: int(c.split("_")[-1]))
    dkgp_score_cols = sorted([c for c in dkgp_df.columns if c.startswith("score_H_MUSE_Volume_")],
                              key=lambda c: int(c.split("_")[-1]))

    nan_mask     = dkgp_df[dkgp_y_cols + dkgp_score_cols].isnull().any(axis=1)
    bad_rows     = dkgp_df[nan_mask]
    bad_subjects = bad_rows['id'].unique().tolist()
    print(f"  Rows with NaN volumes: {len(bad_rows)}")
    print(f"  Subjects with NaN volumes: {len(bad_subjects)}")
    print(f"  Bad subject IDs: {bad_subjects}")
    dkgp_df = dkgp_df[~dkgp_df['id'].isin(bad_subjects)]

    assert len(dkgp_y_cols) == len(dkgp_score_cols) == 145, \
        "Expected 145 y_H_MUSE_Volume_ and 145 score_H_MUSE_Volume_ columns for DKGP"

    dkgp_roi_indices = [int(c.split("_")[-1]) for c in dkgp_y_cols]

    dkgp_out = []
    for roi_idx, y_col, s_col in zip(dkgp_roi_indices, dkgp_y_cols, dkgp_score_cols):
        sub_df = dkgp_df[["id", "time", y_col, s_col]].dropna()
        for subj, g in sub_df.groupby("id", sort=False):
            t = g["time"].values.astype(float)
            y = g[y_col].values.astype(float)
            s = g[s_col].values.astype(float)
            dkgp_out.append({
                "subject_id":         subj,
                "roi":                roi_idx,
                "real_slope":         ols_slope(t, y),
                "pred_slope":         ols_slope(t, s),
                "num_timepoints":     len(t),
                "time_range":         (t.max() - t.min()) if len(t) else np.nan,
                "real_initial_value": y[0] if len(y) else np.nan,
                "pred_initial_value": s[0] if len(s) else np.nan,
                "model":              "DKGP"
            })

    dkgp_roc_df = pd.DataFrame(dkgp_out)
    dkgp_roc_df.to_csv(DKGP_OUT_FILE, index=False)
    print(f"  DKGP RoC: {len(dkgp_out):,} measurements, "
          f"{dkgp_roc_df['subject_id'].nunique()} subjects → {DKGP_OUT_FILE}")

    # ── Combine & run ────────────────────────────────────────────────────────
    combined_roc_df = pd.concat([rnn_roc_df, dkgp_roc_df], ignore_index=True)
    print(f"\nCombined: {len(combined_roc_df):,} total measurements")

    covariates_df = pd.read_csv(
        '/home/cbica/Desktop/LongGPClustering/data1/'
        'longitudinal_covariates_subjectsamples_longclean_hmuse_convs_allstudies.csv'
    )
    print(f"Covariates: {len(covariates_df):,} rows, {covariates_df['PTID'].nunique()} subjects")

    baseline_df = load_baseline_volumes()

    analyze_progressor_classification(combined_roc_df, covariates_df, baseline_df)


# ── Plot-only mode: regenerate figure from cached results ────────────────────
# Usage: python manuscript1_mci_roc_corrected.py --plot-only
if __name__ == '__plot_only__' or (
        __name__ == '__main__' and
        len(__import__('sys').argv) > 1 and
        __import__('sys').argv[1] == '--plot-only'):
    import sys
    CACHE_PATH = './nataging/mci_classification_results_cache.pkl'
    if not os.path.exists(CACHE_PATH):
        print(f"✗ Cache not found at {CACHE_PATH}. Run the full script first.")
        sys.exit(1)
    print(f"Loading cached results from {CACHE_PATH} ...")
    with open(CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)
    os.makedirs('./miccai26', exist_ok=True)
    plot_comprehensive_roc_curves(
        cache['baseline_results'],
        cache['pred_rnn_results'],
        cache['pred_dkgp_results'],
        cache['real_results'],
        save_prefix='mci_stable_vs_progressor'
    )
    print("Done. Check ./miccai26/ for updated figures.")
