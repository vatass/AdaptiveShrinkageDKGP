# ==============================================================================
# Experiment: Multimodal Trajectory Classification of MCI Progression
#             MCI Stable vs MCI Progressors Using Rate-of-Change Features
# ==============================================================================
#
# OVERVIEW
# --------
# This script classifies whether an MCI subject will progress to Alzheimer's
# Disease (MCI Progressor) or remain stable (MCI Stable), using rates of
# change (RoC) derived from DKGP-predicted longitudinal trajectories across
# four modalities: 145 brain ROI volumes, SPARE-AD, ADAS-Cog, and MMSE.
# The central question is whether combining all modalities into a multimodal
# feature set yields better prognostic discrimination than any single
# modality alone.
#
# MOTIVATION
# ----------
# Predicting MCI-to-AD conversion is one of the most clinically valuable
# tasks in Alzheimer's research. Structural brain change (ROI volumes),
# imaging-based AD signatures (SPARE-AD), and cognitive decline (ADAS, MMSE)
# each capture a different aspect of the disease process. If the DKGP model
# produces trajectories that preserve these signals, then RoC features derived
# from predicted trajectories should support accurate progression prediction —
# and combining modalities should outperform any single source.
#
# EXPERIMENTAL DESIGN
# -------------------
# For each subject, a rate-of-change (OLS slope) is computed per biomarker
# from their predicted longitudinal trajectory. These slopes become the
# feature vector for classification. Four feature sets are evaluated:
#
#   1. 145 ROIs        : RoC of 145 MUSE volumetric brain regions (predicted)
#   2. SPARE-AD        : RoC of the SPARE-AD composite imaging score
#   3. Cognitive       : RoC of ADAS-Cog + MMSE scores
#   4. Multimodal      : All of the above concatenated (148 features total)
#
# Each feature set is evaluated with two classifiers:
#   - Logistic Regression (LR) — linear, interpretable
#   - Random Forest (RF)       — non-linear, ensemble
#
# Only subjects with complete data across ALL modalities are included,
# ensuring fair comparison across conditions.
#
# CLASSIFICATION PROTOCOL
# -----------------------
# - Nested cross-validation: outer 5-fold stratified CV for performance
#   estimation; inner 3-fold GridSearchCV for hyperparameter tuning
# - Threshold selection: Youden index estimated on inner OOF predictions
#   (data-leak-free — threshold never touches the outer test fold)
# - OOF probabilities: accumulated across outer folds for DeLong testing
# - Metrics: ROC-AUC (primary), PR-AUC, precision, recall, F1, specificity
#
# SIGNIFICANCE TESTING
# --------------------
# Two complementary tests compare multimodal vs each unimodal baseline:
#   1. Paired t-test on 5 outer-fold AUCs (within-fold pairing controls
#      for data-difficulty variability)
#   2. DeLong test (Sun & Xu 2014) on full OOF predicted probabilities
#      (more powerful — uses the complete probability distributions)
# Both tests apply Bonferroni correction across the 3 unimodal comparisons.
#
# DATA
# ----
# - Subjects        : ADNI cohort (MCI at baseline), multi-study longitudinal
# - Labels          : MCI Stable (Dx=1 throughout) vs MCI Progressor (Dx=1→2)
# - ROI features    : 145 MUSE predicted volumetric RoC (column "score")
# - SPARE-AD        : Predicted SPARE-AD score RoC
# - Cognitive       : Predicted ADAS-Cog + MMSE score RoC
# - Covariates      : Baseline age, sex (available but not used as features)
#
# OUTPUTS
# -------
# - ./nataging/Multimodal_LR_Significance.{png,pdf,svg}
#       Horizontal bar chart: LR ROC-AUC per modality with 95% CI error bars
#       and DeLong significance brackets (multimodal vs each unimodal)
# - ./nataging/significance_ttest.csv
#       Paired t-test results for LR + RF (Bonferroni-corrected)
# - ./nataging/significance_delong.csv
#       DeLong test results for LR + RF (Bonferroni-corrected)
# - ./common_subjects.npy
#       Array of subject IDs included in the analysis (all modalities present)
# ==============================================================================
 

import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_val_predict
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve
)
from scipy import stats


# =============================================================================
# DELONG TEST  (Hanley & McNeil 1983; Sun & Xu 2014 fast implementation)
# =============================================================================

def _compute_midrank(x):
    """Midranks of x (used in DeLong covariance estimation)."""
    J  = np.argsort(x)
    Z  = x[J]
    N  = len(x)
    T  = np.zeros(N, dtype=float)
    i  = 0
    while i < N:
        j = i
        while j < N - 1 and Z[j] == Z[j + 1]:
            j += 1
        T[i:j + 1] = 0.5 * (i + j + 2)
        i = j + 1
    T2    = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    Fast DeLong algorithm (Sun & Xu 2014).

    predictions_sorted_transposed : (n_classifiers, n_samples) array,
                                    samples sorted by label (positives first)
    label_1_count                 : number of positive examples

    Returns
    -------
    aucs      : (n_classifiers,) AUC estimates
    delongcov : (n_classifiers, n_classifiers) covariance matrix
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    k = predictions_sorted_transposed.shape[0]

    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)

    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = (tz[:, :m].sum(axis=1) - tx.sum(axis=1)) / (m * n)
    v01  = (tz[:, :m] - tx[:, :]) / n
    v10  = 1.0 - (tz[:, m:] - ty[:, :]) / m

    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true, prob_a, prob_b):
    """
    DeLong test for the difference in AUC between two classifiers on the
    same subjects.

    Parameters
    ----------
    y_true : array-like  binary ground-truth labels
    prob_a : array-like  predicted probabilities — classifier A (multimodal)
    prob_b : array-like  predicted probabilities — classifier B (unimodal)

    Returns
    -------
    auc_a, auc_b : float  AUC estimates
    z            : float  test statistic
    p            : float  two-sided p-value
    """
    y  = np.asarray(y_true)
    pa = np.asarray(prob_a)
    pb = np.asarray(prob_b)

    order         = np.lexsort((pa, y))[::-1]
    label_1_count = int(y.sum())

    predictions_sorted = np.vstack([pa[order], pb[order]])
    aucs, cov = _fastDeLong(predictions_sorted, label_1_count)

    auc_diff_var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if auc_diff_var <= 0:
        return float(aucs[0]), float(aucs[1]), np.nan, np.nan

    z = (aucs[0] - aucs[1]) / np.sqrt(auc_diff_var)
    p = 2 * stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


# =============================================================================
# SIGNIFICANCE TESTING
# =============================================================================

def paired_ttest_fold_aucs(results_dict, classifier='LR', alpha=0.05):
    """
    Paired t-test on 5 outer-fold AUCs: multimodal vs each unimodal.

    Returns pd.DataFrame with one row per unimodal comparison.
    """
    unimodal_keys = [k for k in results_dict
                     if k != 'Multimodal' and results_dict[k] is not None]

    multimodal_aucs = (results_dict['Multimodal']
                       .query("classifier == @classifier")
                       .sort_values('fold')['roc_auc'].values)
    rows = []
    for mod in unimodal_keys:
        uni_aucs = (results_dict[mod]
                    .query("classifier == @classifier")
                    .sort_values('fold')['roc_auc'].values)

        if len(uni_aucs) != len(multimodal_aucs):
            print(f"   ⚠️  Fold count mismatch for {mod} — skipping")
            continue

        diff  = multimodal_aucs - uni_aucs
        t, p  = stats.ttest_rel(multimodal_aucs, uni_aucs)
        ci_lo = diff.mean() - 1.96 * diff.std() / np.sqrt(len(diff))
        ci_hi = diff.mean() + 1.96 * diff.std() / np.sqrt(len(diff))

        rows.append({
            'Classifier':     classifier,
            'Comparison':     f'Multimodal vs {mod}',
            'AUC_Multimodal': f'{multimodal_aucs.mean():.3f} ± {multimodal_aucs.std():.3f}',
            'AUC_Unimodal':   f'{uni_aucs.mean():.3f} ± {uni_aucs.std():.3f}',
            'Mean_Diff':      diff.mean(),
            'CI_95_low':      ci_lo,
            'CI_95_high':     ci_hi,
            't_stat':         t,
            'p_uncorrected':  p,
        })

    df = pd.DataFrame(rows)
    if len(df):
        n_tests = len(unimodal_keys)
        df['p_bonferroni']           = (df['p_uncorrected'] * n_tests).clip(upper=1.0)
        df['significant_bonferroni'] = df['p_bonferroni'] < alpha
    return df


def run_significance_tests(results_dict, oof_probas, common_y_true, alpha=0.05):
    """
    Run paired t-tests (fold AUCs) + DeLong tests (OOF probabilities).

    Parameters
    ----------
    results_dict  : dict  {modality: fold_results_DataFrame}
    oof_probas    : dict  {'{Modality}_{CLF}': np.ndarray of OOF probas}
                    Keys e.g. 'Multimodal_LR', '145_ROIs_LR', 'SPARE_AD_RF', ...
                    All arrays aligned to common_y_true.
    common_y_true : np.ndarray  binary labels aligned to oof_probas arrays
    alpha         : float  family-wise significance level

    Returns
    -------
    dict with keys 'ttest_LR', 'ttest_RF', 'delong'
    """
    summary = {}

    print("\n" + "=" * 80)
    print("📊  SIGNIFICANCE TESTING — MULTIMODAL GAIN OVER UNIMODAL BASELINES")
    print("=" * 80)

    # ── 1. Paired t-tests on fold AUCs ───────────────────────────────────────
    for clf in ['LR', 'RF']:
        label = 'Logistic Regression' if clf == 'LR' else 'Random Forest'
        print(f"\n{'─'*60}")
        print(f"  Paired t-test (5-fold AUCs) — {label}")
        print(f"  Bonferroni α = {alpha / 3:.4f}  (3 unimodal comparisons)")
        print(f"{'─'*60}")

        ttest_df = paired_ttest_fold_aucs(results_dict, classifier=clf, alpha=alpha)
        summary[f'ttest_{clf}'] = ttest_df

        if len(ttest_df) == 0:
            print("  No results to display.")
            continue

        for _, row in ttest_df.iterrows():
            sig = '✅ significant' if row['significant_bonferroni'] else '❌ not significant'
            print(f"\n  {row['Comparison']}")
            print(f"    Multimodal AUC : {row['AUC_Multimodal']}")
            print(f"    Unimodal  AUC  : {row['AUC_Unimodal']}")
            print(f"    Mean diff      : {row['Mean_Diff']:+.4f}  "
                  f"[95% CI: {row['CI_95_low']:+.4f}, {row['CI_95_high']:+.4f}]")
            print(f"    t = {row['t_stat']:.3f},  "
                  f"p (uncorr.) = {row['p_uncorrected']:.4f},  "
                  f"p (Bonf.)   = {row['p_bonferroni']:.4f}  → {sig}")

    # ── 2. DeLong test on OOF probabilities ──────────────────────────────────
    print(f"\n{'─'*60}")
    print("  DeLong Test (full OOF predicted probabilities — Sun & Xu 2014)")
    print(f"  Bonferroni α = {alpha / 6:.4f}  (6 comparisons: 3 unimodal × 2 classifiers)")
    print(f"{'─'*60}")

    y             = np.asarray(common_y_true)
    unimodal_keys = [k for k in results_dict
                     if k != 'Multimodal' and results_dict[k] is not None]
    delong_rows   = []

    for clf in ['LR', 'RF']:
        mm_key = f'Multimodal_{clf}'
        if mm_key not in oof_probas:
            print(f"  ⚠️  Key '{mm_key}' missing from oof_probas — skipping")
            continue

        for mod in unimodal_keys:
            uni_key = f'{mod}_{clf}'
            if uni_key not in oof_probas:
                print(f"  ⚠️  Key '{uni_key}' missing from oof_probas — skipping")
                continue

            auc_mm, auc_uni, z, p = delong_roc_test(
                y, oof_probas[mm_key], oof_probas[uni_key])

            delong_rows.append({
                'Classifier':     clf,
                'Comparison':     f'Multimodal vs {mod}',
                'AUC_Multimodal': auc_mm,
                'AUC_Unimodal':   auc_uni,
                'z_stat':         z,
                'p_uncorrected':  p,
            })

    if delong_rows:
        delong_df = pd.DataFrame(delong_rows)
        n_tests   = len(delong_df)
        delong_df['p_bonferroni']           = (delong_df['p_uncorrected'] * n_tests).clip(upper=1.0)
        delong_df['significant_bonferroni'] = delong_df['p_bonferroni'] < alpha
        summary['delong'] = delong_df

        for _, row in delong_df.iterrows():
            sig       = '✅ significant' if row['significant_bonferroni'] else '❌ not significant'
            clf_label = 'Logistic Regression' if row['Classifier'] == 'LR' else 'Random Forest'
            print(f"\n  [{clf_label}]  {row['Comparison']}")
            print(f"    AUC Multimodal : {row['AUC_Multimodal']:.4f}")
            print(f"    AUC Unimodal   : {row['AUC_Unimodal']:.4f}")
            print(f"    z = {row['z_stat']:.3f},  "
                  f"p (uncorr.) = {row['p_uncorrected']:.4f},  "
                  f"p (Bonf.)   = {row['p_bonferroni']:.4f}  → {sig}")

    # ── 3. Printed summary tables ─────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  SUMMARY TABLE — Paired t-test (Bonferroni-corrected)")
    print(f"{'='*80}")
    all_ttest = pd.concat(
        [summary[k] for k in ['ttest_LR', 'ttest_RF'] if k in summary],
        ignore_index=True
    )
    if len(all_ttest):
        print(all_ttest[[
            'Classifier', 'Comparison', 'AUC_Multimodal', 'AUC_Unimodal',
            'Mean_Diff', 't_stat', 'p_uncorrected', 'p_bonferroni',
            'significant_bonferroni'
        ]].to_string(index=False))

    if 'delong' in summary:
        print(f"\n{'='*80}")
        print("  SUMMARY TABLE — DeLong test (Bonferroni-corrected)")
        print(f"{'='*80}")
        print(summary['delong'][[
            'Classifier', 'Comparison', 'AUC_Multimodal', 'AUC_Unimodal',
            'z_stat', 'p_uncorrected', 'p_bonferroni', 'significant_bonferroni'
        ]].to_string(index=False))

    print(f"\n{'='*80}\n")
    return summary


# =============================================================================
# NESTED-CV HELPER — returns fold results + OOF predicted probabilities
# =============================================================================

def run_nested_cv(X, y, classifier_name):
    """
    Nested cross-validation for a single classifier.

    Outer loop  : StratifiedKFold(5)  — 5 held-out test folds
    Inner loop  : GridSearchCV with StratifiedKFold(3) tuning on roc_auc
    Threshold   : Youden index estimated on inner OOF predictions (data-leak-free)

    Returns
    -------
    results_df : pd.DataFrame  one row per fold, all metrics + 'classifier'
    oof_proba  : np.ndarray    OOF predicted probabilities aligned to X.index
    """
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    if classifier_name == 'LR':
        base_estimator = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {
            'C':            [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty':      ['l1', 'l2'],
            'solver':       ['liblinear'],
            'class_weight': ['balanced']
        }
    elif classifier_name == 'RF':
        base_estimator = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth':    [None, 5, 10],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced']
        }
    else:
        raise ValueError(f"Unknown classifier '{classifier_name}'. Use 'LR' or 'RF'.")

    outer_results = []
    oof_proba     = np.zeros(len(y))   # accumulated across folds for DeLong

    for fold_idx, (train_idx, test_idx) in enumerate(
            outer_cv.split(X, y), 1):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Scale — fit on train only, apply to test
        scaler  = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns)
        X_test  = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns)

        # Inner hyperparameter search
        search = GridSearchCV(
            base_estimator, param_grid,
            cv=inner_cv, scoring='roc_auc',
            n_jobs=-1, refit=True
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        # Youden threshold from inner OOF predictions (data-leak-free)
        oof_inner         = cross_val_predict(
            best_model, X_train, y_train,
            cv=inner_cv, method='predict_proba', n_jobs=-1)[:, 1]
        fpr_c, tpr_c, thresholds = roc_curve(y_train, oof_inner)
        optimal_threshold         = thresholds[np.argmax(tpr_c - fpr_c)]

        # Refit on full outer train, evaluate on outer test
        best_model.fit(X_train, y_train)
        y_test_proba = best_model.predict_proba(X_test)[:, 1]
        y_test_pred  = (y_test_proba >= optimal_threshold).astype(int)

        oof_proba[test_idx] = y_test_proba   # store for DeLong

        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()

        outer_results.append({
            'classifier':  classifier_name,
            'fold':        fold_idx,
            'roc_auc':     roc_auc_score(y_test, y_test_proba),
            'pr_auc':      average_precision_score(y_test, y_test_proba),
            'precision':   precision_score(y_test, y_test_pred, zero_division=0),
            'recall':      recall_score(y_test, y_test_pred, zero_division=0),
            'f1':          f1_score(y_test, y_test_pred, zero_division=0),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'threshold':   optimal_threshold,
            'best_params': str(search.best_params_)
        })

        print(f"      Fold {fold_idx} [{classifier_name}]: "
              f"AUC={outer_results[-1]['roc_auc']:.3f}  "
              f"threshold={optimal_threshold:.3f}  "
              f"params={search.best_params_}")

    df = pd.DataFrame(outer_results)

    print(f"\n   📊 [{classifier_name}] Mean across 5 folds:")
    for metric in ['roc_auc', 'pr_auc', 'precision', 'recall', 'f1', 'specificity']:
        print(f"      • {metric.upper():12s}: "
              f"{df[metric].mean():.3f} ± {df[metric].std():.3f}")

    return df, oof_proba


def run_both_classifiers(X, y, label=''):
    """
    Run LR + RF nested CV.

    Returns
    -------
    combined_df : pd.DataFrame  all fold results for both classifiers
    oof_lr      : np.ndarray    OOF probabilities from Logistic Regression
    oof_rf      : np.ndarray    OOF probabilities from Random Forest
    """
    print(f"\n   🔵 Logistic Regression")
    lr_df, oof_lr = run_nested_cv(X, y, 'LR')
    print(f"\n   🟢 Random Forest")
    rf_df, oof_rf = run_nested_cv(X, y, 'RF')
    combined = pd.concat([lr_df, rf_df], ignore_index=True)
    return combined, oof_lr, oof_rf


# =============================================================================
# DATA LOADING
# =============================================================================

def load_145_roi_trajectories():
    print("Loading 145 ROI predicted trajectories...")
    combined_file = './manuscript1/HarmonizedMUSEROIs.csv'
    if os.path.exists(combined_file):
        roi_data = pd.read_csv(combined_file)
        print(f"   Subjects: {roi_data['id'].nunique()}")
        print(f"   Shape:    {roi_data.shape}")
        return roi_data

    roi_data = {}
    manuscript1_dir = './miccai26'
    for roi_idx in range(145):
        file_path = (f'{manuscript1_dir}/singletask_MUSE_{roi_idx}'
                     f'_dkgp_population_allstudies.csv')
        if os.path.exists(file_path):
            try:
                data = pd.read_csv(file_path)
                roi_data[f'y_H_MUSE_Volume_{roi_idx}']     = data['y']  # groundtruth
                roi_data[f'score_H_MUSE_Volume_{roi_idx}'] = data['score'] # predicted
                if roi_idx == 0:
                    roi_data['id']   = data['id']
                    roi_data['time'] = data['time']
                print(f"   Loaded ROI {roi_idx}: {data.shape[0]} rows")
            except Exception as e:
                print(f"   Error loading ROI {roi_idx}: {e}")

    roi_df = pd.DataFrame(roi_data)
    print(f"   Combined shape: {roi_df.shape}")
    return roi_df


def load_spare_ad_trajectories():
    print("Loading SPARE-AD predicted trajectories...")
    spare_ad_file = './manuscript1/singletask_SPARE_AD_dkgp_population_allstudies.csv'
    if os.path.exists(spare_ad_file):
        spare_ad_data = pd.read_csv(spare_ad_file)
        spare_ad_data = spare_ad_data.rename(columns={'score': 'SPARE_AD_score'})
        print(f"   Shape:    {spare_ad_data.shape}")
        print(f"   Subjects: {spare_ad_data['id'].nunique()}")
        return spare_ad_data
    print(f"   SPARE-AD file not found: {spare_ad_file}")
    return None


def load_adas_mmse_trajectories():
    print("Loading ADAS and MMSE predicted trajectories...")
    adas_file = './manuscript1/singletask_ADAS_dkgp_population_allstudies.csv'
    mmse_file = './manuscript1/singletask_MMSE_dkgp_population_allstudies.csv'
    adas_data, mmse_data = None, None
    if os.path.exists(adas_file):
        adas_data = pd.read_csv(adas_file)
        print(f"   ADAS shape: {adas_data.shape}  subjects: {adas_data['id'].nunique()}")
    if os.path.exists(mmse_file):
        mmse_data = pd.read_csv(mmse_file)
        print(f"   MMSE shape: {mmse_data.shape}  subjects: {mmse_data['id'].nunique()}")
    return adas_data, mmse_data


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def calculate_rate_of_change(data, biomarker_cols, time_col='time', id_col='id'):
    """OLS slope per subject per biomarker → wide-format DataFrame."""
    print("   Calculating rate of change...")
    roc_data = []
    subjects_with_roc = 0

    for subject_id in data[id_col].unique():
        subject_data = data[data[id_col] == subject_id].sort_values(time_col)
        if len(subject_data) < 2:
            continue
        subjects_with_roc += 1
        for col in biomarker_cols:
            if col not in subject_data.columns:
                continue
            t = subject_data[time_col].values.astype(float)
            v = subject_data[col].values.astype(float)
            if not np.any(np.isnan(v)):
                try:
                    slope = np.polyfit(t, v, 1)[0]
                    roc_data.append({
                        'id': subject_id, 'biomarker': col,
                        'rate_of_change': slope, 'n_timepoints': len(t)
                    })
                except Exception:
                    continue

    if not roc_data:
        print("   ❌ No rate-of-change data calculated")
        return pd.DataFrame()

    roc_df   = pd.DataFrame(roc_data)
    roc_wide = (roc_df
                .pivot(index='id', columns='biomarker', values='rate_of_change')
                .reset_index())
    roc_wide.columns.name = None

    print(f"   Subjects with RoC: {subjects_with_roc}")
    print(f"   Features:          {len(roc_wide.columns) - 1}")
    print(f"   Final shape:       {roc_wide.shape}")
    return roc_wide


# =============================================================================
# PROGRESSION LABELS
# =============================================================================

def get_progression_status(longitudinal_data):
    """
    MCI Stable     : initial Dx = 1, final Dx = 1
    MCI Progressor : initial Dx = 1, final Dx = 2
    """
    print("Identifying MCI progression status...")
    longitudinal_data = longitudinal_data.rename(columns={'PTID': 'id'})
    progression_data  = []

    for subject_id in longitudinal_data['id'].unique():
        subject_data = (longitudinal_data[longitudinal_data['id'] == subject_id]
                        .sort_values('Time'))
        if len(subject_data) < 2:
            continue
        initial_dx = subject_data.iloc[0]['Diagnosis']
        final_dx   = subject_data.iloc[-1]['Diagnosis']
        if initial_dx != 1:
            continue
        if final_dx == 1:
            status = 'MCI_Stable'
        elif final_dx == 2:
            status = 'MCI_Progressor'
        else:
            continue
        progression_data.append({
            'id':                 subject_id,
            'initial_diagnosis':  initial_dx,
            'final_diagnosis':    final_dx,
            'progression_status': status,
            'baseline_age':       subject_data.iloc[0]['Age'],
            'sex':                subject_data.iloc[0]['Sex'],
        })

    df     = pd.DataFrame(progression_data)
    mci_df = df[df['progression_status'].isin(
        ['MCI_Stable', 'MCI_Progressor'])].copy()

    stable     = (mci_df['progression_status'] == 'MCI_Stable').sum()
    progressor = (mci_df['progression_status'] == 'MCI_Progressor').sum()
    total      = len(mci_df)
    print(f"   Total MCI : {total}")
    print(f"   Stable    : {stable}  ({stable/total*100:.1f}%)")
    print(f"   Progressor: {progressor}  ({progressor/total*100:.1f}%)")
    return mci_df


# =============================================================================
# COMMON SUBJECTS
# =============================================================================

def find_common_subjects(roi_roc_data, spare_ad_roc_data,
                         cognitive_roc_data, progression_data):
    """Intersection of subjects with data in ALL modalities + progression label."""
    print("Finding subjects common across all modalities...")

    progression_subjects = set(progression_data['id'].unique())
    print(f"   MCI subjects in progression data: {len(progression_subjects)}")

    modality_subjects = {}

    if roi_roc_data is not None and len(roi_roc_data):
        modality_subjects['ROI'] = set(roi_roc_data['id'].unique())
        print(f"   ROI subjects:       {len(modality_subjects['ROI'])}")
    else:
        modality_subjects['ROI'] = set()

    if spare_ad_roc_data is not None and len(spare_ad_roc_data):
        modality_subjects['SPARE_AD'] = set(spare_ad_roc_data['id'].unique())
        print(f"   SPARE-AD subjects:  {len(modality_subjects['SPARE_AD'])}")
    else:
        modality_subjects['SPARE_AD'] = set()

    if cognitive_roc_data and len(cognitive_roc_data):
        cog_subjects = set()
        for data in cognitive_roc_data.values():
            cog_subjects.update(data['id'].unique())
        modality_subjects['Cognitive'] = cog_subjects
        print(f"   Cognitive subjects: {len(cog_subjects)}")
    else:
        modality_subjects['Cognitive'] = set()

    common = progression_subjects.copy()
    for subjects in modality_subjects.values():
        if subjects:
            common &= subjects

    print(f"\n   ✅ Common subjects (all modalities): {len(common)}")
    chosen_subjects = sorted(list(common))

    chosen_prog = progression_data[progression_data['id'].isin(chosen_subjects)]
    stable = (chosen_prog['progression_status'] == 'MCI_Stable').sum()
    prog   = (chosen_prog['progression_status'] == 'MCI_Progressor').sum()
    print(f"   MCI Stable    : {stable}  ({stable/len(chosen_subjects)*100:.1f}%)")
    print(f"   MCI Progressor: {prog}   ({prog/len(chosen_subjects)*100:.1f}%)")

    return chosen_subjects


# =============================================================================
# FEATURE PREP HELPER
# =============================================================================

def prepare_X_y(merged_data):
    """Drop ID/label columns, coerce to float, fill NaNs, return X, y."""
    feature_cols = [c for c in merged_data.columns
                    if c not in ['id', 'progression_status']]
    X = merged_data[feature_cols].copy()
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0).astype(float)
    y = (merged_data['progression_status'] == 'MCI_Progressor').astype(int)
    return X, y


# =============================================================================
# CLASSIFICATION FUNCTIONS — return results + OOF probas per classifier
# =============================================================================

def classify_progression_145_rois_fair(roi_roc_data, progression_data,
                                        common_subjects):
    print("\n" + "=" * 80)
    print("145 ROI RoC — MCI Stable vs MCI Progressor  (LR + RF)")
    print("=" * 80)

    roi_roc_common = roi_roc_data[roi_roc_data['id'].isin(common_subjects)]
    prog_common    = progression_data[progression_data['id'].isin(common_subjects)]
    merged_data    = roi_roc_common.merge(
        prog_common[['id', 'progression_status']], on='id', how='inner')

    print(f"   Subjects: {len(merged_data)}")
    X, y = prepare_X_y(merged_data)
    print(f"   Features: {X.shape[1]}   Progressors: {y.sum()}/{len(y)}")

    results, oof_lr, oof_rf = run_both_classifiers(X, y, label='145_ROIs')
    return results, merged_data, oof_lr, oof_rf


def classify_progression_spare_ad_fair(spare_ad_data, progression_data,
                                        common_subjects):
    print("\n" + "=" * 80)
    print("🔬  SPARE-AD RoC — MCI Stable vs MCI Progressor  (LR + RF)")
    print("=" * 80)

    if spare_ad_data is None:
        print("No SPARE-AD data")
        return None, None, None, None

    spare_common = spare_ad_data[spare_ad_data['id'].isin(common_subjects)]
    prog_common  = progression_data[progression_data['id'].isin(common_subjects)]

    spare_roc = calculate_rate_of_change(spare_common, ['SPARE_AD_score'], 'time', 'id')
    if len(spare_roc) == 0:
        print("Could not compute SPARE-AD RoC")
        return None, None, None, None

    merged_data = spare_roc.merge(
        prog_common[['id', 'progression_status']], on='id', how='inner')

    print(f"   Subjects: {len(merged_data)}")
    X, y = prepare_X_y(merged_data)
    print(f"   Features: {X.shape[1]}   Progressors: {y.sum()}/{len(y)}")

    results, oof_lr, oof_rf = run_both_classifiers(X, y, label='SPARE_AD')
    return results, merged_data, oof_lr, oof_rf


def classify_progression_adas_mmse_fair(adas_data, mmse_data,
                                         progression_data, common_subjects):
    print("\n" + "=" * 80)
    print("Cognitive RoC (ADAS + MMSE) — MCI Stable vs MCI Progressor  (LR + RF)")
    print("=" * 80)

    prog_common    = progression_data[progression_data['id'].isin(common_subjects)]
    cognitive_data = {}

    for name, raw in [('ADAS', adas_data), ('MMSE', mmse_data)]:
        if raw is None:
            continue
        subset = raw[raw['id'].isin(common_subjects)]
        roc    = calculate_rate_of_change(subset, ['score'], 'time', 'id')
        if len(roc):
            roc = roc.rename(columns={'score': f'{name}_score'})
            cognitive_data[name] = roc
            print(f"   {name} RoC shape: {roc.shape}")

    if not cognitive_data:
        print("No cognitive data available")
        return None, None, None, None

    combined_cognitive = None
    for name, data in cognitive_data.items():
        if combined_cognitive is None:
            combined_cognitive = data.copy()
        else:
            combined_cognitive = combined_cognitive.merge(data, on='id', how='outer')

    merged_data = combined_cognitive.merge(
        prog_common[['id', 'progression_status']], on='id', how='inner')

    print(f"   Subjects: {len(merged_data)}")
    X, y = prepare_X_y(merged_data)
    print(f"   Features: {X.shape[1]}  {list(X.columns)}  Progressors: {y.sum()}/{len(y)}")

    results, oof_lr, oof_rf = run_both_classifiers(X, y, label='Cognitive')
    return results, merged_data, oof_lr, oof_rf


def classify_multimodal_roc_fair(roi_roc_data, spare_ad_roc_data,
                                  cognitive_roc_data, progression_data,
                                  common_subjects):
    print("\n" + "=" * 80)
    print("Multimodal RoC (145 ROI + SPARE-AD + Cognitive) — LR + RF")
    print("=" * 80)

    prog_common     = progression_data[
        progression_data['id'].isin(common_subjects)][['id', 'progression_status']].copy()
    multimodal_data = prog_common.copy()

    if roi_roc_data is not None and len(roi_roc_data):
        roi_common      = roi_roc_data[roi_roc_data['id'].isin(common_subjects)]
        multimodal_data = multimodal_data.merge(roi_common, on='id', how='inner')
        print(f"   After ROI merge:      {len(multimodal_data)} subjects")

    if spare_ad_roc_data is not None and len(spare_ad_roc_data):
        spare_common    = spare_ad_roc_data[spare_ad_roc_data['id'].isin(common_subjects)]
        multimodal_data = multimodal_data.merge(spare_common, on='id', how='inner')
        print(f"   After SPARE-AD merge: {len(multimodal_data)} subjects")

    if cognitive_roc_data:
        combined_cog = None
        for name, data in cognitive_roc_data.items():
            data_renamed = data.rename(
                columns={c: f'{name}_{c}' for c in data.columns if c != 'id'})
            if combined_cog is None:
                combined_cog = data_renamed.copy()
            else:
                combined_cog = combined_cog.merge(data_renamed, on='id', how='outer')
        cog_common      = combined_cog[combined_cog['id'].isin(common_subjects)]
        multimodal_data = multimodal_data.merge(cog_common, on='id', how='inner')
        print(f"   After Cognitive merge:{len(multimodal_data)} subjects")

    print(f"\nFinal multimodal dataset: {len(multimodal_data)} subjects")

    X, y = prepare_X_y(multimodal_data)

    roi_feats  = len([c for c in X.columns if 'H_MUSE_Volume' in c])
    spare_feat = len([c for c in X.columns if 'SPARE_AD' in c])
    cog_feats  = len([c for c in X.columns if 'ADAS' in c or 'MMSE' in c])
    print(f"\n   Feature breakdown:")
    print(f"      ROI features:      {roi_feats}")
    print(f"      SPARE-AD features: {spare_feat}")
    print(f"      Cognitive features:{cog_feats}")
    print(f"      Total:             {X.shape[1]}")
    expected = 145 + 1 + 2
    if X.shape[1] != expected:
        print(f"Expected {expected} features, got {X.shape[1]}")
    else:
        print(f"Feature count correct ({expected})")
    print(f"   Progressors: {y.sum()}/{len(y)}")

    results, oof_lr, oof_rf = run_both_classifiers(X, y, label='Multimodal')
    return results, multimodal_data, oof_lr, oof_rf


# =============================================================================
# VISUALISATION — LR-only horizontal bar chart with significance brackets
# =============================================================================

def pval_label(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'


def plot_lr_significance(all_results, sig_summary, output_dir):
    """
    Horizontal bar chart — LR classifier only.
    Bars ordered bottom-to-top: 145 ROIs, SPARE-AD, Cognitive, Multimodal.
    Error bars: 95% CI = 1.96 * std / sqrt(n_folds).
    Significance brackets on the right: multimodal vs each unimodal only,
    using DeLong Bonferroni-corrected p-values.

    Parameters
    ----------
    all_results  : dict  {modality_key: fold_results_DataFrame}
    sig_summary  : dict  returned by run_significance_tests
    output_dir   : str   directory for saved figures
    """
    # ── Data — LR only, fixed display order ──────────────────────────────────
    modality_order = ['145_ROIs', 'SPARE_AD', 'Cognitive', 'Multimodal']
    display_labels = {
        '145_ROIs':  '145 ROIs',
        'SPARE_AD':  'SPARE-AD',
        'Cognitive': 'Cognitive\n(ADAS + MMSE)',
        'Multimodal': 'Multimodal\n(All)',
    }
    bar_colors = {
        '145_ROIs':  '#5B7DB1',
        'SPARE_AD':  '#6AAB7A',
        'Cognitive': '#B15B5B',
        'Multimodal': '#7B6BAF',
    }
    n_folds = 5

    means, ci95s = [], []
    for mod in modality_order:
        df_mod = all_results[mod]
        lr_aucs = (df_mod[df_mod['classifier'] == 'LR']
                   .sort_values('fold')['roc_auc'].values)
        means.append(lr_aucs.mean())
        ci95s.append(1.96 * lr_aucs.std() / np.sqrt(n_folds))

    means = np.array(means)
    ci95s = np.array(ci95s)
    y_pos = np.arange(len(modality_order))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # xlim wide enough to contain bars + value labels + brackets (no clip_on needed)
    ax.set_xlim(0.50, 1.04)

    ax.barh(y_pos, means, height=0.55,
            color=[bar_colors[m] for m in modality_order],
            alpha=0.82, zorder=2)

    ax.errorbar(means, y_pos, xerr=ci95s,
                fmt='none', color='black',
                capsize=4, capthick=1.2, elinewidth=1.2, zorder=3)

    # Value labels to the right of each error bar
    for i, (m, ci) in enumerate(zip(means, ci95s)):
        ax.text(m + ci + 0.004, i, f'{m:.3f} ± {ci:.3f}',
                va='center', ha='left', fontsize=9.5, color='#222222')

    # ── Significance brackets — fully inside xlim, no clip_on ────────────────
    delong_p = {}
    if sig_summary is not None and 'delong' in sig_summary:
        for _, row in sig_summary['delong'].iterrows():
            if row['Classifier'] == 'LR':
                mod_key = row['Comparison'].replace('Multimodal vs ', '')
                delong_p[mod_key] = row['p_bonferroni']

    mm_idx         = modality_order.index('Multimodal')
    x_anchor       = 0.930   # starts past value labels
    x_step         = 0.020   # gap between successive brackets
    unimodal_order = [m for m in modality_order if m != 'Multimodal']

    for step, mod_key in enumerate(unimodal_order):
        p_val   = delong_p.get(mod_key, np.nan)
        label   = pval_label(p_val) if not np.isnan(p_val) else '?'
        uni_idx = modality_order.index(mod_key)
        x_left  = x_anchor + step * x_step

        ax.plot([x_left, x_left], [uni_idx, mm_idx],
                color='#333333', lw=1.0, zorder=5)
        ax.plot([x_left - 0.005, x_left], [mm_idx, mm_idx],
                color='#333333', lw=1.0, zorder=5)
        ax.plot([x_left - 0.005, x_left], [uni_idx, uni_idx],
                color='#333333', lw=1.0, zorder=5)

        fs    = 11 if label != 'ns' else 9
        style = 'normal' if label != 'ns' else 'italic'
        ax.text(x_left + 0.005, (mm_idx + uni_idx) / 2, label,
                va='center', ha='left', fontsize=fs, color='#222222',
                style=style, zorder=5)

    # ── Axes & style ──────────────────────────────────────────────────────────
    ax.set_yticks(y_pos)
    ax.set_yticklabels([display_labels[m] for m in modality_order], fontsize=11)
    ax.set_xlabel('ROC-AUC (Mean ± 95% CI)', fontsize=11)
    ax.set_title('MCI Progression prediction by modality', fontsize=13, pad=12)

    # Ticks and axis spine only up to 0.90
    ax.set_xticks(np.arange(0.50, 0.91, 0.05))
    ax.spines['bottom'].set_bounds(0.50, 0.90)
    ax.xaxis.set_tick_params(labelsize=10)

    ax.xaxis.grid(True, linestyle='--', alpha=0.35, color='gray', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)

    # Footnote in figure coordinates so it is never clipped
    fig.text(0.12, 0.01,
             ('Significance vs Multimodal (DeLong test, Bonferroni-corrected):'
              '  * p<0.05   ns not significant'),
             fontsize=8, color='#555555', ha='left', va='bottom')

    # ── Save — no bbox_inches='tight' so figsize is respected exactly ─────────
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(f'{output_dir}/Multimodal_LR_Significance.pdf', dpi=300)
    plt.savefig(f'{output_dir}/Multimodal_LR_Significance.png', dpi=300)
    plt.savefig(f'{output_dir}/Multimodal_LR_Significance.svg', dpi=300)

    plt.close()
    print(f"   Figure saved to {output_dir}/Multimodal_LR_Significance.png/.pdf")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("MCI STABLE vs MCI PROGRESSOR — MULTIMODAL TRAJECTORY CLASSIFICATION")
    print("Classifiers: Logistic Regression (linear) + Random Forest (non-linear)")
    print("Significance: Paired t-test (fold AUCs) + DeLong test (OOF probas)")
    print("=" * 80)

    # ── Step 1: Load covariates ───────────────────────────────────────────────
    print(f"\n{'='*20} STEP 1: LOADING DATA {'='*20}")
    try:
        longitudinal_covariates = pd.read_csv(
            '../LongGPClustering/data1/'
            'longitudinal_covariates_subjectsamples_longclean_hmuse_convs_allstudies.csv'
        )
        print(f"   Loaded longitudinal data: {len(longitudinal_covariates):,} rows")
    except FileNotFoundError:
        print("Longitudinal covariates file not found"); return

    progression_data = get_progression_status(longitudinal_covariates)
    if len(progression_data) == 0:
        print("No MCI subjects found"); return

    # ── Step 2: Load trajectories ─────────────────────────────────────────────
    print(f"\n{'='*20} STEP 2: LOADING TRAJECTORIES {'='*20}")
    roi_data             = load_145_roi_trajectories()
    spare_ad_data        = load_spare_ad_trajectories()
    adas_data, mmse_data = load_adas_mmse_trajectories()

    # ── Step 3: Rate of change ────────────────────────────────────────────────
    print(f"\n{'='*20} STEP 3: FEATURE EXTRACTION {'='*20}")

    roi_biomarker_cols = [c for c in roi_data.columns
                          if c.startswith('score_H_MUSE_Volume_')]
    print(f"\n   ROI score columns: {len(roi_biomarker_cols)}")
    roi_roc_data = calculate_rate_of_change(
        roi_data, roi_biomarker_cols, 'time', 'id')

    spare_ad_roc_data = None
    if spare_ad_data is not None:
        print("\n   SPARE-AD RoC:")
        spare_ad_roc_data = calculate_rate_of_change(
            spare_ad_data, ['SPARE_AD_score'], 'time', 'id')

    cognitive_roc_data = {}
    if adas_data is not None:
        print("\n   ADAS RoC:")
        adas_roc = calculate_rate_of_change(adas_data, ['score'], 'time', 'id')
        if len(adas_roc):
            cognitive_roc_data['ADAS'] = adas_roc

    if mmse_data is not None:
        print("\n   MMSE RoC:")
        mmse_roc = calculate_rate_of_change(mmse_data, ['score'], 'time', 'id')
        if len(mmse_roc):
            cognitive_roc_data['MMSE'] = mmse_roc

    # ── Step 4: Common subjects ───────────────────────────────────────────────
    print(f"\n{'='*20} STEP 4: COMMON SUBJECTS {'='*20}")
    common_subjects = find_common_subjects(
        roi_roc_data, spare_ad_roc_data, cognitive_roc_data, progression_data)

    if not common_subjects:
        print("No common subjects found"); return

    np.save('common_subjects.npy', np.array(sorted(common_subjects)))
    print(f"   💾 Saved {len(common_subjects)} subjects to common_subjects.npy")

    # ── Step 5: Classification + OOF proba collection ────────────────────────
    print(f"\n{'='*20} STEP 5: CLASSIFICATION {'='*20}")

    roi_results, roi_merged, roi_oof_lr, roi_oof_rf = \
        classify_progression_145_rois_fair(
            roi_roc_data, progression_data, common_subjects)

    spare_ad_results, spare_ad_merged, spare_oof_lr, spare_oof_rf = \
        classify_progression_spare_ad_fair(
            spare_ad_data, progression_data, common_subjects)

    cognitive_results, cognitive_merged, cog_oof_lr, cog_oof_rf = \
        classify_progression_adas_mmse_fair(
            adas_data, mmse_data, progression_data, common_subjects)

    multimodal_results, multimodal_merged, mm_oof_lr, mm_oof_rf = \
        classify_multimodal_roc_fair(
            roi_roc_data, spare_ad_roc_data, cognitive_roc_data,
            progression_data, common_subjects)

    # ── Step 6: Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*20} STEP 6: FINAL SUMMARY {'='*20}")

    all_results = {
        '145_ROIs':   roi_results,
        'SPARE_AD':   spare_ad_results,
        'Cognitive':  cognitive_results,
        'Multimodal': multimodal_results,
    }

    print("\n📊 RESULTS SUMMARY (mean ± std across 5 folds):")
    print("=" * 80)

    best_overall_auc        = 0.0
    best_overall_modality   = ''
    best_overall_classifier = ''

    for modality, results_df in all_results.items():
        if results_df is None:
            continue
        print(f"\n🔍 {modality.upper()}:")
        print("-" * 60)
        for clf in ['LR', 'RF']:
            clf_df = results_df[results_df['classifier'] == clf]
            label  = 'Logistic Regression' if clf == 'LR' else 'Random Forest'
            print(f"\n   [{label}]")
            for metric in ['roc_auc', 'pr_auc', 'precision',
                           'recall', 'f1', 'specificity']:
                print(f"      • {metric:12s}: "
                      f"{clf_df[metric].mean():.3f} ± {clf_df[metric].std():.3f}")
            mean_auc = clf_df['roc_auc'].mean()
            if mean_auc > best_overall_auc:
                best_overall_auc        = mean_auc
                best_overall_modality   = modality
                best_overall_classifier = label

    print(f"\n{'='*80}")
    print(f"BEST: {best_overall_modality}  [{best_overall_classifier}]  "
          f"AUC = {best_overall_auc:.3f}")
    print("=" * 80)

    # ── Step 7: Significance testing ─────────────────────────────────────────
    print(f"\n{'='*20} STEP 7: SIGNIFICANCE TESTING {'='*20}")

    # Ground-truth labels aligned to sorted(common_subjects)
    common_y_true = (
        multimodal_merged
        .set_index('id')
        .loc[sorted(common_subjects), 'progression_status']
        .map({'MCI_Stable': 0, 'MCI_Progressor': 1})
        .values
    )

    # OOF probabilities keyed as '{Modality}_{CLF}'
    oof_probas = {}
    for key, arr in [
        ('145_ROIs_LR',   roi_oof_lr),
        ('145_ROIs_RF',   roi_oof_rf),
        ('SPARE_AD_LR',   spare_oof_lr),
        ('SPARE_AD_RF',   spare_oof_rf),
        ('Cognitive_LR',  cog_oof_lr),
        ('Cognitive_RF',  cog_oof_rf),
        ('Multimodal_LR', mm_oof_lr),
        ('Multimodal_RF', mm_oof_rf),
    ]:
        if arr is not None:
            oof_probas[key] = arr

    sig_summary = run_significance_tests(
        all_results, oof_probas=oof_probas, common_y_true=common_y_true)

    # Save significance results to CSV
    os.makedirs('./nataging/', exist_ok=True)
    if 'ttest_LR' in sig_summary and sig_summary['ttest_LR'] is not None:
        pd.concat([sig_summary.get('ttest_LR', pd.DataFrame()),
                   sig_summary.get('ttest_RF', pd.DataFrame())],
                  ignore_index=True
        ).to_csv('./nataging/significance_ttest.csv', index=False)
        print("   💾 Paired t-test results saved to ./nataging/significance_ttest.csv")

    if 'delong' in sig_summary:
        sig_summary['delong'].to_csv(
            './nataging/significance_delong.csv', index=False)
        print("   💾 DeLong test results saved to ./nataging/significance_delong.csv")

    # ── Step 8: Plot ──────────────────────────────────────────────────────────
    print(f"\n{'='*20} STEP 8: VISUALISATION {'='*20}")

    plot_lr_significance(all_results, sig_summary, output_dir='./nataging/')

    print("\n" + "=" * 80)
    print("🎉 PIPELINE COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
