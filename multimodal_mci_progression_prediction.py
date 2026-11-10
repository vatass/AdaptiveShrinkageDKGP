#!/usr/bin/env python3
"""
Section 2.10 / Methods 4.14 / Supplementary F
Multimodal predicted trajectory slopes for 3-year MCI-to-AD progression
classification (Figure 9, Supplementary Table 6).

Cohort (fixed 36-month horizon)
-------------------------------
  Baseline-MCI subjects (>= 2 visits) from the longitudinal covariates.
    MCI Progressor : first diagnosis change is to AD, at <= 36 months
    MCI Stable     : no diagnosis change, last visit >= 36 months
  Excluded: conversion after 36 months, < 36 months follow-up without
  conversion, and diagnosis change to anything other than AD within 36
  months (e.g. reversion to CN).
  Analytic cohort: subjects with slopes in all three modalities.

Features
--------
  Slopes are OLS fits of p-DKGP population-model predictions (baseline scan
  and covariates only) against time, using each subject's visits at or
  before 36 months (identical, outcome-blind window for both classes):
    beta_145        : 145 regional brain volumes
    beta_SPARE-AD   : SPARE-AD
    beta_Cognitive  : ADAS-Cog 13 and MMSE
    beta_Multimodal : concatenation (148 features)
  A subject with slopes for only one of the two cognitive measures is
  retained; the missing slope is set to 0 (as in the submitted analysis).

Classification (identical for every feature set)
------------------------------------------------
  Logistic regression, nested CV: 5 outer / 3 inner stratified folds
  (seed 42), features standardized on training data, grid search over
  penalty {L1, L2} x C {0.001, 0.01, 0.1, 1, 10, 100} (liblinear, balanced
  class weights, ROC-AUC). Decision threshold per outer fold = Youden-optimal
  threshold on inner out-of-fold predictions of the training set.
  Metrics on outer test folds: ROC-AUC, PR-AUC, precision, recall, F1,
  specificity (mean +/- SD across folds; Supplementary Table 6).
  Figure 9 error bars: 1.96 * SD / sqrt(5).

Significance (multimodal vs each unimodal set)
----------------------------------------------
  Paired t-test on the five outer-fold AUCs, and DeLong test on pooled
  out-of-fold probabilities (n = 395); each Bonferroni-corrected for three
  comparisons. A comparison is significant only if both tests agree.

Outputs (--outdir)
------------------
  multimodal_3yr_log.txt, multimodal_3yr_fold_metrics.csv,
  multimodal_3yr_summary.csv (Table 6), multimodal_3yr_significance.csv,
  multimodal_3yr_oof_probabilities.csv, common_subjects_3yr.csv,
  Fig9_multimodal_3yr.{png,pdf,svg}
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

HORIZON = 36.0
SEED = 42
PARAM_GRID = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2'],
              'solver': ['liblinear'], 'class_weight': ['balanced']}
MODALITIES = ['145_ROIs', 'SPARE_AD', 'Cognitive', 'Multimodal']
DISPLAY = {'145_ROIs': '145 ROIs', 'SPARE_AD': 'SPARE-AD',
           'Cognitive': 'Cognitive\n(ADAS + MMSE)', 'Multimodal': 'Multimodal\n(All)'}
COLORS = {'145_ROIs': '#5B7DB1', 'SPARE_AD': '#6AAB7A',
          'Cognitive': '#B15B5B', 'Multimodal': '#7B6BAF'}


# =============================================================================
# COHORT
# =============================================================================

def progression_status(cov):
    """Fixed 36-month MCI progression labels."""
    rows = []
    for pid in cov['PTID'].unique():
        s = cov[cov['PTID'] == pid].sort_values('Time')
        if len(s) < 2 or s.iloc[0]['Diagnosis'] != 1:
            continue
        t, dx = s['Time'].values.astype(float), s['Diagnosis'].values
        change = next((i for i in range(1, len(dx)) if dx[i] != dx[0]), None)
        if change is None:
            if t.max() < HORIZON:
                continue
            status = 'MCI_Stable'
        else:
            if t[change] > HORIZON or dx[change] != 2:
                continue
            status = 'MCI_Progressor'
        rows.append({'id': pid, 'status': status})
    return pd.DataFrame(rows)


# =============================================================================
# FEATURES
# =============================================================================

def slopes(data, cols, eligible):
    """OLS slope per subject and column over visits at or before HORIZON.
    Returns wide DataFrame indexed by id (eligible subjects only)."""
    out = {}
    for sid, g in data[data['id'].isin(eligible)].groupby('id'):
        g = g.sort_values('time')
        g = g[g['time'] <= HORIZON]
        if len(g) < 2:
            continue
        t = g['time'].values.astype(float)
        row = {}
        for c in cols:
            v = g[c].values.astype(float)
            if not np.isnan(v).any():
                row[c] = np.polyfit(t, v, 1)[0]
        if row:
            out[sid] = row
    return pd.DataFrame.from_dict(out, orient='index').sort_index()


def build_features(args, labels):
    elig = set(labels['id'])
    roi = pd.read_csv(args.roi_csv)
    # string order, matching the column order of the submitted analysis
    roi_cols = sorted(c for c in roi.columns if c.startswith('score_H_MUSE_Volume_'))
    assert len(roi_cols) == 145
    f_roi = slopes(roi, roi_cols, elig)

    spare = pd.read_csv(args.spare_csv).rename(columns={'score': 'SPARE_AD_score'})
    f_spare = slopes(spare, ['SPARE_AD_score'], elig)

    cog = []
    for name, path in (('ADAS', args.adas_csv), ('MMSE', args.mmse_csv)):
        s = slopes(pd.read_csv(path), ['score'], elig)
        cog.append(s.rename(columns={'score': f'{name}_score'}))
    f_cog = cog[0].join(cog[1], how='outer')          # either measure suffices

    common = sorted(elig & set(f_roi.index) & set(f_spare.index) & set(f_cog.index))
    f_cog = f_cog.loc[common]
    print(f'  Missing ADAS-Cog 13 slope (set to 0): {f_cog["ADAS_score"].isna().sum()}')
    print(f'  Missing MMSE slope (set to 0)       : {f_cog["MMSE_score"].isna().sum()}')

    feats = {'145_ROIs': f_roi.loc[common], 'SPARE_AD': f_spare.loc[common],
             'Cognitive': f_cog}
    feats['Multimodal'] = pd.concat([feats['145_ROIs'], feats['SPARE_AD'], f_cog], axis=1)
    feats = {k: v.fillna(0.0).astype(float) for k, v in feats.items()}
    assert feats['Multimodal'].shape[1] == 148
    y = (labels.set_index('id').loc[common, 'status'] == 'MCI_Progressor').astype(int)
    return feats, y, common


# =============================================================================
# NESTED CROSS-VALIDATION
# =============================================================================

def nested_cv(X, y):
    outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
    inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
    oof, folds = np.zeros(len(y)), []
    for k, (tr, te) in enumerate(outer.split(X, y), 1):
        sc = StandardScaler().fit(X.iloc[tr])
        Xtr, Xte = sc.transform(X.iloc[tr]), sc.transform(X.iloc[te])
        ytr, yte = y.iloc[tr].values, y.iloc[te].values
        gs = GridSearchCV(LogisticRegression(random_state=SEED, max_iter=1000),
                          PARAM_GRID, cv=inner, scoring='roc_auc',
                          n_jobs=-1, refit=True).fit(Xtr, ytr)
        model = gs.best_estimator_
        p_in = cross_val_predict(model, Xtr, ytr, cv=inner,
                                 method='predict_proba', n_jobs=-1)[:, 1]
        fpr, tpr, thr = roc_curve(ytr, p_in)
        threshold = thr[np.argmax(tpr - fpr)]                # Youden J
        model.fit(Xtr, ytr)
        prob = model.predict_proba(Xte)[:, 1]
        pred = (prob >= threshold).astype(int)
        oof[te] = prob
        tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
        folds.append({'fold': k, 'roc_auc': roc_auc_score(yte, prob),
                      'pr_auc': average_precision_score(yte, prob),
                      'precision': precision_score(yte, pred, zero_division=0),
                      'recall': recall_score(yte, pred, zero_division=0),
                      'f1': f1_score(yte, pred, zero_division=0),
                      'specificity': tn / (tn + fp) if (tn + fp) else 0.0,
                      'threshold': threshold, 'best_params': str(gs.best_params_)})
    return pd.DataFrame(folds), oof


# =============================================================================
# SIGNIFICANCE
# =============================================================================

def _midrank(x):
    j = np.argsort(x); z = x[j]; n = len(x)
    t = np.zeros(n); i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1)
        i = k
    out = np.empty(n); out[j] = t + 1
    return out


def delong(y, pa, pb):
    """Paired DeLong test (Sun & Xu 2014). Returns AUC_a, AUC_b, z, p."""
    y = np.asarray(y)
    order = np.argsort(-y, kind='mergesort')
    preds = np.vstack([pa, pb])[:, order]
    m = int(y.sum()); n = preds.shape[1] - m
    tx = np.array([_midrank(r[:m]) for r in preds])
    ty = np.array([_midrank(r[m:]) for r in preds])
    tz = np.array([_midrank(r) for r in preds])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs[0], aucs[1], z, 2 * stats.norm.sf(abs(z))


def significance(folds, oof, y, alpha=0.05):
    mm = folds['Multimodal']['roc_auc'].values
    rows, n_tests = [], 3
    for mod in MODALITIES[:-1]:
        uni = folds[mod]['roc_auc'].values
        d = mm - uni
        t, p_t = stats.ttest_rel(mm, uni)
        half = 1.96 * d.std() / np.sqrt(len(d))
        a_mm, a_uni, z, p_d = delong(y, oof['Multimodal'], oof[mod])
        r = {'comparison': f'Multimodal vs {mod}', 'mean_auc_diff': d.mean(),
             'diff_ci95_low': d.mean() - half, 'diff_ci95_high': d.mean() + half,
             't': t, 'p_t': p_t, 'p_t_bonf': min(p_t * n_tests, 1.0),
             'pooled_auc_multimodal': a_mm, 'pooled_auc_unimodal': a_uni,
             'z': z, 'p_delong': p_d, 'p_delong_bonf': min(p_d * n_tests, 1.0)}
        r['significant_both'] = r['p_t_bonf'] < alpha and r['p_delong_bonf'] < alpha
        rows.append(r)
    return pd.DataFrame(rows)


# =============================================================================
# FIGURE 9
# =============================================================================

def figure9(folds, path):
    means = np.array([folds[m]['roc_auc'].mean() for m in MODALITIES])
    ci = np.array([1.96 * folds[m]['roc_auc'].values.std() / np.sqrt(5)
                   for m in MODALITIES])
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(MODALITIES))
    ax.barh(y, means, height=0.55, color=[COLORS[m] for m in MODALITIES],
            alpha=0.82, zorder=2)
    ax.errorbar(means, y, xerr=ci, fmt='none', color='black', capsize=4,
                capthick=1.2, elinewidth=1.2, zorder=3)
    for i, (m, c) in enumerate(zip(means, ci)):
        ax.text(m + c + 0.004, i, f'{m:.3f} ± {c:.3f}', va='center',
                fontsize=9.5, color='#222222')
    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY[m] for m in MODALITIES], fontsize=11)
    ax.set_xlim(0.50, 1.0)
    ax.set_xticks(np.arange(0.50, 0.91, 0.05))
    ax.spines['bottom'].set_bounds(0.50, 0.90)
    ax.set_xlabel('ROC-AUC (Mean ± 95% CI)', fontsize=11)
    ax.set_title('Multimodal trajectory slopes for 3-year MCI-to-AD '
                 'progression classification.', fontsize=12, pad=12)
    ax.xaxis.grid(True, linestyle='--', alpha=0.35, color='gray', zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0)
    plt.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        plt.savefig(f'{path}.{ext}', dpi=300)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

class Tee:
    def __init__(self, path):
        self.f, self.s = open(path, 'w'), sys.__stdout__
    def write(self, t):
        self.s.write(t); self.f.write(t)
    def flush(self):
        self.s.flush(); self.f.flush()


def main():
    ap = argparse.ArgumentParser(description='Section 2.10 multimodal 3-year classification')
    ap.add_argument('--covariates_csv', default='longitudinal_covariates_'
                    'subjectsamples_longclean_hmuse_convs_allstudies.csv')
    ap.add_argument('--roi_csv', default='./manuscript1/HarmonizedMUSEROIs.csv')
    ap.add_argument('--spare_csv', default='./manuscript1/singletask_SPARE_AD_dkgp_population_allstudies.csv')
    ap.add_argument('--adas_csv', default='./manuscript1/singletask_ADAS_dkgp_population_allstudies.csv')
    ap.add_argument('--mmse_csv', default='./manuscript1/singletask_MMSE_dkgp_population_allstudies.csv')
    ap.add_argument('--outdir', default='./manuscript1/multimodal_3yr')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    out = lambda n: os.path.join(a.outdir, n)
    sys.stdout = Tee(out('multimodal_3yr_log.txt'))

    print('Section 2.10 - multimodal predicted slopes, 3-year MCI-to-AD horizon')
    cov = pd.read_csv(a.covariates_csv)
    labels = progression_status(cov)
    order_sorted = list(labels['id']) == sorted(labels['id'])
    print(f'  Fixed-horizon eligible MCI: {len(labels)} '
          f'({(labels.status == "MCI_Progressor").sum()} progressors)')
    print(f'  Covariate subject order already sorted by PTID: {order_sorted}')

    feats, y, common = build_features(a, labels)
    pd.Series(common, name='id').to_csv(out('common_subjects_3yr.csv'), index=False)
    print(f'  Analytic cohort: n = {len(y)}  ({int((y == 0).sum())} stable, '
          f'{int(y.sum())} progressors, {100 * y.mean():.1f}%)')

    folds, oof, summary = {}, {}, []
    for mod in MODALITIES:
        print(f'\n-- {mod} ({feats[mod].shape[1]} features) --')
        folds[mod], oof[mod] = nested_cv(feats[mod], y)
        f = folds[mod]
        for _, r in f.iterrows():
            print(f"   fold {r['fold']}: AUC={r['roc_auc']:.3f}  {r['best_params']}")
        row = {'feature_set': mod}
        for k in ('roc_auc', 'pr_auc', 'precision', 'recall', 'f1', 'specificity'):
            row[k], row[k + '_sd'] = f[k].mean(), f[k].std()
            print(f'   {k:<12}: {f[k].mean():.3f} ± {f[k].std():.3f}')
        row['roc_auc_ci95'] = 1.96 * f['roc_auc'].values.std() / np.sqrt(5)
        summary.append(row)

    pd.concat([f.assign(feature_set=m) for m, f in folds.items()]).to_csv(
        out('multimodal_3yr_fold_metrics.csv'), index=False)
    pd.DataFrame(summary).to_csv(out('multimodal_3yr_summary.csv'), index=False)
    pd.DataFrame({'id': common, 'y': y.values, **oof}).to_csv(
        out('multimodal_3yr_oof_probabilities.csv'), index=False)

    print('\n-- Significance: Multimodal vs unimodal (Bonferroni x3; both tests required) --')
    sig = significance(folds, oof, y.values)
    for _, r in sig.iterrows():
        print(f"  {r['comparison']}: diff={r['mean_auc_diff']:+.3f} "
              f"[{r['diff_ci95_low']:+.3f}, {r['diff_ci95_high']:+.3f}]  "
              f"t={r['t']:.3f} p_bonf={r['p_t_bonf']:.3f} | "
              f"DeLong z={r['z']:.3f} p_bonf={r['p_delong_bonf']:.3f}  "
              f"-> {'significant' if r['significant_both'] else 'not significant'}")
    sig.to_csv(out('multimodal_3yr_significance.csv'), index=False)

    figure9(folds, out('Fig9_multimodal_3yr'))
    print(f'\nOutputs -> {a.outdir}')


if __name__ == '__main__':
    main()