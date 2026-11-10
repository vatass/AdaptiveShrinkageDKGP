#!/usr/bin/env python3
"""
Section 2.5 (MCI-to-AD progression classification, Fig. 4b) /
Methods 4.7

Labels (fixed ~3-year horizon, 36 +/- 6 months), baseline-MCI subjects with
>= 2 diagnostic visits:
  Progressor     : first AD diagnosis at <= 42 months
  Non-progressor : no AD diagnosis by 42 months and diagnostic follow-up
                   >= 30 months
  Excluded       : non-converters with < 30 months of follow-up

Feature sets (145 ROIs each), matched cohort = subjects with all four:
  Baseline volumes : ROI volumes at the first visit
  p-DKGP predicted : (score(t_e) - score(0)) / t_e, where t_e is the
  RNN-AD predicted   prediction time closest to 36 months within [30, 42]
                     (baseline-only forecasts; no observed follow-up used)
  Observed (oracle): the same endpoint difference on observed volumes
                     (y columns) - not deployable, reference only

Classifier (identical for every feature set): L2 logistic regression,
class-balanced, standardization inside the pipeline; nested CV with 5 outer /
3 inner stratified folds (seed 42); C in {0.01, 0.1, 1, 10} chosen by inner
ROC-AUC. Thresholded metrics use the default 0.5 probability.

Reported:
  - Fold-mean ROC-AUC and PR-AUC (+/- SD across folds), fold-mean precision,
    recall, F1
  - Paired tests on pooled out-of-fold probabilities: DeLong (ROC-AUC);
    paired subject bootstrap (B = 5000) for PR-AUC, precision, recall, FPR,
    F1; paired permutation test for PR-AUC; exact McNemar tests for recall
    (progressors) and FPR (non-progressors)
  - Figure 4b ROC curves (mean across folds)
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

TARGET, TOL = 36.0, 6.0
SEED = 42
EPS = 1e-6
MCI, AD = 1, 2
N_BOOT = N_PERM = 5000
FEATURES = ['Baseline volumes', 'RNN-AD predicted', 'p-DKGP predicted', 'Observed (oracle)']
COMPARISONS = [('p-DKGP predicted', 'Baseline volumes'),
               ('p-DKGP predicted', 'RNN-AD predicted'),
               ('p-DKGP predicted', 'Observed (oracle)'),
               ('RNN-AD predicted', 'Baseline volumes'),
               ('Observed (oracle)', 'Baseline volumes')]


# =============================================================================
# LABELS AND FEATURES
# =============================================================================

def labels(cov):
    rows = []
    for pid, g in cov.groupby('PTID'):
        g = g.sort_values('Time')
        if len(g) < 2 or not pd.notna(g.iloc[0]['Diagnosis']) or int(g.iloc[0]['Diagnosis']) != MCI:
            continue
        rel = g['Time'].astype(float) - float(g.iloc[0]['Time'])
        ad = rel[g['Diagnosis'] == AD]
        first_ad = float(ad.iloc[0]) if len(ad) else np.nan
        if np.isfinite(first_ad) and first_ad <= TARGET + TOL + EPS:
            rows.append((pid, 1))
        elif rel.max() >= TARGET - TOL - EPS:
            rows.append((pid, 0))
    return pd.Series(dict(rows), name='is_progressor')


def roi_cols(df, prefix):
    return sorted([c for c in df.columns if c.startswith(prefix)],
                  key=lambda c: int(c.split('_')[-1]))


def endpoint_slopes(path, prefix):
    """(value at endpoint - value at baseline) / endpoint time, where the
    endpoint is the time closest to 36 months within [30, 42]."""
    df = pd.read_csv(path)
    cols = roi_cols(df, prefix)
    assert len(cols) == 145, f'{path}: {len(cols)} {prefix} columns'
    out = {}
    for sid, g in df.groupby('id', sort=False):
        g = g.sort_values('time').copy()
        g['rel'] = g['time'].astype(float) - float(g['time'].min())
        base = g[np.isclose(g['rel'], 0.0, atol=EPS)]
        win = g[(g['rel'] >= TARGET - TOL - EPS) & (g['rel'] <= TARGET + TOL + EPS)].copy()
        if base.empty or win.empty:
            continue
        win['dist'] = (win['rel'] - TARGET).abs()
        end = win.sort_values(['dist', 'rel']).iloc[0]
        te = float(end['rel'])
        b = base.iloc[0]
        out[sid] = [(float(end[c]) - float(b[c])) / te if te > 0 else np.nan for c in cols]
    return pd.DataFrame.from_dict(out, orient='index',
                                  columns=[int(c.split('_')[-1]) for c in cols])


def baseline_volumes(path):
    df = pd.read_csv(path)
    cols = roi_cols(df, 'y_H_MUSE_Volume_')
    assert len(cols) == 145
    return df.sort_values('time').groupby('id', sort=False).head(1).set_index('id')[cols]


# =============================================================================
# CLASSIFICATION
# =============================================================================

def nested_cv(X, y):
    outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
    inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('clf', LogisticRegression(max_iter=5000, class_weight='balanced',
                                                random_state=SEED))])
    grid = np.linspace(0, 1, 300)
    folds, tprs, yt, yp = [], [], [], []
    for tr, te in outer.split(X, y):
        gs = GridSearchCV(pipe, {'clf__C': [0.01, 0.1, 1.0, 10.0]}, cv=inner,
                          scoring='roc_auc', n_jobs=-1, refit=True).fit(X[tr], y[tr])
        prob = gs.best_estimator_.predict_proba(X[te])[:, 1]
        pred = gs.best_estimator_.predict(X[te])
        folds.append({'roc_auc': roc_auc_score(y[te], prob),
                      'pr_auc': average_precision_score(y[te], prob),
                      'precision': precision_score(y[te], pred, zero_division=0),
                      'recall': recall_score(y[te], pred, zero_division=0),
                      'f1': f1_score(y[te], pred, zero_division=0),
                      'C': gs.best_params_['clf__C']})
        fpr, tpr, _ = roc_curve(y[te], prob)
        t = np.interp(grid, fpr, tpr); t[0], t[-1] = 0.0, 1.0
        tprs.append(t)
        yt += list(y[te]); yp += list(prob)
    mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
    return pd.DataFrame(folds), np.array(yt), np.array(yp), grid, mean_tpr


# =============================================================================
# PAIRED TESTS (pooled out-of-fold probabilities)
# =============================================================================

def _midrank(x):
    o = np.argsort(x); s = x[o]; n = len(x); m = np.zeros(n); i = 0
    while i < n:
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        m[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n); out[o] = m
    return out


def delong(y, a, b):
    o = np.argsort(-y); m = int((y == 1).sum()); P = np.vstack([a, b])[:, o]; n = P.shape[1] - m
    tx = np.array([_midrank(r[:m]) for r in P]); ty = np.array([_midrank(r[m:]) for r in P])
    tz = np.array([_midrank(r) for r in P])
    auc = tz[:, :m].sum(1) / (m * n) - (m + 1.0) / (2.0 * n)
    cov = np.cov((tz[:, :m] - tx) / n) / m + np.cov(1.0 - (tz[:, m:] - ty) / m) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (auc[0] - auc[1]) / np.sqrt(var)
    return auc[0], auc[1], z, 2 * stats.norm.sf(abs(z))


def binary(y, s, thr=0.5):
    p = s >= thr
    tp, fn = np.sum(y & p), np.sum(y & ~p)
    fp, tn = np.sum(~y & p), np.sum(~y & ~p)
    rec = tp / (tp + fn) if tp + fn else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * prec * rec / (prec + rec) if np.isfinite(prec) and np.isfinite(rec) and prec + rec else np.nan
    return {'precision': prec, 'recall': rec, 'f1': f1,
            'fpr': fp / (fp + tn) if fp + tn else np.nan}


def metric(y, s, name):
    if name == 'roc_auc':
        return roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan
    if name == 'pr_auc':
        return average_precision_score(y, s) if y.sum() else np.nan
    return binary(y.astype(bool), s)[name]


def bootstrap(y, a, b, name):
    rng = np.random.default_rng(SEED)
    d0 = metric(y, a, name) - metric(y, b, name)
    d = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        va, vb = metric(y[i], a[i], name), metric(y[i], b[i], name)
        if np.isfinite(va) and np.isfinite(vb):
            d.append(va - vb)
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))
    return d0, lo, hi, p


def permutation_pr(y, a, b):
    rng = np.random.default_rng(SEED)
    d0 = average_precision_score(y, a) - average_precision_score(y, b)
    null = []
    for _ in range(N_PERM):
        sw = rng.random(len(y)) < 0.5
        pa, pb = a.copy(), b.copy()
        pa[sw], pb[sw] = b[sw], a[sw]
        null.append(average_precision_score(y, pa) - average_precision_score(y, pb))
    return d0, (np.sum(np.abs(null) >= abs(d0)) + 1) / (N_PERM + 1)


def mcnemar(ea, eb):
    ao, bo = int(np.sum(ea & ~eb)), int(np.sum(~ea & eb))
    p = 1.0 if ao + bo == 0 else stats.binomtest(min(ao, bo), ao + bo, 0.5).pvalue
    return ao, bo, p


def paired_tests(res):
    rows = []
    for a, b in COMPARISONS:
        y, sa, sb = res[a]['y'], res[a]['p'], res[b]['p']
        assert np.array_equal(y, res[b]['y'])
        A, B, z, p = delong(y, sa, sb)
        rows.append({'comparison': f'{a} vs {b}', 'metric': 'roc_auc', 'test': 'DeLong',
                     'a': A, 'b': B, 'delta': A - B, 'stat': z, 'p': p})
        for m in ('pr_auc', 'precision', 'recall', 'fpr', 'f1'):
            d, lo, hi, p = bootstrap(y, sa, sb, m)
            rows.append({'comparison': f'{a} vs {b}', 'metric': m, 'test': 'paired bootstrap',
                         'a': metric(y, sa, m), 'b': metric(y, sb, m), 'delta': d,
                         'ci_low': lo, 'ci_high': hi, 'p': p})
        d, p = permutation_pr(y, sa, sb)
        rows.append({'comparison': f'{a} vs {b}', 'metric': 'pr_auc',
                     'test': 'paired permutation', 'delta': d, 'p': p})
        yb = y.astype(bool)
        for m, mask in (('recall', yb), ('fpr', ~yb)):
            ao, bo, p = mcnemar((sa >= 0.5)[mask], (sb >= 0.5)[mask])
            rows.append({'comparison': f'{a} vs {b}', 'metric': m, 'test': 'exact McNemar',
                         'a_only': ao, 'b_only': bo, 'p': p})
    return pd.DataFrame(rows)


# =============================================================================
# FIGURE 4b
# =============================================================================

STYLE = {'Baseline volumes': ('#9B59B6', (5, 2), 1.8, 'Baseline volumes'),
         'RNN-AD predicted': ('#E74C3C', (4, 2, 1, 2), 2.0, 'RNN-AD predicted'),
         'p-DKGP predicted': ('#1A5276', None, 2.6, 'p-DKGP predicted'),
         'Observed (oracle)': ('#7F8C8D', (2, 2), 1.9, 'Real measurements')}


def figure4b(res, path):
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    for f in FEATURES:
        c, dash, lw, lab = STYLE[f]
        line, = ax.plot(res[f]['fpr'], res[f]['tpr'], color=c, lw=lw, label=lab)
        if dash:
            line.set_dashes(dash)
    ax.plot([0, 1], [0, 1], '--', color='#AAAAAA', lw=1.2, alpha=0.7)
    ax.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.01))
    ax.set_xlabel('False Positive Rate', fontsize=16)
    ax.set_ylabel('True Positive Rate', fontsize=16)
    ax.set_title('3-year MCI-to-AD progression classification', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc='lower right', fontsize=13, frameon=False)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        plt.savefig(f'{path}.{ext}', dpi=600 if ext == 'png' else None, bbox_inches='tight')
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
    ap = argparse.ArgumentParser(description='Section 2.5 MCI-to-AD progression classification')
    ap.add_argument('--covariates_csv', default='longitudinal_covariates_'
                    'subjectsamples_longclean_hmuse_convs_allstudies.csv')
    ap.add_argument('--pdkgp_csv', default='./manuscript1/OldHarmonizedMUSEROIs.csv')
    ap.add_argument('--rnn_csv', default='../Standalone_Nguyen2020_RNNAD/manuscript1/RNN_AD_Consolidated.csv')
    ap.add_argument('--baseline_csv', default='./manuscript1/HarmonizedROIVolumes.csv')
    ap.add_argument('--outdir', default='./manuscript1/mci_progression_3yr')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    out = lambda n: os.path.join(a.outdir, n)
    sys.stdout = Tee(out('mci_progression_3yr_log.txt'))
    pd.set_option('display.width', 220)

    lab = labels(pd.read_csv(a.covariates_csv))
    print(f'Eligible baseline-MCI: {len(lab)} ({int(lab.sum())} progressors)')
    feats = {'Baseline volumes': baseline_volumes(a.baseline_csv),
             'RNN-AD predicted': endpoint_slopes(a.rnn_csv, 'score_H_MUSE_Volume_'),
             'p-DKGP predicted': endpoint_slopes(a.pdkgp_csv, 'score_H_MUSE_Volume_'),
             'Observed (oracle)': endpoint_slopes(a.pdkgp_csv, 'y_H_MUSE_Volume_')}
    for k, v in feats.items():
        print(f'  {k:<18}: {len(set(v.index) & set(lab.index))} labelled subjects')
    ids = sorted(set(lab.index).intersection(*[set(v.index) for v in feats.values()]))
    y = lab.loc[ids].values.astype(int)
    print(f'Matched cohort: n = {len(y)} ({y.sum()} progressors, {(y == 0).sum()} non-progressors)')
    pd.DataFrame({'PTID': ids, 'is_progressor': y}).to_csv(out('cohort.csv'), index=False)

    res, summary = {}, []
    for f in FEATURES:
        X = feats[f].loc[ids].dropna(axis=1, how='all')
        if X.isna().any().any():
            print(f'  {f}: {int(X.isna().sum().sum())} missing values imputed with column medians')
            X = X.fillna(X.median())
        folds, yt, yp, fpr, tpr = nested_cv(X.values.astype(float), y)
        res[f] = {'y': yt, 'p': yp, 'fpr': fpr, 'tpr': tpr}
        s = {'feature_set': f, 'roc_auc': folds.roc_auc.mean(), 'roc_auc_sd': folds.roc_auc.std(ddof=0),
             'pr_auc': folds.pr_auc.mean(), 'pr_auc_sd': folds.pr_auc.std(ddof=0),
             'precision': folds.precision.mean(), 'recall': folds.recall.mean(),
             'f1': folds.f1.mean(), 'C_per_fold': list(folds.C)}
        pooled = binary(yt.astype(bool), yp)
        s.update({f'pooled_{k}': v for k, v in pooled.items()})
        summary.append(s)
    sm = pd.DataFrame(summary)
    print('\n-- Classification (logistic regression; fold mean, +/- SD across folds) --')
    print(sm.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    sm.to_csv(out('summary.csv'), index=False)

    pt = paired_tests(res)
    print('\n-- Paired tests (pooled out-of-fold predictions) --')
    print(pt.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    pt.to_csv(out('paired_tests.csv'), index=False)

    figure4b(res, out('Fig4b_mci_progression_roc'))
    print(f'\nOutputs -> {a.outdir}')


if __name__ == '__main__':
    main()