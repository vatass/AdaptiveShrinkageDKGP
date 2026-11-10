#!/usr/bin/env python3
"""
Manuscript 1 - Section 2.9 / Methods 4.13
SPARE-AD predicted rate of change: discrimination of CN-to-MCI and
MCI-to-AD transitions at a fixed 5-year horizon (Figure 8).

Cohort (fixed 5-year horizon, 60 months)
----------------------------------------
  CN-to-MCI progressor : baseline CN, first diagnosis change is to MCI, at <= 60 mo
  MCI-to-AD progressor : baseline MCI, first diagnosis change is to AD, at <= 60 mo
  CN stable / MCI stable : no diagnosis change at any visit, last visit >= 60 mo
  All other subjects (baseline AD or unknown, reverters, CN->AD directly,
  conversions after 60 mo, stable with < 60 mo follow-up) are excluded.

Prediction-timing windows
-------------------------
  0-1 yr : visits with time in [0, 12) months
  2-3 yr : visits with time in [24, 36) months
  For each subject and window, the most recent visit in the window is the
  prediction time tau. Annualized rate of change at tau:
      predicted = score(tau) / (tau / 12)
      observed  = y(tau)     / (tau / 12)     (retrospective reference)
  tau = 0 (baseline only in window) gives no rate and the subject is
  dropped from that window.

  Converters whose conversion occurred before a window opens are excluded
  from that window. The 2-3 yr window is restricted to subjects with a
  valid predicted rate in the 0-1 yr window (nested, later risk set).

Which personalized prediction is used at tau  (--row_selection)
---------------------------------------------------------------
  The predictions file stores, for every personalization level History=h,
  the full trajectory predicted from the subject's first h visits, with
  Tobs = time of visit h.
    prospective (default) : use the most personalized prediction whose
                            Tobs <= tau, i.e. only observations available at
                            the time the prediction is issued.
    original              : use the highest History available at tau,
                            regardless of Tobs (as in the submitted
                            analysis; may use visits after tau).

Metrics per window x comparison x feature
-----------------------------------------
  ROC-AUC (Hanley-McNeil SE), PR-AUC / average precision (bootstrap SE,
  B = 1000, seed 42; chance level = prevalence), optimal-threshold F1 with
  its precision and recall (maximized along the PR curve).

Outputs (--outdir)
------------------
  SPAREAD_5yr_<selection>_log.txt
  SPAREAD_5yr_<selection>_rows.csv      one row per subject x window
  SPAREAD_5yr_<selection>_metrics.csv
  SPAREAD_5yr_<selection>_Fig8a_CNtoMCI.{png,pdf,svg}
  SPAREAD_5yr_<selection>_Fig8b_MCItoAD.{png,pdf,svg}
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (auc, average_precision_score,
                             precision_recall_curve, roc_curve)

HORIZON = 60.0
WINDOWS = {'0-1yr': (0.0, 12.0), '2-3yr': (24.0, 36.0)}
WINDOW_TITLES = {'0-1yr': '0-1 year since baseline', '2-3yr': '2-3 years since baseline'}
COMPARISONS = [  # label, positive, negative, figure panel
    ('CN-to-MCI', 'CN_to_MCI_5y', 'CN_stable', 'Fig8a_CNtoMCI'),
    ('MCI-to-AD', 'MCI_to_AD_5y', 'MCI_stable', 'Fig8b_MCItoAD'),
]
FEATURES = [('Predicted', 'predicted_slope'), ('Real', 'real_slope')]
N_BOOT, SEED = 1000, 42


# =============================================================================
# COHORT
# =============================================================================

def horizon_status(dx, tobs):
    """Return (status, conversion_time) for one subject."""
    base = dx[0]
    if base not in ('CN', 'MCI'):
        return 'ineligible', None
    change = next(((i, d) for i, d in enumerate(dx)
                   if d != 'UKN' and d != base), None)
    if change is None:
        if max(tobs) < HORIZON:
            return 'ineligible', None
        return f'{base}_stable', None
    i, to = change
    if tobs[i] > HORIZON:
        return 'ineligible', None
    if (base, to) == ('CN', 'MCI'):
        return 'CN_to_MCI_5y', tobs[i]
    if (base, to) == ('MCI', 'AD'):
        return 'MCI_to_AD_5y', tobs[i]
    return 'ineligible', None


def build_cohort(cov, pred):
    """Diagnoses from the covariates file are paired, in order, with the
    subject's sorted prediction time points (truncated to the shorter)."""
    times = {s: np.sort(g['time'].unique()) for s, g in pred.groupby('id')}
    rec, mismatch = [], 0
    for ptid, g in cov.groupby('PTID', sort=False):
        if ptid not in times:
            continue
        dx, t = g['Diagnosis'].tolist(), list(times[ptid])
        if len(dx) != len(t):
            mismatch += 1
            k = min(len(dx), len(t)); dx, t = dx[:k], t[:k]
        if not dx:
            continue
        st, ct = horizon_status(dx, t)
        rec.append({'PTID': ptid, 'status': st, 'conversion_time': ct})
    df = pd.DataFrame(rec)
    print(f'  diagnosis/time length mismatches (truncated): {mismatch}')
    print(df['status'].value_counts().to_string())
    return df[df['status'] != 'ineligible'].set_index('PTID')


# =============================================================================
# WINDOW ROWS
# =============================================================================

def window_rows(pred, cohort, selection):
    rows, excluded = [], {w: 0 for w in WINDOWS}
    for sid, g in pred.groupby('id'):
        if sid not in cohort.index:
            continue
        st, ct = cohort.at[sid, 'status'], cohort.at[sid, 'conversion_time']
        for w, (lo, hi) in WINDOWS.items():
            if pd.notna(ct) and ct < lo:          # already converted
                excluded[w] += 1
                continue
            win = g[(g['time'] >= lo) & (g['time'] < hi)]
            if win.empty:
                continue
            tau = win['time'].max()
            cand = win[win['time'] == tau]
            if selection == 'prospective':
                cand = cand[cand['Tobs'] <= tau]
                if cand.empty:
                    continue
            row = cand.sort_values('History', ascending=False).iloc[0]
            yrs = tau / 12.0
            rows.append({'id': sid, 'status': st, 'window': w, 'tau': tau,
                         'History': int(row['History']), 'Tobs': row['Tobs'],
                         'score': row['score'], 'y': row['y'],
                         'predicted_slope': row['score'] / yrs if tau > 0 else np.nan,
                         'real_slope':      row['y'] / yrs if tau > 0 else np.nan})
    for w, n in excluded.items():
        print(f'  {w}: {n} converters excluded (converted before window opened)')

    df = pd.DataFrame(rows)
    # Nested later risk set: 2-3 yr restricted to subjects valid at 0-1 yr
    ids01 = set(df.loc[(df['window'] == '0-1yr') & df['predicted_slope'].notna(), 'id'])
    df = df[(df['window'] == '0-1yr') | df['id'].isin(ids01)]
    return df


# =============================================================================
# METRICS
# =============================================================================

def hanley_mcneil(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    a = auc(fpr, tpr)
    n1, n2 = int(y.sum()), int(len(y) - y.sum())
    q1, q2 = a / (2 - a), 2 * a ** 2 / (1 + a)
    se = np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a ** 2)
                  + (n2 - 1) * (q2 - a ** 2)) / (n1 * n2))
    return a, se


def pr_auc_boot(y, s):
    ap = average_precision_score(y, s)
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(N_BOOT):
        i = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[i])) == 2:
            boot.append(average_precision_score(y[i], s[i]))
    return ap, np.std(boot)


def best_f1(y, s):
    p, r, _ = precision_recall_curve(y, s)
    d = p[:-1] + r[:-1]
    f1 = np.divide(2 * p[:-1] * r[:-1], d, out=np.zeros_like(d), where=d > 0)
    k = int(np.argmax(f1))
    return f1[k], p[k], r[k]


def evaluate(df, pos, neg, col):
    sub = df[df['status'].isin([pos, neg])].dropna(subset=[col])
    y = (sub['status'] == pos).astype(int).values
    s = sub[col].astype(float).values
    if len(y) < 10 or len(np.unique(y)) < 2:
        return None
    a, a_se = hanley_mcneil(y, s)
    ap, ap_se = pr_auc_boot(y, s)
    f1, p, r = best_f1(y, s)
    return {'N': len(y), 'N_pos': int(y.sum()), 'prevalence': y.mean(),
            'ROC_AUC': a, 'ROC_AUC_SE': a_se, 'PR_AUC': ap, 'PR_AUC_SE': ap_se,
            'F1': f1, 'precision': p, 'recall': r, '_y': y, '_s': s}


# =============================================================================
# FIGURE 8
# =============================================================================

def figure8(cache, comp, pos_lbl, neg_lbl, path):
    plt.rcParams.update({'font.size': 12, 'axes.linewidth': 1.5,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'legend.frameon': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    style = {'Real': ('red', '--'), 'Predicted': ('blue', '-')}
    for r, w in enumerate(WINDOWS):
        ax_roc, ax_pr = axes[r]
        res = {f: cache.get((w, comp, f)) for f, _ in FEATURES}
        ref = res['Predicted'] or res['Real']
        if ref is None:
            ax_roc.set_visible(False); ax_pr.set_visible(False); continue
        for f in ('Real', 'Predicted'):
            if res[f] is None:
                continue
            c, ls = style[f]
            fpr, tpr, _ = roc_curve(res[f]['_y'], res[f]['_s'])
            ax_roc.plot(fpr, tpr, color=c, ls=ls, lw=2.5,
                        label=f"{f} (AUC={res[f]['ROC_AUC']:.3f})")
            p, rc, _ = precision_recall_curve(res[f]['_y'], res[f]['_s'])
            ax_pr.plot(rc, p, color=c, ls=ls, lw=2.5,
                       label=f"{f} (AP={res[f]['PR_AUC']:.3f})")
        ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.4, lw=1)
        ax_pr.axhline(ref['prevalence'], color='gray', ls=':', lw=1.5,
                      label=f"Baseline (prev={ref['prevalence']:.3f})")
        title = f"{neg_lbl} vs {pos_lbl}\nn={ref['N']}"
        ax_roc.set(xlim=(0, 1), ylim=(0, 1), xlabel='False Positive Rate',
                   ylabel='True Positive Rate', title=title, aspect='equal')
        ax_pr.set(xlim=(0, 1), ylim=(0, 1), xlabel='Recall',
                  ylabel='Precision', title=title, aspect='equal')
        ax_roc.legend(fontsize=10); ax_pr.legend(fontsize=10)
        fig.text(0.5, 0.97 - 0.49 * r, WINDOW_TITLES[w],
                 ha='center', fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.96), h_pad=4)
    for ext in ('png', 'pdf', 'svg'):
        plt.savefig(f'{path}.{ext}', dpi=300, bbox_inches='tight')
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
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--predictions_csv', default='./manuscript1/'
                    'adaptive_shrinkage_predictions_alpha_simple_dkgp_SPARE_AD_allstudies.csv')
    ap.add_argument('--covariates_csv', default='longitudinal_covariates_'
                    'subjectsamples_longclean_hmuse_convs_allstudies.csv')
    ap.add_argument('--row_selection', choices=['prospective', 'original'],
                    default='prospective')
    ap.add_argument('--outdir', default='./manuscript1/spare_ad_5yr_horizon')
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    pre = os.path.join(a.outdir, f'SPAREAD_5yr_{a.row_selection}')
    sys.stdout = Tee(f'{pre}_log.txt')
    print(f'SPARE-AD 5-year horizon discrimination | row selection: {a.row_selection}')

    cov = pd.read_csv(a.covariates_csv)
    cov['Diagnosis'] = cov['Diagnosis'].replace(
        {-1.0: 'UKN', 0.0: 'CN', 1.0: 'MCI', 2.0: 'AD'})
    pred = pd.read_csv(a.predictions_csv)

    print('\n-- Horizon status --')
    cohort = build_cohort(cov, pred)

    print('\n-- Window rows --')
    rows = window_rows(pred, cohort, a.row_selection)
    rows.to_csv(f'{pre}_rows.csv', index=False)
    valid = rows.dropna(subset=['predicted_slope'])
    print(valid.groupby(['window', 'status']).size().unstack(fill_value=0).to_string())
    print('\n  Personalization level used at tau (History) by window:')
    print(valid.groupby(['window', 'History']).size().unstack(fill_value=0).to_string())
    n_future = int((valid['Tobs'] > valid['tau']).sum())
    print(f'  rows whose prediction used visits after tau: {n_future}')

    print('\n-- Discrimination --')
    out, cache = [], {}
    for w in WINDOWS:
        dfw = rows[rows['window'] == w]
        for comp, pos, neg, _ in COMPARISONS:
            for f, col in FEATURES:
                r = evaluate(dfw, pos, neg, col)
                if r is None:
                    print(f'  {w} {comp} [{f}]: insufficient data'); continue
                cache[(w, comp, f)] = r
                print(f"  {w} {comp} [{f}]: n={r['N']} (pos={r['N_pos']}, "
                      f"prev={r['prevalence']:.3f})  ROC-AUC={r['ROC_AUC']:.3f}"
                      f"±{r['ROC_AUC_SE']:.3f}  PR-AUC={r['PR_AUC']:.3f}"
                      f"±{r['PR_AUC_SE']:.3f}  F1={r['F1']:.3f} "
                      f"(P={r['precision']:.3f}, R={r['recall']:.3f})")
                out.append({'window': w, 'comparison': comp, 'feature': f,
                            **{k: v for k, v in r.items() if not k.startswith('_')}})
    pd.DataFrame(out).to_csv(f'{pre}_metrics.csv', index=False)

    figure8(cache, 'CN-to-MCI', 'CN→MCI', 'CN stable', f'{pre}_Fig8a_CNtoMCI')
    figure8(cache, 'MCI-to-AD', 'MCI→AD', 'MCI stable', f'{pre}_Fig8b_MCItoAD')
    print(f'\nOutputs -> {a.outdir}')


if __name__ == '__main__':
    main()