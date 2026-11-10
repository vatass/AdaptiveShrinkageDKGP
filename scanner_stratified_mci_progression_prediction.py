#!/usr/bin/env python3
"""
Section 2.6 / Methods 4.9
Predicted vs observed rates of change for MCI progression classification
under scanner-change conditions.

What this script reproduces
---------------------------
Results 2.6, Figure 5, Methods 4.9:

  1. Pooled MCI cohort (n = 882; 296 progressors) split into
       - scanner-stable : one MRI scanner model across all visits   (549)
       - scanner-change : >= 1 transition between scanner models    (331)
     Subjects with no scanner model recorded at any visit (2) belong to
     neither subgroup; they are used for training only.

  2. Two feature sets, 145 regional slopes each:
       - beta_pDKGP : OLS slope of p-DKGP predictions (baseline scan only)
       - beta_obs   : OLS slope of observed volumes over full follow-up

  3. L2-regularized, class-balanced logistic regression, nested CV
     (5 outer / 3 inner stratified folds, C in {0.01, 0.1, 1, 10}, ROC-AUC
     selection), features standardized within each fold on training data.
     ONE model per fold is trained on the full 882-subject cohort; metrics
     are computed separately within each held-out test fold for the
     scanner subgroups (training distribution held fixed).

  4. 1:1 nearest-neighbour propensity score matching (age, sex, total
     brain volume, progression label; caliper 0.2 SD; without
     replacement; change subjects ordered by propensity score)
     -> 323 matched pairs (n = 646). Subgroup metrics are recomputed on the
     matched subset, in addition to the unmatched cohort.

  5. DeLong test, beta_pDKGP vs beta_obs, on pooled out-of-fold predicted
     probabilities from the full unmatched 882-subject cohort (p = 0.80).

  6. Figure 5b: ROC curves for the matched scanner-stable and matched
     scanner-change subgroups, with the operating point (default 0.5
     probability threshold of the class-balanced classifier) marked.
     Figure 5a: hippocampus / ventricle trajectories for matched
     scanner-change progressors missed by beta_obs but caught by
     beta_pDKGP.

Outputs (OUT_DIR)
-----------------
  scanner_subgroup_results.txt     full log
  subgroup_metrics.csv             all subgroup x feature metrics
  subgroup_roc_panels_matched.*    Figure 5b
  subgroup_roc_panels_unmatched.*  supplementary/sanity
  scanner_contamination_candidates.csv, ScannerContamination_*.*  Fig 5a
  scanner_subgroup_cache.pkl
"""

import os
import sys
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================

OUT_DIR    = './manuscript1/scanner_stratified'
COVARIATES = 'longitudinal_covariates_subjectsamples_longclean_hmuse_convs_allstudies.csv'
DKGP_IN    = Path('./manuscript1/OldHarmonizedMUSEROIs.csv')
# The RNN-AD file supplies the observed volumes used for beta_obs, and its
# subject list is part of the common-cohort intersection that defines n=882.
RNN_IN     = Path('../Standalone_Nguyen2020_RNNAD/manuscript1/RNN_AD_Consolidated.csv')
BASELINE_IN = Path('./manuscript1/HarmonizedROIVolumes.csv')

SEED          = 42
OUTER_FOLDS   = 5
INNER_FOLDS   = 3
C_GRID        = [0.01, 0.1, 1.0, 10.0]
CALIPER_SD    = 0.2
HIPPOCAMPUS   = 48     # MUSE ROI index
VENTRICLES    = 52     # MUSE ROI index
N_EXAMPLES    = 5      # Fig. 5a candidates to plot

# As run for the manuscript, subjects with no scanner model recorded are
# encoded as is_change = 0 and are therefore eligible as scanner-stable
# controls during matching. Set True to restrict matching to subjects with
# a known scanner group (may change the 323 matched pairs).
RESTRICT_MATCHING_TO_KNOWN_SCANNER = False

FEATURES = {'p-DKGP': 'Predicted', 'Observed': 'Observed'}   # key -> fig label


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_scanner_groups(cov):
    """Scanner-stable: one scanner model across all visits with a recorded
    model. Scanner-change: >= 1 distinct model. Subjects with no recorded
    model at any visit are returned in neither set."""
    stable, change = set(), set()
    for ptid, grp in cov.groupby('PTID'):
        models = grp['MRI_Scanner_Model'].dropna().unique()
        if len(models) == 0:
            continue
        (change if len(models) > 1 else stable).add(ptid)
    return stable, change


def ols_slope(t, v):
    return np.polyfit(t, v, 1)[0] if len(t) >= 2 else np.nan


def paired_slopes(df):
    """Per-subject OLS slopes against time (months) for each of the 145 ROIs,
    over all visits where both the observed (y_) and predicted (score_)
    value exist. Returns (observed_wide, predicted_wide), indexed by PTID."""
    key = lambda c: int(c.split('_')[-1])
    y_cols = sorted([c for c in df.columns if c.startswith('y_H_MUSE_Volume_')], key=key)
    s_cols = sorted([c for c in df.columns if c.startswith('score_H_MUSE_Volume_')], key=key)
    assert len(y_cols) == len(s_cols) == 145
    obs, pred = {}, {}
    for yc, sc in zip(y_cols, s_cols):
        roi = key(yc)
        sub = df[['id', 'time', yc, sc]].dropna()
        g = sub.groupby('id', sort=False)
        obs[roi]  = g.apply(lambda d: ols_slope(d['time'].values.astype(float),
                                                d[yc].values.astype(float)))
        pred[roi] = g.apply(lambda d: ols_slope(d['time'].values.astype(float),
                                                d[sc].values.astype(float)))
    obs, pred = pd.DataFrame(obs), pd.DataFrame(pred)
    obs.index.name = pred.index.name = 'PTID'
    return obs, pred


def baseline_volumes(path):
    """ROI volumes at each subject's first visit."""
    bdf = pd.read_csv(path)
    vcols = [c for c in bdf.columns if c.startswith('y_H_MUSE_Volume')]
    base = (bdf.sort_values('time').groupby('id').head(1)
               .set_index('id')[vcols])
    base.index.name = 'PTID'
    return base


def progression_labels(cov):
    """Baseline-MCI subjects (>= 2 visits). Progressor: last recorded
    diagnosis is AD. Non-progressor: otherwise."""
    rec = []
    for ptid, grp in cov.groupby('PTID'):
        grp = grp.sort_values('Time')
        if len(grp) < 2:
            continue
        try:
            dx0 = float(grp.iloc[0]['Diagnosis'])
            dxN = float(grp.iloc[-1]['Diagnosis'])
        except (TypeError, ValueError):
            continue
        if dx0 == 1:
            rec.append({'PTID': ptid, 'is_progressor': int(dxN == 2)})
    return pd.DataFrame(rec).set_index('PTID')['is_progressor']


# =============================================================================
# PROPENSITY SCORE MATCHING
# =============================================================================

def propensity_score_matching(ptids, y, cov, base, change_set, stable_set):
    """1:1 nearest-neighbour matching of scanner-change to scanner-stable
    subjects on standardized age, sex, total brain volume and progression
    label, caliper = CALIPER_SD * SD(propensity score), without replacement."""
    first = cov.sort_values('Time').groupby('PTID').head(1).set_index('PTID')
    rows = []
    for ptid, lab in zip(ptids, y):
        if ptid not in first.index:
            continue
        if RESTRICT_MATCHING_TO_KNOWN_SCANNER and \
                ptid not in change_set and ptid not in stable_set:
            continue
        sex = first.at[ptid, 'Sex']
        if isinstance(sex, str):
            sex = 1.0 if sex.upper() in ('M', 'MALE', '1') else 0.0
        rows.append({'PTID': ptid,
                     'age': first.at[ptid, 'Age'],
                     'sex': sex,
                     'tbv': base.loc[ptid].sum() if ptid in base.index else np.nan,
                     'is_progressor': int(lab),
                     'is_change': int(ptid in change_set)})
    m = pd.DataFrame(rows).dropna(subset=['age', 'sex', 'tbv', 'is_progressor'])

    covars = ['age', 'sex', 'tbv', 'is_progressor']
    Xs = StandardScaler().fit_transform(m[covars].values)
    m['ps'] = (LogisticRegression(max_iter=1000, random_state=SEED)
               .fit(Xs, m['is_change'].values).predict_proba(Xs)[:, 1])
    caliper = CALIPER_SD * m['ps'].std()

    ch = m[m['is_change'] == 1].sort_values('ps').reset_index(drop=True)
    st = m[m['is_change'] == 0].reset_index(drop=True)
    used, mc, ms = set(), [], []
    for _, r in ch.iterrows():
        cand = st[~st['PTID'].isin(used)].copy()
        cand['dist'] = (cand['ps'] - r['ps']).abs()
        cand = cand[cand['dist'] <= caliper]
        if cand.empty:
            continue
        best = cand.loc[cand['dist'].idxmin(), 'PTID']
        mc.append(r['PTID']); ms.append(best); used.add(best)

    matched = m[m['PTID'].isin(set(mc) | set(ms))]
    print(f'  Propensity caliper : {CALIPER_SD} SD = {caliper:.4f}')
    print(f'  Matched pairs      : {len(mc)}  (n = {2*len(mc)})')
    for c in covars:
        a = matched.loc[matched['is_change'] == 1, c].mean()
        b = matched.loc[matched['is_change'] == 0, c].mean()
        sd = matched[c].std()
        print(f'    SMD {c:<14}: {(a - b) / sd if sd > 0 else 0:+.3f}')
    for lbl, flag in (('change', 1), ('stable', 0)):
        rate = matched.loc[matched['is_change'] == flag, 'is_progressor'].mean()
        print(f'    Progression rate ({lbl}): {100*rate:.1f}%')
    return set(mc), set(ms)


# =============================================================================
# CLASSIFICATION (one model per fold, metrics stratified by subgroup)
# =============================================================================

def subgroup_metrics(yt, yp, yb):
    if len(yt) < 5 or yt.sum() < 2:
        return None
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    fpr_c, tpr_c, _ = roc_curve(yt, yb)
    return {'n': len(yt), 'n_prog': int(yt.sum()),
            'auc': roc_auc_score(yt, yb),
            'pr_auc': average_precision_score(yt, yb),
            'precision': precision_score(yt, yp, zero_division=0),
            'recall': recall_score(yt, yp, zero_division=0),
            'f1': f1_score(yt, yp, zero_division=0),
            'fpr': fp / (fp + tn) if (fp + tn) else 0.0,
            'roc': (fpr_c, tpr_c)}


def evaluate(X, y, ptids, subgroups):
    """Nested CV logistic regression on the full cohort.
    subgroups: dict name -> set of PTIDs evaluated within each test fold."""
    outer = StratifiedKFold(OUTER_FOLDS, shuffle=True, random_state=SEED)
    inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=SEED)
    grid  = np.linspace(0, 1, 300)

    folds = {g: [] for g in subgroups}
    oof   = {'PTID': [], 'y_true': [], 'y_prob': [], 'y_pred': []}

    for tr, te in outer.split(X, y):
        sc = StandardScaler().fit(X[tr])
        gs = GridSearchCV(LogisticRegression(max_iter=2000, class_weight='balanced',
                                             random_state=SEED),
                          {'C': C_GRID}, cv=inner, scoring='roc_auc',
                          n_jobs=-1, refit=True).fit(sc.transform(X[tr]), y[tr])
        Xte  = sc.transform(X[te])
        prob = gs.best_estimator_.predict_proba(Xte)[:, 1]
        pred = gs.best_estimator_.predict(Xte)          # 0.5 threshold
        oof['PTID'] += list(ptids[te]); oof['y_true'] += list(y[te])
        oof['y_prob'] += list(prob);    oof['y_pred'] += list(pred)

        for g, members in subgroups.items():
            mask = np.isin(ptids[te], list(members))
            m = subgroup_metrics(y[te][mask], pred[mask], prob[mask])
            if m is not None:
                folds[g].append(m)

    summary = {}
    for g, fl in folds.items():
        aucs = np.array([f['auc'] for f in fl])
        tprs = []
        for f in fl:
            t = np.interp(grid, *f['roc']); t[0] = 0.0; tprs.append(t)
        mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
        summary[g] = {
            'n': int(sum(f['n'] for f in fl)),
            'n_prog': int(sum(f['n_prog'] for f in fl)),
            'roc_auc': aucs.mean(), 'roc_auc_std': aucs.std(),
            **{k: np.mean([f[k] for f in fl])
               for k in ('pr_auc', 'precision', 'recall', 'f1', 'fpr')},
            'fpr_curve': grid, 'tpr_curve': mean_tpr,
        }
    return summary, pd.DataFrame(oof)


def delong_test(y_true, prob_a, prob_b):
    """Paired DeLong test for two correlated ROC-AUCs."""
    y_true = np.asarray(y_true)

    def components(p):
        pos, neg = p[y_true == 1], p[y_true == 0]
        v10 = np.array([np.mean(x > neg) + 0.5 * np.mean(x == neg) for x in pos])
        v01 = np.array([np.mean(pos > x) + 0.5 * np.mean(pos == x) for x in neg])
        return v10.mean(), v10, v01

    auc_a, v10a, v01a = components(np.asarray(prob_a))
    auc_b, v10b, v01b = components(np.asarray(prob_b))
    S = (np.cov(np.vstack([v10a, v10b])) / len(v10a) +
         np.cov(np.vstack([v01a, v01b])) / len(v01a))
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return auc_a, auc_b, 0.0, 1.0
    z = (auc_a - auc_b) / np.sqrt(var)
    return auc_a, auc_b, z, 2 * stats.norm.sf(abs(z))


# =============================================================================
# FIGURES
# =============================================================================

STYLE = {'Observed': '#666666', 'p-DKGP': '#8B0000'}
plt.rcParams.update({'font.size': 16, 'axes.linewidth': 1.5,
                     'axes.labelsize': 16, 'xtick.labelsize': 16,
                     'ytick.labelsize': 16, 'figure.dpi': 300,
                     'axes.grid': True, 'grid.alpha': 0.3,
                     'grid.linestyle': '--'})


def plot_roc_panels(results, stable_key, change_key, suffix):
    """Figure 5b: ROC curves + operating point (recall at the 0.5 threshold)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, (g, title) in zip(axes, [(stable_key, 'Scanner-stable'),
                                     (change_key, 'Scanner-change')]):
        for feat in ('Observed', 'p-DKGP'):
            r, col = results[feat][g], STYLE[feat]
            ax.plot(r['fpr_curve'], r['tpr_curve'], color=col, lw=2,
                    label=f"{FEATURES[feat]} (AUC {r['roc_auc']:.3f}±{r['roc_auc_std']:.3f})")
            ax.scatter(r['fpr'], r['recall'], color=col, s=70, zorder=5,
                       edgecolor='black', linewidth=1.0)
            ax.annotate(f"{r['recall']:.3f}", (r['fpr'], r['recall']),
                        xytext=(8, -4 if feat == 'Observed' else 8),
                        textcoords='offset points', fontsize=13,
                        color=col, fontweight='bold')
        ax.plot([0, 1], [0, 1], '--', color='#AAAAAA', lw=1.5, label='Chance')
        ax.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.01), title=title,
               xlabel='False Positive Rate', ylabel='True Positive Rate')
        ax.legend(title=f"n={results['p-DKGP'][g]['n']} MCI subjects",
                  fontsize=15, title_fontsize=15, loc='lower right')
    plt.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        plt.savefig(f'{OUT_DIR}/subgroup_roc_panels_{suffix}.{ext}',
                    dpi=300, bbox_inches='tight')
    plt.close()


def scanner_change_times(ptid, cov):
    s = cov[cov['PTID'] == ptid].sort_values('Time').dropna(subset=['MRI_Scanner_Model'])
    m, t = s['MRI_Scanner_Model'].tolist(), s['Time'].tolist()
    return [t[i] for i in range(1, len(m)) if m[i] != m[i - 1]]


def plot_trajectory(traj, roi, label, ptid, age, changes):
    """Figure 5a panel: observed vs p-DKGP-predicted volume for one ROI."""
    traj = traj.sort_values('time')
    yc, sc = f'y_H_MUSE_Volume_{roi}', f'score_H_MUSE_Volume_{roi}'
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(traj['time'], traj[yc], 'o-', label='Actual', color='#666666',
            alpha=0.8, markersize=8, lw=2)
    ax.plot(traj['time'], traj[sc], 's-', label='Predicted', color='#8B0000',
            alpha=0.8, markersize=8, lw=2)
    for i, t in enumerate(changes):
        ax.axvline(t, color='#8B0000', ls='--', alpha=0.8, lw=2,
                   label='Scanner Change' if i == 0 else '')
    dx_map = {0: 'CN', 1: 'MCI', 2: 'AD'}
    for _, r in traj.dropna(subset=['Diagnosis']).iterrows():
        ax.annotate(dx_map.get(int(r['Diagnosis']), str(r['Diagnosis'])),
                    (r['time'], r[yc]), xytext=(5, 5), textcoords='offset points',
                    fontsize=12, bbox=dict(boxstyle='round,pad=0.3',
                                           facecolor='white', alpha=0.8))
    ax.set(xlabel='Time from baseline (in months)', ylabel=f'{label} Volume',
           title=label)
    ax.legend(fontsize=14)
    fig.suptitle(f'Subject {ptid} (n={len(traj)} visits'
                 + (f', baseline age {age:.1f} y)' if age is not None else ')'),
                 fontsize=18, y=1.02)
    plt.tight_layout()
    for ext in ('png', 'pdf', 'svg'):
        plt.savefig(f'{OUT_DIR}/ScannerContamination_{label}_{ptid}.{ext}',
                    dpi=300, bbox_inches='tight')
    plt.close()


def figure_5a_examples(oof, change_set, dkgp_raw, cov):
    """Matched scanner-change true progressors classified as non-progressor
    by beta_obs (p < 0.5) and as progressor by beta_pDKGP (p >= 0.5),
    ranked by probability gap."""
    df = (oof['Observed'][['PTID', 'y_true', 'y_prob']]
          .rename(columns={'y_prob': 'prob_obs'})
          .merge(oof['p-DKGP'][['PTID', 'y_prob']]
                 .rename(columns={'y_prob': 'prob_dkgp'}), on='PTID'))
    df = df[df['PTID'].isin(change_set) & (df['y_true'] == 1)
            & (df['prob_obs'] < 0.5) & (df['prob_dkgp'] >= 0.5)].copy()
    df['prob_gap'] = df['prob_dkgp'] - df['prob_obs']
    df = df.sort_values('prob_gap', ascending=False)
    df.to_csv(f'{OUT_DIR}/scanner_contamination_candidates.csv', index=False)
    print(f'  {len(df)} candidates')
    if not df.empty:
        print(df.to_string(index=False))

    for ptid in df['PTID'].head(N_EXAMPLES):
        traj = dkgp_raw[dkgp_raw['id'] == ptid]
        sc = cov[cov['PTID'] == ptid].sort_values('Time')
        traj = traj.merge(sc[['Time', 'Diagnosis']].rename(columns={'Time': 'time'}),
                          on='time', how='left')
        age = sc.iloc[0]['Age'] if len(sc) else None
        ch  = scanner_change_times(ptid, cov)
        plot_trajectory(traj, HIPPOCAMPUS, 'Hippocampus', ptid, age, ch)
        plot_trajectory(traj, VENTRICLES, 'Ventricles', ptid, age, ch)


# =============================================================================
# MAIN
# =============================================================================

class Tee:
    def __init__(self, path):
        self._f, self._s = open(path, 'w'), sys.__stdout__
    def write(self, t):
        self._s.write(t); self._f.write(t)
    def flush(self):
        self._s.flush(); self._f.flush()


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    sys.stdout = Tee(f'{OUT_DIR}/scanner_subgroup_results.txt')
    sep = '=' * 70

    # ---- Inputs --------------------------------------------------------
    cov = pd.read_csv(COVARIATES)
    stable_set, change_set = load_scanner_groups(cov)

    dkgp_raw = pd.read_csv(DKGP_IN)
    roi_cols = [c for c in dkgp_raw.columns if c.startswith(('y_H_MUSE', 'score_H_MUSE'))]
    dkgp_raw = dkgp_raw[~dkgp_raw['id'].isin(
        dkgp_raw.loc[dkgp_raw[roi_cols].isnull().any(axis=1), 'id'].unique())]
    rnn_raw = pd.read_csv(RNN_IN)

    _, beta_dkgp = paired_slopes(dkgp_raw)            # beta_pDKGP (predicted)
    beta_obs, _  = paired_slopes(rnn_raw)             # beta_obs (observed)
    rnn_ids      = set(beta_obs.index)
    base         = baseline_volumes(BASELINE_IN)
    labels    = progression_labels(cov)

    # ---- Common cohort (n = 882) --------------------------------------
    ptids = np.array(sorted(set(labels.index) & set(beta_dkgp.index)
                            & set(beta_obs.index) & rnn_ids & set(base.index)))
    y = labels.loc[ptids].values
    X = {'p-DKGP':   beta_dkgp.loc[ptids].values,
         'Observed': beta_obs.loc[ptids].values}

    print(sep)
    print('SECTION 2.6 - MCI PROGRESSION UNDER SCANNER CHANGE')
    print(sep)
    print(f'Cohort          : n = {len(ptids)}  ({y.sum()} progressors, '
          f'{100*y.mean():.1f}%)')
    print(f'Scanner-stable  : {np.isin(ptids, list(stable_set)).sum()}')
    print(f'Scanner-change  : {np.isin(ptids, list(change_set)).sum()}')
    print(f'No scanner info : {(~np.isin(ptids, list(stable_set | change_set))).sum()}')

    # ---- Propensity score matching ------------------------------------
    print('\n-- Propensity score matching --')
    m_change, m_stable = propensity_score_matching(
        ptids, y, cov, base, change_set, stable_set)

    # ---- Classification -----------------------------------------------
    subgroups = {'stable': stable_set, 'change': change_set,
                 'matched_stable': m_stable, 'matched_change': m_change}
    results, oof = {}, {}
    for feat in FEATURES:
        results[feat], oof[feat] = evaluate(X[feat], y, ptids, subgroups)

    # ---- Summary table ------------------------------------------------
    rows = []
    for g in subgroups:
        print(f"\n-- {g} (n={results['p-DKGP'][g]['n']}, "
              f"progressors={results['p-DKGP'][g]['n_prog']}) --")
        print(f"  {'':<10}{'AUC':>16}{'PR-AUC':>9}{'Prec':>8}{'Recall':>8}{'F1':>8}{'FPR':>8}")
        for feat in FEATURES:
            r = results[feat][g]
            print(f"  {feat:<10}{r['roc_auc']:>9.3f}±{r['roc_auc_std']:.3f}"
                  f"{r['pr_auc']:>9.3f}{r['precision']:>8.3f}{r['recall']:>8.3f}"
                  f"{r['f1']:>8.3f}{r['fpr']:>8.3f}")
            rows.append({'subgroup': g, 'feature': feat,
                         **{k: r[k] for k in ('n', 'n_prog', 'roc_auc', 'roc_auc_std',
                                              'pr_auc', 'precision', 'recall',
                                              'f1', 'fpr')}})
    pd.DataFrame(rows).to_csv(f'{OUT_DIR}/subgroup_metrics.csv', index=False)

    rc = results['p-DKGP']['matched_change']['recall'] - \
         results['Observed']['matched_change']['recall']
    print(f'\nMatched scanner-change recall advantage (p-DKGP - observed): '
          f'{100*rc:+.1f} percentage points')

    # ---- DeLong, pooled full cohort -----------------------------------
    a, b = oof['p-DKGP'], oof['Observed']
    assert (a['PTID'].values == b['PTID'].values).all()
    auc_a, auc_b, z, p = delong_test(a['y_true'], a['y_prob'], b['y_prob'])
    print(f'\nDeLong (pooled out-of-fold, n={len(a)}): '
          f'AUC p-DKGP={auc_a:.3f}, observed={auc_b:.3f}, z={z:.3f}, p={p:.2f}')

    # ---- Figures ------------------------------------------------------
    plot_roc_panels(results, 'matched_stable', 'matched_change', 'matched')   # Fig 5b
    plot_roc_panels(results, 'stable', 'change', 'unmatched')
    print('\n-- Figure 5a candidates --')
    figure_5a_examples(oof, m_change, dkgp_raw, cov)

    with open(f'{OUT_DIR}/scanner_subgroup_cache.pkl', 'wb') as f:
        pickle.dump({'results': results, 'oof': oof,
                     'matched_change': m_change, 'matched_stable': m_stable,
                     'delong': {'auc_pdkgp': auc_a, 'auc_obs': auc_b,
                                'z': z, 'p': p}}, f)
    print(f'\n{sep}\nDONE -> {OUT_DIR}\n{sep}')