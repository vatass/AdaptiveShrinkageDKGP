#!/usr/bin/env python3
"""
Section 2.8 (predicted brain age gap) / Methods 4.12

Brain age gap (BAG) = SPARE-BA - chronological age, at every visit.
  predicted BAG : p-DKGP population-model SPARE-BA (baseline inputs only)
  observed BAG  : measured SPARE-BA

Per-subject BAG slope (years/year)
----------------------------------
  OLS regression of BAG on chronological age over the subject's visits.
  A subject is included if all of the following hold:
    >= 2 visits, no missing age/BAG, age span >= 1 year,
    non-zero variance in age, observed BAG and predicted BAG,
    |observed slope| <= 10 and |predicted slope| <= 10.
  Groups: baseline diagnosis CN / MCI / AD (first recorded diagnosis).

Statistics (reported in Section 2.8)
------------------------------------
  - Group mean +/- SD of predicted and observed slopes
  - One-way ANOVA across CN/MCI/AD; eta^2 = SS_between / SS_total
  - Variance ratio (predicted / observed) within each group
  - Pairwise t-tests (CN-MCI, MCI-AD, CN-AD)
  - Glass's Delta = difference in group mean PREDICTED slopes divided by
    the pooled SD of the OBSERVED slopes of the two groups
  - Pearson r between BAG and chronological age within each group,
    observed vs predicted, computed over all visits; difference tested
    with Fisher's z-transformation

Outputs (--outdir)
------------------
  bag_log.txt, bag_subject_slopes.csv, bag_group_statistics.csv,
  bag_anova.csv, bag_pairwise.csv, bag_age_correlation.csv
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# SPARE-BA z-score parameters used to put predictions back in years
SPARE_BA_MEAN = 74.40936529687391
SPARE_BA_STD = 13.093899067591588
GROUPS = ['CN', 'MCI', 'AD']
PAIRS = [('CN', 'MCI'), ('MCI', 'AD'), ('CN', 'AD')]


# =============================================================================
# DATA
# =============================================================================

def load(pred_csv, cov_csv):
    pred = pd.read_csv(pred_csv)
    cov = pd.read_csv(cov_csv)
    pred['score'] = pred['score'] * SPARE_BA_STD + SPARE_BA_MEAN
    df = pred.merge(cov, left_on=['id', 'time'], right_on=['PTID', 'Time'], how='inner')
    df['predicted_bag'] = df['score'] - df['Age']
    df['observed_bag'] = df['SPARE_BA'] - df['Age']

    # baseline diagnosis = first recorded diagnosis per subject
    base_dx = cov.groupby('PTID').first()['Diagnosis']
    df['Group'] = df['PTID'].map(base_dx.map({0: 'CN', 1: 'MCI', 2: 'AD'}))
    return df[df['Group'].isin(GROUPS)].copy()


# =============================================================================
# SLOPES
# =============================================================================

def bag_slope(s):
    """Observed and predicted BAG-vs-age slopes, or None if excluded."""
    s = s.sort_values('Age')
    cols = ['Age', 'observed_bag', 'predicted_bag']
    if (len(s) < 2 or s[cols].isna().any().any() or s['Age'].var() == 0
            or s['observed_bag'].var() == 0 or s['predicted_bag'].var() == 0
            or s['Age'].max() - s['Age'].min() < 1.0):
        return None
    obs = stats.linregress(s['Age'], s['observed_bag']).slope
    pred = stats.linregress(s['Age'], s['predicted_bag']).slope
    if not (np.isfinite(obs) and np.isfinite(pred)) or abs(obs) > 10 or abs(pred) > 10:
        return None
    return obs, pred


def subject_slopes(df):
    rows, n2 = [], 0
    for pid in df['PTID'].unique():
        s = df[df['PTID'] == pid]
        if len(s) >= 2:
            n2 += 1
        r = bag_slope(s) if len(s) >= 2 else None
        if r is None:
            continue
        rows.append({'PTID': pid, 'Group': s.iloc[0]['Group'],
                     'Baseline_Age': s.iloc[0]['Age'], 'N_Visits': len(s),
                     'Age_Span': s['Age'].max() - s['Age'].min(),
                     'Observed_BAG_Slope': r[0], 'Predicted_BAG_Slope': r[1]})
    out = pd.DataFrame(rows)
    print(f'  Subjects in CN/MCI/AD          : {df["PTID"].nunique()}')
    print(f'  With >= 2 visits               : {n2}')
    print(f'  With valid slopes (analysed)   : {len(out)}')
    return out


# =============================================================================
# STATISTICS
# =============================================================================

def eta_squared(groups):
    allv = np.concatenate(groups)
    gm = allv.mean()
    ss_b = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
    ss_t = ((allv - gm) ** 2).sum()
    return ss_b / ss_t


def group_statistics(sl):
    rows = []
    for g in GROUPS:
        d = sl[sl['Group'] == g]
        o, p = d['Observed_BAG_Slope'], d['Predicted_BAG_Slope']
        rows.append({'Group': g, 'N': len(d),
                     'Observed_mean': o.mean(), 'Observed_SD': o.std(),
                     'Predicted_mean': p.mean(), 'Predicted_SD': p.std(),
                     'Variance_ratio_pred_over_obs': p.var() / o.var()})
    return pd.DataFrame(rows)


def anova(sl):
    rows = []
    for name, col in (('Predicted', 'Predicted_BAG_Slope'), ('Observed', 'Observed_BAG_Slope')):
        g = [sl.loc[sl['Group'] == k, col].values for k in GROUPS]
        f, p = stats.f_oneway(*g)
        n = sum(len(x) for x in g)
        rows.append({'Slopes': name, 'df_between': len(g) - 1, 'df_within': n - len(g),
                     'F': f, 'p': p, 'eta_squared': eta_squared(g)})
    return pd.DataFrame(rows)


def pairwise(sl):
    rows = []
    for a, b in PAIRS:
        da, db = sl[sl['Group'] == a], sl[sl['Group'] == b]
        pa, pb = da['Predicted_BAG_Slope'].values, db['Predicted_BAG_Slope'].values
        oa, ob = da['Observed_BAG_Slope'].values, db['Observed_BAG_Slope'].values
        na, nb = len(pa), len(pb)
        pooled_obs = np.sqrt(((na - 1) * oa.var(ddof=1) + (nb - 1) * ob.var(ddof=1))
                             / (na + nb - 2))
        t_p, p_p = stats.ttest_ind(pa, pb)
        t_o, p_o = stats.ttest_ind(oa, ob)
        rows.append({'Comparison': f'{a} vs {b}', 'N_a': na, 'N_b': nb,
                     'Predicted_mean_diff': pa.mean() - pb.mean(),
                     'Pooled_observed_SD': pooled_obs,
                     'Glass_Delta': (pa.mean() - pb.mean()) / pooled_obs,
                     't_predicted': t_p, 'p_predicted': p_p,
                     't_observed': t_o, 'p_observed': p_o})
    return pd.DataFrame(rows)


def age_correlation(df):
    """Pearson r(BAG, age) within group over all visits; Fisher z difference."""
    rows = []
    for g in GROUPS:
        d = df[df['Group'] == g]
        if len(d) <= 10:
            continue
        r_o, p_o = stats.pearsonr(d['Age'], d['observed_bag'])
        r_p, p_p = stats.pearsonr(d['Age'], d['predicted_bag'])
        z = (np.arctanh(r_p) - np.arctanh(r_o)) / np.sqrt(2 / (len(d) - 3))
        rows.append({'Group': g, 'N_visits': len(d), 'N_subjects': d['PTID'].nunique(),
                     'r_observed': r_o, 'p_observed': p_o,
                     'r_predicted': r_p, 'p_predicted': p_p,
                     'Fisher_z': z, 'Fisher_p': 2 * stats.norm.sf(abs(z))})
    return pd.DataFrame(rows)


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
    ap = argparse.ArgumentParser(description='Section 2.8 brain age gap slopes')
    ap.add_argument('--predictions_csv',
                    default='./manuscript1/singletask_SPARE_BA_dkgp_population_allstudies.csv')
    ap.add_argument('--covariates_csv', default='longitudinal_covariates_'
                    'subjectsamples_longclean_hmuse_convs_allstudies.csv')
    ap.add_argument('--outdir', default='./manuscript1/brain_age_gap')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    out = lambda n: os.path.join(a.outdir, n)
    sys.stdout = Tee(out('bag_log.txt'))
    pd.set_option('display.width', 200)

    print('Section 2.8 - predicted brain age gap slopes (p-DKGP population model)')
    df = load(a.predictions_csv, a.covariates_csv)
    sl = subject_slopes(df)
    sl.to_csv(out('bag_subject_slopes.csv'), index=False)

    gs = group_statistics(sl)
    an = anova(sl)
    pw = pairwise(sl)
    cr = age_correlation(df)
    for name, tab in (('Group statistics', gs), ('One-way ANOVA', an),
                      ("Pairwise t-tests and Glass's Delta", pw),
                      ('BAG vs age correlation (all visits)', cr)):
        print(f'\n-- {name} --\n{tab.to_string(index=False)}')
    gs.to_csv(out('bag_group_statistics.csv'), index=False)
    an.to_csv(out('bag_anova.csv'), index=False)
    pw.to_csv(out('bag_pairwise.csv'), index=False)
    cr.to_csv(out('bag_age_correlation.csv'), index=False)
    print(f'\nOutputs -> {a.outdir}')


if __name__ == '__main__':
    main()