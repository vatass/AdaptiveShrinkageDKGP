# ==============================================================================
# Experiment: Longitudinal Brain Age Gap (BAG) Slope Analysis
#             by Diagnostic and Progression Groups
# ==============================================================================
#
# OVERVIEW
# --------
# This script analyses how the Brain Age Gap (BAG = predicted brain age minus
# chronological age) changes over time across diagnostic groups (CN, MCI, AD)
# and fine-grained progression statuses (e.g. MCI Stable vs MCI Progressor).
# BAG slopes are computed from both real (observed) SPARE-BA values and
# DKGP-predicted SPARE-BA trajectories, allowing direct comparison of whether
# the predicted trajectories preserve clinically meaningful longitudinal signal.
#
# MOTIVATION
# ----------
# A positive BAG slope (brain aging faster than chronological age) is a marker
# of accelerated neurodegeneration. If the DKGP model captures true biological
# trajectories, predicted BAG slopes should: (1) increase systematically from
# CN to MCI to AD, (2) separate MCI Progressors from MCI Stable subjects, and
# (3) correlate with real BAG slopes at the individual level. Validating these
# properties establishes that DKGP-predicted trajectories are biologically
# meaningful and not merely smooth averages.
#
# KEY METHODOLOGICAL DECISIONS
# ----------------------------
# 1. Effect size — Glass's Delta (not Cohen's d):
#      DKGP produces smoother trajectories than raw longitudinal data because
#      it learns the systematic signal while attenuating subject-level noise.
#      This mechanically compresses within-group variance in predicted slopes,
#      making Cohen's d computed on predicted slopes artificially inflated and
#      incomparable to Cohen's d on real slopes. Glass's Delta resolves this
#      by anchoring the denominator to the pooled SD of the OBSERVED slopes,
#      making the effect size invariant to the degree of model smoothing.
#      Cohen's d on predicted slopes is computed for reference only and is
#      explicitly flagged as not reportable.
#
# 2. Eta squared — computed from sums of squares:
#      η² is computed directly from SS_between / SS_total rather than from
#      the F-statistic formula, which is only correct for balanced designs.
#      The SS approach is exact for unequal group sizes.
#
# 3. Pairwise comparisons — all three clinically relevant pairs:
#      CN vs MCI, MCI vs AD, and CN vs AD (rather than CN vs AD only).
#
# 4. BAG slope quality filters:
#      Subjects are excluded if they have fewer than 2 timepoints, an age
#      span < 1 year, zero variance in age or BAG, or any NaN in the relevant
#      columns. Slopes with |slope| > 10 years/year are also excluded as
#      implausible outliers.
#
# EXPERIMENTAL DESIGN
# -------------------
# Two levels of grouping are analysed:
#
#   Level 1 — Baseline diagnosis (3 groups):
#     CN (Diagnosis=0), MCI (Diagnosis=1), AD (Diagnosis=2)
#
#   Level 2 — Longitudinal progression status (6 groups):
#     Healthy Control, CN→MCI Progressor, CN→AD Progressor,
#     MCI Stable, MCI Progressor, AD
#
# For each subject with ≥2 timepoints, a BAG slope is estimated as the OLS
# regression coefficient of BAG on age (years/year). Slopes are computed
# separately for real and predicted BAG values and compared across groups.
#
# DATA
# ----
# - SPARE-BA predictions : singletask_SPARE_BA_dkgp_population_allstudies.csv
#                          (population-level DKGP model, column "score")
# - Adaptive shrinkage   : adaptive_shrinkage_predictions_alpha_simple_dkgp_
#                          SPARE_BA_allstudies.csv (alternative model)
# - Longitudinal covars  : longitudinal_covariates_subjectsamples_longclean_
#                          hmuse_convs_allstudies.csv
#                          (Age, Sex, Diagnosis, SPARE_BA per visit)
# - SPARE-BA rescaling   : scores are z-normalised in storage; rescaled to
#                          original units using mean=74.409, std=13.094
#
# STATISTICAL TESTS
# -----------------
# - One-way ANOVA + η² (SS-based): group differences in BAG slopes
# - Pairwise independent-samples t-tests: all three CN/MCI/AD pairs
# - Glass's Delta: group separation in predicted slopes, anchored to
#   observed SD (smoothing-invariant, suitable for reporting)
# - Pearson correlation + Fisher z-test: BAG–age correlation within groups,
#   and whether real vs predicted correlations differ significantly
#
# OUTPUTS
# -------
# Figures (all saved to ./nataging/):
#   SPAREBA_Experiment_bag_slope_distributions_by_group.{png,svg}
#       Violin + box plots of real and predicted BAG slopes by CN/MCI/AD
#   SPAREBA_Experiment_nature_slope_distributions.{png,svg}
#       Nature Aging-style version of the above
#   SPAREBA_Experiment_nature_bag_vs_age_scatter.{png,svg}
#       BAG vs age scatterplots with regression lines (2×3 grid)
#   SPAREBA_Experiment_nature_correlation_comparison.{png,svg}
#       Bar chart of BAG–age correlation by group (real vs predicted)
#   SPAREBA_Experiment_nature_mean_slopes.{png,svg}
#       Mean BAG slope bar chart with error bars (real vs predicted)
#   SPAREBA_Experiment_progression_slope_distributions.{png,svg}
#       Violin + box plots by fine-grained progression status (6 groups)
#   SPAREBA_Experiment_progression_mean_slopes.{png,svg}
#       Mean BAG slope bar chart by progression status
#   SPAREBA_Experiment_progression_real_vs_pred_scatter.{png,svg}
#       Scatter of real vs predicted BAG slopes coloured by progression status
#
# CSVs (all saved to ./nataging/):
#   SPAREBA_Experiment_individual_bag_slopes.csv
#       Per-subject real and predicted BAG slopes + metadata
#   SPAREBA_Experiment_group_slope_statistics.csv
#       Summary statistics (mean, std, median, % positive) per CN/MCI/AD group
#   SPAREBA_Experiment_glass_delta_summary.csv
#       Glass's Delta and Cohen's d for all three pairwise comparisons
#   SPAREBA_Experiment_correlation_analysis.csv
#       BAG–age correlation results (real vs predicted) per group
#   SPAREBA_Experiment_nature_statistical_summary.csv
#       Compact summary table for manuscript
#   SPAREBA_Experiment_progression_individual_slopes.csv
#       Per-subject slopes filtered to the 6 progression status groups
#   SPAREBA_Experiment_progression_group_statistics.csv
#       Summary statistics per progression status group
# ==============================================================================
 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import scipy.stats as stats
import matplotlib as mpl
from scipy.stats import f_oneway, kruskal
import os
import warnings
warnings.filterwarnings('ignore')

# ── Ensure output directory exists ───────────────────────────────────────────
os.makedirs('./nataging', exist_ok=True)

# ── Publication-quality plot parameters ──────────────────────────────────────
plt.style.use('default')
mpl.rcParams['font.size']       = 12
mpl.rcParams['axes.linewidth']  = 1.5
mpl.rcParams['axes.labelsize']  = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['figure.dpi']      = 300

print("="*80)
print("LONGITUDINAL BAG SLOPE ANALYSIS BY DIAGNOSTIC GROUPS")
print("="*80)


# =============================================================================
# EFFECT SIZE FUNCTIONS
# =============================================================================

def compute_glass_delta(pred_slopes_a, pred_slopes_b,
                         obs_slopes_a,  obs_slopes_b,
                         label_a='Group A', label_b='Group B'):
    """
    Compute Glass's Delta for group separation in predicted BAG slopes,
    normalised by the pooled standard deviation of the OBSERVED slopes.

    Rationale
    ---------
    p-DKGP produces smoother trajectories than raw longitudinal measurements
    because it learns the systematic signal while attenuating subject-level
    noise. This model smoothing mechanically reduces within-group variance in
    predicted slopes relative to observed slopes. Standard Cohen's d computed
    on predicted slopes therefore has a compressed denominator regardless of
    whether group separation has genuinely improved, making it an unreliable
    measure of effect size and incomparable across real and predicted slopes.

    Glass's Delta resolves this by anchoring the denominator to the pooled
    standard deviation of the OBSERVED slopes, making the effect size invariant
    to the degree of smoothing introduced by the predictive model.
    """
    pred_slopes_a = np.asarray(pred_slopes_a)
    pred_slopes_b = np.asarray(pred_slopes_b)
    obs_slopes_a  = np.asarray(obs_slopes_a)
    obs_slopes_b  = np.asarray(obs_slopes_b)

    n_a = len(pred_slopes_a)
    n_b = len(pred_slopes_b)

    mean_diff = np.mean(pred_slopes_a) - np.mean(pred_slopes_b)

    # Pooled SD from OBSERVED slopes — invariant to model smoothing
    pooled_sd_obs = np.sqrt(
        ((n_a - 1) * np.std(obs_slopes_a, ddof=1)**2 +
         (n_b - 1) * np.std(obs_slopes_b, ddof=1)**2) /
        (n_a + n_b - 2)
    )

    # Pooled SD from PREDICTED slopes — deflated by smoothing; not reported
    pooled_sd_pred = np.sqrt(
        ((n_a - 1) * np.std(pred_slopes_a, ddof=1)**2 +
         (n_b - 1) * np.std(pred_slopes_b, ddof=1)**2) /
        (n_a + n_b - 2)
    )

    glass_delta   = mean_diff / pooled_sd_obs  if pooled_sd_obs  > 0 else float('nan')
    cohens_d_pred = mean_diff / pooled_sd_pred if pooled_sd_pred > 0 else float('nan')

    print(f"\n{'─'*60}")
    print(f"  {label_a} vs {label_b}")
    print(f"{'─'*60}")
    print(f"  Mean predicted slope ({label_a:<22}): "
          f"{np.mean(pred_slopes_a):+.4f}")
    print(f"  Mean predicted slope ({label_b:<22}): "
          f"{np.mean(pred_slopes_b):+.4f}")
    print(f"  Mean difference (predicted)          : {mean_diff:+.4f}")
    print(f"\n  SD predicted ({label_a:<22}): "
          f"{np.std(pred_slopes_a, ddof=1):.4f}  [compressed by smoothing]")
    print(f"  SD predicted ({label_b:<22}): "
          f"{np.std(pred_slopes_b, ddof=1):.4f}  [compressed by smoothing]")
    print(f"  SD observed  ({label_a:<22}): "
          f"{np.std(obs_slopes_a, ddof=1):.4f}")
    print(f"  SD observed  ({label_b:<22}): "
          f"{np.std(obs_slopes_b, ddof=1):.4f}")
    print(f"\n  Pooled SD (predicted) — smoothing-affected  : "
          f"{pooled_sd_pred:.4f}")
    print(f"  Pooled SD (observed)  — smoothing-invariant : "
          f"{pooled_sd_obs:.4f}")
    print(f"\n  Cohen's d  (predicted SD) — inflated, DO NOT REPORT : "
          f"{cohens_d_pred:.3f}")
    print(f"  Glass's Delta (observed SD) — REPORT THIS            : "
          f"{glass_delta:.3f}")

    return {
        'glass_delta':    glass_delta,
        'cohens_d_pred':  cohens_d_pred,
        'mean_diff':      mean_diff,
        'pooled_sd_obs':  pooled_sd_obs,
        'pooled_sd_pred': pooled_sd_pred,
        'n_a':            n_a,
        'n_b':            n_b,
    }


def compute_all_pairwise_glass_delta(pred_slopes_by_group,
                                      obs_slopes_by_group,
                                      pairs=None):
    """
    Compute Glass's Delta for all specified pairwise group comparisons.
    Defaults to all three clinically relevant pairs: HC/MCI, MCI/AD, HC/AD.
    """
    if pairs is None:
        pairs = [('HC', 'MCI'), ('MCI', 'AD'), ('HC', 'AD')]

    print(f"\n{'='*60}")
    print("GLASS'S DELTA — PAIRWISE GROUP COMPARISONS")
    print("(Effect size invariant to model smoothing)")
    print(f"{'='*60}")

    results = {}
    for label_a, label_b in pairs:
        if label_a not in pred_slopes_by_group or \
           label_b not in pred_slopes_by_group:
            print(f"\n  Skipping {label_a} vs {label_b} — group not found")
            continue
        key = f"{label_a} vs {label_b}"
        results[key] = compute_glass_delta(
            pred_slopes_by_group[label_a],
            pred_slopes_by_group[label_b],
            obs_slopes_by_group[label_a],
            obs_slopes_by_group[label_b],
            label_a=label_a,
            label_b=label_b
        )

    print(f"\n{'─'*60}")
    print("SUMMARY — Glass's Delta (smoothing-invariant, REPORT THESE)")
    print(f"{'─'*60}")
    print(f"  {'Comparison':<20} {'Mean Diff':>10}  "
          f"{'Glass Δ':>10}  {'Cohen d (pred, inflated)':>25}")
    print(f"  {'-'*70}")
    for key, res in results.items():
        print(f"  {key:<20} {res['mean_diff']:>+10.4f}  "
              f"{res['glass_delta']:>10.3f}  "
              f"{res['cohens_d_pred']:>25.3f}")

    return results


def compute_eta_squared_from_ss(groups):
    """
    Compute eta squared (η²) directly from sums of squares.
    Correct for unequal group sizes.
    """
    all_values = np.concatenate(groups)
    grand_mean = np.mean(all_values)
    ss_total   = np.sum((all_values - grand_mean)**2)
    ss_between = sum(
        len(g) * (np.mean(g) - grand_mean)**2 for g in groups
    )
    eta_sq = ss_between / ss_total if ss_total > 0 else float('nan')
    return eta_sq, ss_between, ss_total


# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n1. LOADING DATA...")
population = pd.read_csv(
    "./manuscript1/singletask_SPARE_BA_dkgp_population_allstudies.csv")
longitudinal_data = pd.read_csv(
    "../LongGPClustering/data1/"
    "longitudinal_covariates_subjectsamples_longclean_hmuse_convs_allstudies.csv")
adaptive_shrinkage_spare_ba = pd.read_csv(
    "./manuscript1/"
    "adaptive_shrinkage_predictions_alpha_simple_dkgp_SPARE_BA_allstudies.csv")

SPARE_BA_MEAN = 74.40936529687391
SPARE_BA_STD  = 13.093899067591588

population['score']                  = (population['score']
                                        * SPARE_BA_STD + SPARE_BA_MEAN)
adaptive_shrinkage_spare_ba['score'] = (adaptive_shrinkage_spare_ba['score']
                                        * SPARE_BA_STD + SPARE_BA_MEAN)

print("Merging datasets...")
merged_data = pd.merge(population, longitudinal_data,
                       left_on=['id', 'time'],
                       right_on=['PTID', 'Time'], how='inner')
adaptive_merged = pd.merge(adaptive_shrinkage_spare_ba, longitudinal_data,
                            left_on=['id', 'time'],
                            right_on=['PTID', 'Time'], how='inner')

print("Calculating BAG values...")
merged_data['predicted_brain_age_gap'] = merged_data['score'] - merged_data['Age']
merged_data['real_brain_age_gap']      = merged_data['SPARE_BA'] - merged_data['Age']
adaptive_merged['predicted_brain_age_gap'] = adaptive_merged['score'] - adaptive_merged['Age']
adaptive_merged['real_brain_age_gap']      = adaptive_merged['SPARE_BA'] - adaptive_merged['Age']

print(f"Total subjects     : {merged_data['PTID'].nunique()}")
print(f"Total observations : {len(merged_data)}")


# =============================================================================
# 2. DEFINE GROUPS
# =============================================================================
print("\n" + "="*60)
print("2. DEFINING DIAGNOSTIC GROUPS")
print("="*60)

print("Available diagnoses:")
print(merged_data['Diagnosis'].value_counts())


def assign_group(diagnosis):
    if diagnosis == 0:   return 'CN'
    elif diagnosis == 1: return 'MCI'
    elif diagnosis == 2: return 'AD'
    return 'Other'


baseline_diagnoses = (longitudinal_data.groupby('PTID')
                      .first()[['Diagnosis']].reset_index())
baseline_diagnoses.columns = ['PTID', 'Baseline_Diagnosis']
baseline_diagnoses['Group'] = baseline_diagnoses['Baseline_Diagnosis'].apply(assign_group)

merged_data     = merged_data.merge(
    baseline_diagnoses[['PTID', 'Group']], on='PTID', how='left')
adaptive_merged = adaptive_merged.merge(
    baseline_diagnoses[['PTID', 'Group']], on='PTID', how='left')

valid_groups             = ['CN', 'MCI', 'AD']
merged_data_filtered     = merged_data[merged_data['Group'].isin(valid_groups)].copy()
adaptive_merged_filtered = adaptive_merged[adaptive_merged['Group'].isin(valid_groups)].copy()

print("Baseline diagnosis distribution:")
for group, count in baseline_diagnoses['Group'].value_counts().items():
    print(f"  {group}: {count} subjects")
print("\nGroup distribution (filtered):")
for group, count in merged_data_filtered['Group'].value_counts().items():
    print(f"  {group}: {count} subjects")


# =============================================================================
# 3. COMPUTE LONGITUDINAL BAG SLOPES
# =============================================================================
print("\n" + "="*60)
print("3. COMPUTING LONGITUDINAL BAG SLOPES")
print("="*60)


def calculate_bag_slope(subject_data):
    if len(subject_data) < 2:
        return None, None, None, None
    try:
        subject_data = subject_data.sort_values('Age')
        nan_cols = ['Age', 'real_brain_age_gap', 'predicted_brain_age_gap']
        if subject_data[nan_cols].isna().any().any():
            return None, None, None, None
        if subject_data['Age'].var() == 0:
            return None, None, None, None
        if (subject_data['real_brain_age_gap'].var() == 0 or
                subject_data['predicted_brain_age_gap'].var() == 0):
            return None, None, None, None
        if (subject_data['Age'].max() - subject_data['Age'].min()) < 1.0:
            return None, None, None, None

        real_slope, _, _, real_p, _ = stats.linregress(
            subject_data['Age'], subject_data['real_brain_age_gap'])
        pred_slope, _, _, pred_p, _ = stats.linregress(
            subject_data['Age'], subject_data['predicted_brain_age_gap'])

        if not (np.isfinite(real_slope) and np.isfinite(pred_slope)):
            return None, None, None, None
        if abs(real_slope) > 10 or abs(pred_slope) > 10:
            return None, None, None, None

        return real_slope, pred_slope, real_p, pred_p
    except Exception:
        return None, None, None, None


print("Checking input data quality...")
print(f"  NaN in Age           : {merged_data_filtered['Age'].isna().sum()}")
print(f"  NaN in real BAG      : {merged_data_filtered['real_brain_age_gap'].isna().sum()}")
print(f"  NaN in predicted BAG : {merged_data_filtered['predicted_brain_age_gap'].isna().sum()}")

print("Calculating slopes...")
slope_results              = []
subjects_with_2plus        = 0
subjects_with_valid_slopes = 0

for subject_id in merged_data_filtered['PTID'].unique():
    sd = merged_data_filtered[merged_data_filtered['PTID'] == subject_id]
    if len(sd) >= 2:
        subjects_with_2plus += 1
        rs, ps, rp, pp = calculate_bag_slope(sd)
        if rs is not None:
            subjects_with_valid_slopes += 1
            bd = sd.iloc[0]
            slope_results.append({
                'PTID':               subject_id,
                'Group':              bd['Group'],
                'Baseline_Diagnosis': bd['Diagnosis'],
                'Baseline_Age':       bd['Age'],
                'N_Timepoints':       len(sd),
                'Time_Span':          sd['Age'].max() - sd['Age'].min(),
                'Real_BAG_Slope':     rs,
                'Pred_BAG_Slope':     ps,
                'Real_Slope_P':       rp,
                'Pred_Slope_P':       pp,
                'Real_Slope_Sig':     rp < 0.05,
                'Pred_Slope_Sig':     pp < 0.05,
            })

slopes_df = pd.DataFrame(slope_results)
print(f"  Total subjects           : {merged_data_filtered['PTID'].nunique()}")
print(f"  With 2+ timepoints       : {subjects_with_2plus}")
print(f"  With valid slopes        : {subjects_with_valid_slopes}")
print(f"  Excluded (age span < 1y) : {subjects_with_2plus - subjects_with_valid_slopes}")
for group in valid_groups:
    print(f"  {group}: {(slopes_df['Group'] == group).sum()} subjects")


# =============================================================================
# 4. WITHIN-GROUP SLOPE DISTRIBUTIONS
# =============================================================================
print("\n" + "="*60)
print("4. WITHIN-GROUP SLOPE DISTRIBUTIONS")
print("="*60)

group_stats = {}
for group in valid_groups:
    gd = slopes_df[slopes_df['Group'] == group]
    if len(gd) == 0:
        continue
    rs = gd['Real_BAG_Slope']
    ps = gd['Pred_BAG_Slope']
    group_stats[group] = {
        'N_Subjects':              len(gd),
        'Real_Slope_Mean':         rs.mean(),
        'Real_Slope_Median':       rs.median(),
        'Real_Slope_Std':          rs.std(),
        'Real_Slope_Var':          rs.var(),
        'Real_Slope_Positive_Pct': (rs > 0).mean() * 100,
        'Pred_Slope_Mean':         ps.mean(),
        'Pred_Slope_Median':       ps.median(),
        'Pred_Slope_Std':          ps.std(),
        'Pred_Slope_Var':          ps.var(),
        'Pred_Slope_Positive_Pct': (ps > 0).mean() * 100,
    }
    st = group_stats[group]
    print(f"\n{group} (N = {st['N_Subjects']}):")
    print(f"  Real  : mean={st['Real_Slope_Mean']:.4f}  "
          f"std={st['Real_Slope_Std']:.4f}  var={st['Real_Slope_Var']:.4f}")
    print(f"  Pred  : mean={st['Pred_Slope_Mean']:.4f}  "
          f"std={st['Pred_Slope_Std']:.4f}  var={st['Pred_Slope_Var']:.4f}")
    print(f"  Variance ratio (pred/real): "
          f"{st['Pred_Slope_Var']/st['Real_Slope_Var']:.3f}  "
          f"[< 1 confirms smoothing-induced variance reduction]")


# =============================================================================
# 5. VIOLIN PLOTS — DIAGNOSTIC GROUPS
# =============================================================================
print("\n" + "="*60)
print("5. CREATING VIOLIN PLOTS — DIAGNOSTIC GROUPS")
print("="*60)

colors     = ['#1f4e79', '#2E8B57', '#8B0000']
color_dict = {'CN': colors[0], 'MCI': colors[1], 'AD': colors[2]}

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('BAG Slope Distributions by Diagnostic Group',
             fontsize=16, fontweight='normal')

# Natural data range for each panel independently — no percentile clipping
all_real  = slopes_df['Real_BAG_Slope'].dropna().values
all_pred  = slopes_df['Pred_BAG_Slope'].dropna().values

for ax, slope_col, ylabel, title in [
    (axes[0], 'Real_BAG_Slope',
     'Real BAG vs Age Slope (years/year)',      'Real BAG Slope Distributions'),
    (axes[1], 'Pred_BAG_Slope',
     'Predicted BAG vs Age Slope (years/year)', 'Predicted BAG Slope Distributions'),
]:
    data, labels = [], []
    for i, group in enumerate(valid_groups):
        gd   = slopes_df[slopes_df['Group'] == group]
        vals = gd[slope_col].dropna().values
        if len(vals) > 0:
            data.append(vals)
            labels.append(f'{group}\n(n={len(gd)})')

    vp = ax.violinplot(data, positions=range(len(data)))
    for i, pc in enumerate(vp['bodies']):
        pc.set_facecolor(colors[i]); pc.set_alpha(0.7)
        pc.set_edgecolor('black');   pc.set_linewidth(1)
    bp = ax.boxplot(data, positions=range(len(data)),
                    widths=0.3, patch_artist=True)
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i]); box.set_alpha(0.9)

    ax.set_xlabel('Diagnostic Group', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='normal')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    # Independent y-limits per panel using natural data range + 10% padding
    if slope_col == 'Real_BAG_Slope':
        pad = 0.1 * (all_real.max() - all_real.min())
        ax.set_ylim(all_real.min() - pad, all_real.max() + pad)
    else:
        pad = 0.1 * (all_pred.max() - all_pred.min())
        ax.set_ylim(all_pred.min() - pad, all_pred.max() + pad)

# Remove top and right spines from both axes after all drawing is complete
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('nataging/SPAREBA_Experiment_bag_slope_distributions_by_group.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('nataging/SPAREBA_Experiment_bag_slope_distributions_by_group.svg',
            bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Diagnostic group violin plots saved")


# =============================================================================
# 6. BETWEEN-GROUP COMPARISONS
# =============================================================================
print("\n" + "="*60)
print("6. BETWEEN-GROUP COMPARISONS")
print("="*60)

real_slopes_by_group = {}
pred_slopes_by_group = {}
for group in valid_groups:
    gd = slopes_df[slopes_df['Group'] == group]
    if len(gd) > 0:
        real_slopes_by_group[group] = gd['Real_BAG_Slope'].values
        pred_slopes_by_group[group] = gd['Pred_BAG_Slope'].values

# One-way ANOVA with η² from sum of squares
for label, slopes_dict in [
    ("Real BAG Slopes",      real_slopes_by_group),
    ("Predicted BAG Slopes", pred_slopes_by_group),
]:
    groups_list = list(slopes_dict.values())
    if len(groups_list) < 2:
        continue
    try:
        f_stat, p_val = f_oneway(*groups_list)
    except Exception as e:
        print(f"ANOVA failed for {label}: {e}"); continue

    eta_sq, ss_between, ss_total = compute_eta_squared_from_ss(groups_list)
    total_n    = sum(len(g) for g in groups_list)
    df_between = len(groups_list) - 1
    df_within  = total_n - len(groups_list)
    print(f"\n{label} — One-way ANOVA:")
    print(f"  F({df_between}, {df_within}) = {f_stat:.4f},  p = {p_val:.6e}")
    print(f"  η² (from SS, unequal-n corrected) = {eta_sq:.4f}")
    print(f"  {'SIGNIFICANT' if p_val < 0.05 else 'NOT significant'}")

# Pairwise t-tests — all three pairs
pairwise_pairs = [('CN', 'MCI'), ('MCI', 'AD'), ('CN', 'AD')]
print(f"\n{'='*60}")
print("PAIRWISE T-TESTS (all three pairs)")
print(f"{'='*60}")
for label_a, label_b in pairwise_pairs:
    if label_a not in real_slopes_by_group or label_b not in real_slopes_by_group:
        continue
    print(f"\n  {label_a} vs {label_b}")
    for data_label, sd in [("Real BAG", real_slopes_by_group),
                            ("Predicted BAG", pred_slopes_by_group)]:
        t_stat, t_p = stats.ttest_ind(sd[label_a], sd[label_b])
        print(f"    {data_label}: t = {t_stat:.4f},  p = {t_p:.6e}  "
              f"({'*' if t_p < 0.05 else 'ns'})")

# Glass's Delta — all three pairs
glass_results = compute_all_pairwise_glass_delta(
    pred_slopes_by_group, real_slopes_by_group, pairs=pairwise_pairs)


# =============================================================================
# 7. SAVE RESULTS
# =============================================================================
print("\n" + "="*60)
print("7. SAVING RESULTS")
print("="*60)

slopes_df.to_csv(
    "nataging/SPAREBA_Experiment_individual_bag_slopes.csv", index=False)
pd.DataFrame(group_stats).T.to_csv(
    "nataging/SPAREBA_Experiment_group_slope_statistics.csv")

glass_summary = [
    {
        'Comparison':                 key,
        'N_A':                        res['n_a'],
        'N_B':                        res['n_b'],
        'Mean_Diff':                  res['mean_diff'],
        'Glass_Delta':                res['glass_delta'],
        'Cohens_d_pred_NOT_REPORTED': res['cohens_d_pred'],
        'Pooled_SD_Obs':              res['pooled_sd_obs'],
        'Pooled_SD_Pred':             res['pooled_sd_pred'],
    }
    for key, res in glass_results.items()
]
pd.DataFrame(glass_summary).to_csv(
    "nataging/SPAREBA_Experiment_glass_delta_summary.csv", index=False)
print("✓ Individual slopes, group statistics, and Glass's Delta summary saved")


# =============================================================================
# 8. SUMMARY REPORT
# =============================================================================
print("\n" + "="*60)
print("SUMMARY REPORT")
print("="*60)
for group in valid_groups:
    if group in group_stats:
        st = group_stats[group]
        print(f"\n{group} (N = {st['N_Subjects']}):")
        print(f"  Real BAG slope : {st['Real_Slope_Mean']:.4f} ± {st['Real_Slope_Std']:.4f}  "
              f"+ve: {st['Real_Slope_Positive_Pct']:.1f}%")
        print(f"  Pred BAG slope : {st['Pred_Slope_Mean']:.4f} ± {st['Pred_Slope_Std']:.4f}  "
              f"+ve: {st['Pred_Slope_Positive_Pct']:.1f}%")


# =============================================================================
# 8b. BAG vs Age correlation within each group
# =============================================================================
print("\n" + "="*60)
print("8b. BAG vs AGE CORRELATION WITHIN EACH GROUP")
print("="*60)

correlation_results = []
for group in valid_groups:
    gd = merged_data_filtered[merged_data_filtered['Group'] == group]
    if len(gd) <= 10:
        continue
    rc, rp = scipy.stats.pearsonr(gd['Age'], gd['real_brain_age_gap'])
    pc, pp = scipy.stats.pearsonr(gd['Age'], gd['predicted_brain_age_gap'])

    corr_diff_p = np.nan
    if abs(rc) < 0.999 and abs(pc) < 0.999:
        z_diff = np.arctanh(pc) - np.arctanh(rc)
        se     = np.sqrt(2 / (len(gd) - 3))
        corr_diff_p = 2 * (1 - scipy.stats.norm.cdf(abs(z_diff / se)))

    correlation_results.append({
        'Group': group, 'N_Observations': len(gd),
        'N_Subjects': gd['PTID'].nunique(),
        'Real_BAG_Age_Corr': rc, 'Real_BAG_Age_P': rp,
        'Pred_BAG_Age_Corr': pc, 'Pred_BAG_Age_P': pp,
        'Correlation_Diff': pc - rc, 'Correlation_Diff_P': corr_diff_p,
    })
    print(f"\n{group} (n={gd['PTID'].nunique()}):")
    print(f"  Real BAG–Age : r = {rc:.4f}  (p = {rp:.6e})")
    print(f"  Pred BAG–Age : r = {pc:.4f}  (p = {pp:.6e})")
    print(f"  Δr = {pc-rc:+.4f}  (Fisher z p = {corr_diff_p:.6f}  "
          f"{'significant' if corr_diff_p < 0.05 else 'not significant'})")

if correlation_results:
    pd.DataFrame(correlation_results).to_csv(
        "nataging/SPAREBA_Experiment_correlation_analysis.csv", index=False)
    print("\n✓ Correlation analysis saved")


# =============================================================================
# 9. NATURE AGING-STYLE VISUALISATIONS
# =============================================================================
print("\n" + "="*60)
print("9. NATURE AGING-STYLE VISUALISATIONS")
print("="*60)

plt.style.use('default')
mpl.rcParams.update({
    'font.size': 8, 'axes.linewidth': 0.5,
    'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 7, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
    'lines.linewidth': 1.0, 'lines.markersize': 4,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False,
})

# ── 9a. Nature-style violin/box plots ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
fig.suptitle('BAG Slope Distributions by Diagnostic Group',
             fontsize=10, fontweight='normal', y=0.95)

# Natural data range for each panel independently — no percentile clipping
all_real  = slopes_df['Real_BAG_Slope'].dropna().values
all_pred  = slopes_df['Pred_BAG_Slope'].dropna().values

for ax, slope_col, ylabel, title in [
    (axes[0], 'Real_BAG_Slope',  'Real BAG Slope (years/year)',      'Real BAG Slopes'),
    (axes[1], 'Pred_BAG_Slope', 'Predicted BAG Slope (years/year)', 'Predicted BAG Slopes'),
]:
    data, labels = [], []
    for i, group in enumerate(valid_groups):
        gd   = slopes_df[slopes_df['Group'] == group]
        vals = gd[slope_col].dropna().values
        if len(vals) > 0:
            data.append(vals)
            labels.append(group)
    vp = ax.violinplot(data, positions=range(len(data)))
    for i, pc in enumerate(vp['bodies']):
        pc.set_facecolor(colors[i]); pc.set_alpha(0.7)
        pc.set_edgecolor('black');   pc.set_linewidth(0.5)
    bp = ax.boxplot(data, positions=range(len(data)), widths=0.3, patch_artist=True)
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i]); box.set_alpha(0.9); box.set_linewidth(0.5)
    ax.set_xlabel('Diagnostic Group', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9, fontweight='normal')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=0.5)
    ax.tick_params(axis='both', which='major', width=0.5, length=3)
    # Independent y-limits per panel using natural data range + 10% padding
    if slope_col == 'Real_BAG_Slope':
        pad = 0.1 * (all_real.max() - all_real.min())
        ax.set_ylim(all_real.min() - pad, all_real.max() + pad)
    else:
        pad = 0.1 * (all_pred.max() - all_pred.min())
        ax.set_ylim(all_pred.min() - pad, all_pred.max() + pad)

# Remove top and right spines from both axes after all drawing is complete
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('nataging/SPAREBA_Experiment_nature_slope_distributions.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('nataging/SPAREBA_Experiment_nature_slope_distributions.svg',
            bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Nature-style slope distributions saved")

# ── 9b. BAG vs Age scatterplots ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(10.8, 7.2))
fig.suptitle('BAG vs Age Relationships by Diagnostic Group',
             fontsize=10, fontweight='normal', y=0.98)
for i, group in enumerate(valid_groups):
    gd = merged_data_filtered[merged_data_filtered['Group'] == group]
    if len(gd) == 0:
        continue
    for row, (bag_col, ylabel) in enumerate([
        ('real_brain_age_gap',      'Real BAG (years)'),
        ('predicted_brain_age_gap', 'Predicted BAG (years)'),
    ]):
        ax = axes[row, i]
        corr, pval = scipy.stats.pearsonr(gd['Age'], gd[bag_col])
        ax.scatter(gd['Age'], gd[bag_col], alpha=0.6, s=20,
                   color=color_dict[group], edgecolors='white', linewidth=0.3)
        z = np.polyfit(gd['Age'], gd[bag_col], 1)
        ax.plot(gd['Age'], np.poly1d(z)(gd['Age']),
                color='black', linewidth=1, alpha=0.8)
        label = 'Real' if row == 0 else 'Predicted'
        ax.set_title(f'{group} — {label} BAG\nr = {corr:.3f},  p = {pval:.3e}',
                     fontsize=8, fontweight='normal')
        ax.set_xlabel('Age (years)', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis='both', which='major', width=0.5, length=3, labelsize=7)
        ax.set_ylim(-35, 65)
plt.tight_layout()
plt.savefig('nataging/SPAREBA_Experiment_nature_bag_vs_age_scatter.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('nataging/SPAREBA_Experiment_nature_bag_vs_age_scatter.svg',
            bbox_inches='tight', facecolor='white')
plt.close()
print("✓ BAG vs Age scatterplots saved")

# ── 9c. Correlation comparison bar plot ───────────────────────────────────────
corr_data = []
for group in valid_groups:
    gd = merged_data_filtered[merged_data_filtered['Group'] == group]
    if len(gd) > 0:
        rc, _ = scipy.stats.pearsonr(gd['Age'], gd['real_brain_age_gap'])
        pc, _ = scipy.stats.pearsonr(gd['Age'], gd['predicted_brain_age_gap'])
        corr_data.append({'Group': group, 'Real_Corr': rc, 'Pred_Corr': pc})

if corr_data:
    corr_df = pd.DataFrame(corr_data)
    fig, ax = plt.subplots(figsize=(8, 5))
    x, w = np.arange(len(corr_df)), 0.35
    ax.bar(x - w/2, corr_df['Real_Corr'], w, label='Real BAG',
           color='#2c3e50', alpha=0.9)
    ax.bar(x + w/2, corr_df['Pred_Corr'], w, label='Predicted BAG',
           color='#2E8B57', alpha=0.9)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Diagnostic Group', fontsize=12)
    ax.set_ylabel('Correlation Coefficient (r)', fontsize=12)
    ax.set_title('BAG vs Age Correlation by Diagnostic Group',
                 fontsize=14, fontweight='normal')
    ax.set_xticks(x)
    ax.set_xticklabels(corr_df['Group'], fontsize=11)
    ax.legend(frameon=False, fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    for i, row in corr_df.iterrows():
        for val, offset in [(row['Real_Corr'], -w/2), (row['Pred_Corr'], w/2)]:
            ax.text(i + offset, val + (0.02 if val >= 0 else -0.02),
                    f'{val:.3f}', ha='center',
                    va='bottom' if val >= 0 else 'top', fontsize=9)
    plt.tight_layout()
    plt.savefig('nataging/SPAREBA_Experiment_nature_correlation_comparison.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('nataging/SPAREBA_Experiment_nature_correlation_comparison.svg',
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Correlation comparison bar plot saved")

# ── 9d. Mean slope barplot ────────────────────────────────────────────────────
gp, rm, pm, rs_std, ps_std = [], [], [], [], []
for group in valid_groups:
    gd = slopes_df[slopes_df['Group'] == group]
    if len(gd) > 0:
        gp.append(group)
        rm.append(gd['Real_BAG_Slope'].mean())
        pm.append(gd['Pred_BAG_Slope'].mean())
        rs_std.append(gd['Real_BAG_Slope'].std())
        ps_std.append(gd['Pred_BAG_Slope'].std())

if gp:
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.8))
    x, w = np.arange(len(gp)), 0.35
    ax.bar(x - w/2, rm, w, label='Real BAG',
           color='#2c3e50', alpha=0.9, yerr=rs_std, capsize=3)
    ax.bar(x + w/2, pm, w, label='Predicted BAG',
           color='#2E8B57', alpha=0.9, yerr=ps_std, capsize=3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Diagnostic Group', fontsize=9)
    ax.set_ylabel('Mean BAG Slope (years/year)', fontsize=9)
    ax.set_title('Mean BAG Slopes by Group', fontsize=9, fontweight='normal')
    ax.set_xticks(x); ax.set_xticklabels(gp, fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc='upper right')
    ax.tick_params(axis='both', which='major', width=0.5, length=3)
    for i, (r, p) in enumerate(zip(rm, pm)):
        ax.text(i - w/2, r + (0.01 if r >= 0 else -0.01), f'{r:.3f}',
                ha='center', va='bottom' if r >= 0 else 'top', fontsize=7)
        ax.text(i + w/2, p + (0.01 if p >= 0 else -0.01), f'{p:.3f}',
                ha='center', va='bottom' if p >= 0 else 'top', fontsize=7)
    plt.tight_layout()
    plt.savefig('nataging/SPAREBA_Experiment_nature_mean_slopes.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('nataging/SPAREBA_Experiment_nature_mean_slopes.svg',
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Mean slopes barplot saved")

# Statistical summary CSV
stat_summary = []
for group in valid_groups:
    gd = slopes_df[slopes_df['Group'] == group]
    if len(gd) > 0:
        rs = gd['Real_BAG_Slope']; ps = gd['Pred_BAG_Slope']
        stat_summary.append({
            'Group': group, 'N': len(gd),
            'Real_Mean': f"{rs.mean():.4f}", 'Real_Std': f"{rs.std():.4f}",
            'Real_Median': f"{rs.median():.4f}",
            'Real_Positive_Pct': f"{(rs > 0).mean()*100:.1f}%",
            'Pred_Mean': f"{ps.mean():.4f}", 'Pred_Std': f"{ps.std():.4f}",
            'Pred_Median': f"{ps.median():.4f}",
            'Pred_Positive_Pct': f"{(ps > 0).mean()*100:.1f}%",
        })
if stat_summary:
    summary_df = pd.DataFrame(stat_summary)
    summary_df.to_csv(
        "nataging/SPAREBA_Experiment_nature_statistical_summary.csv",
        index=False)
    print("\nSTATISTICAL SUMMARY TABLE")
    print(summary_df.to_string(index=False))


# =============================================================================
# 10. PROGRESSION STATUS ANALYSIS
# =============================================================================
print("\n" + "="*60)
print("10. PROGRESSION STATUS ANALYSIS")
print("="*60)

longitudinal_covariates = pd.read_csv(
    '../LongGPClustering/data1/'
    'longitudinal_covariates_subjectsamples_longclean_hmuse_convs_allstudies.csv')
longitudinal_covariates = longitudinal_covariates.rename(columns={'PTID': 'id'})
longitudinal_covariates['baseline_age'] = (
    longitudinal_covariates.groupby('id')['Age'].transform('first'))
longitudinal_covariates['sex'] = (
    longitudinal_covariates.groupby('id')['Sex'].transform('first'))


def get_progression_status(group):
    first_dx = group['Diagnosis'].iloc[0]
    last_dx  = group['Diagnosis'].iloc[-1]
    if first_dx == 0:
        if last_dx == 0: return 'Healthy Control'
        if last_dx == 1: return 'CN to MCI Progressor'
        if last_dx == 2: return 'CN to AD Progressor'
    elif first_dx == 1:
        if last_dx == 1: return 'MCI Stable'
        if last_dx == 2: return 'MCI Progressor'
    elif first_dx == 2:
        if last_dx == 2: return 'AD'
    return np.nan


longitudinal_covariates['progression_status'] = (
    longitudinal_covariates.groupby('id')
    .apply(get_progression_status)
    .reset_index(level=0, drop=True)
)
progression_status_map = (
    longitudinal_covariates.groupby('id')['progression_status']
    .first().to_dict()
)
slopes_df['progression_status'] = slopes_df['PTID'].map(progression_status_map)

print("Progression status distribution:")
print(slopes_df['progression_status'].value_counts())
print(f"Missing: {slopes_df['progression_status'].isna().sum()}")

progression_groups = [
    'CN', 'CN to MCI Progressor', 'CN to AD Progressor',
    'MCI Stable', 'MCI Progressor', 'AD',
]
slopes_df_progression = slopes_df[
    slopes_df['progression_status'].isin(progression_groups)].copy()
print(f"Subjects in defined progression groups: {len(slopes_df_progression)}")

progression_stats = {}
for group in progression_groups:
    gd = slopes_df_progression[
        slopes_df_progression['progression_status'] == group]
    if len(gd) == 0:
        continue
    rs = gd['Real_BAG_Slope']; ps = gd['Pred_BAG_Slope']
    progression_stats[group] = {
        'N_Subjects': len(gd),
        'Real_Slope_Mean': rs.mean(), 'Real_Slope_Std': rs.std(),
        'Real_Slope_Median': rs.median(),
        'Real_Slope_Positive_Pct': (rs > 0).mean() * 100,
        'Pred_Slope_Mean': ps.mean(), 'Pred_Slope_Std': ps.std(),
        'Pred_Slope_Median': ps.median(),
        'Pred_Slope_Positive_Pct': (ps > 0).mean() * 100,
    }
    print(f"\n{group} (N = {len(gd)}):")
    print(f"  Real : {rs.mean():.4f} ± {rs.std():.4f}")
    print(f"  Pred : {ps.mean():.4f} ± {ps.std():.4f}")


# =============================================================================
# 11. NATURE AGING-STYLE PROGRESSION VISUALISATIONS
# =============================================================================
print("\n" + "="*60)
print("11. NATURE AGING-STYLE PROGRESSION VISUALISATIONS")
print("="*60)

mpl.rcParams.update({
     'font.size': 8, 'axes.linewidth': 0.5,
    'axes.labelsize': 9, 'xtick.labelsize': 7, 'ytick.labelsize': 8,
    'legend.fontsize': 7, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.grid': False,
})

progression_colors = {
    'Healthy Control':      '#1f4e79',
    'CN to MCI Progressor': '#2E8B57',
    'CN to AD Progressor':  '#8B0000',
    'MCI Stable':           '#FF8C00',
    'MCI Progressor':       '#9932CC',
    'AD':                   '#DC143C',
}

# ── 11a. Violin plots by progression status ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('BAG Slope Distributions by Progression Status',
             fontsize=12, fontweight='normal', y=0.95)

# Natural data range for each panel independently — no percentile clipping
all_real_prog = slopes_df_progression['Real_BAG_Slope'].dropna().values
all_pred_prog = slopes_df_progression['Pred_BAG_Slope'].dropna().values

for ax, slope_col, ylabel, title in [
    (axes[0], 'Real_BAG_Slope',  'Real BAG Slope (years/year)',      'Real BAG Slopes'),
    (axes[1], 'Pred_BAG_Slope', 'Predicted BAG Slope (years/year)', 'Predicted BAG Slopes'),
]:
    data, labels = [], []
    for group in progression_groups:
        gd   = slopes_df_progression[
            slopes_df_progression['progression_status'] == group]
        vals = gd[slope_col].dropna().values
        if len(vals) > 0:
            data.append(vals)
            labels.append(f'{group}\n(n={len(gd)})')
    vp = ax.violinplot(data, positions=range(len(data)))
    for i, pc in enumerate(vp['bodies']):
        pc.set_facecolor(list(progression_colors.values())[i])
        pc.set_alpha(0.7); pc.set_edgecolor('black'); pc.set_linewidth(0.5)
    bp = ax.boxplot(data, positions=range(len(data)), widths=0.3, patch_artist=True)
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(list(progression_colors.values())[i])
        box.set_alpha(0.9); box.set_linewidth(0.5)
    ax.set_xlabel('Progression Status', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='normal')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=0.5)
    ax.tick_params(axis='both', which='major', width=0.5, length=3)
    # Independent y-limits per panel using natural data range + 10% padding
    if slope_col == 'Real_BAG_Slope':
        pad = 0.1 * (all_real_prog.max() - all_real_prog.min())
        ax.set_ylim(all_real_prog.min() - pad, all_real_prog.max() + pad)
    else:
        pad = 0.1 * (all_pred_prog.max() - all_pred_prog.min())
        ax.set_ylim(all_pred_prog.min() - pad, all_pred_prog.max() + pad)

# Remove top and right spines from both axes after all drawing is complete
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('nataging/SPAREBA_Experiment_progression_slope_distributions.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('nataging/SPAREBA_Experiment_progression_slope_distributions.svg',
            bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Progression slope distributions saved")

# ── 11b. Mean slope barplot by progression status ─────────────────────────────
gp, rm, pm, rs_std, ps_std = [], [], [], [], []
for group in progression_groups:
    gd = slopes_df_progression[slopes_df_progression['progression_status'] == group]
    if len(gd) > 0:
        gp.append(group)
        rm.append(gd['Real_BAG_Slope'].mean())
        pm.append(gd['Pred_BAG_Slope'].mean())
        rs_std.append(gd['Real_BAG_Slope'].std())
        ps_std.append(gd['Pred_BAG_Slope'].std())
if gp:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    x, w = np.arange(len(gp)), 0.35
    ax.bar(x - w/2, rm, w, label='Real BAG',
           color='#2c3e50', alpha=0.9, yerr=rs_std, capsize=3)
    ax.bar(x + w/2, pm, w, label='Predicted BAG',
           color='#2E8B57', alpha=0.9, yerr=ps_std, capsize=3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Progression Status', fontsize=10)
    ax.set_ylabel('Mean BAG Slope (years/year)', fontsize=10)
    ax.set_title('Mean BAG Slopes by Progression Status',
                 fontsize=12, fontweight='normal')
    ax.set_xticks(x)
    ax.set_xticklabels(gp, fontsize=9, rotation=45, ha='right')
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    ax.tick_params(axis='both', which='major', width=0.5, length=3)
    ax.grid(True, alpha=0.3, axis='y')
    for i, (r, p) in enumerate(zip(rm, pm)):
        ax.text(i - w/2, r + (0.01 if r >= 0 else -0.01), f'{r:.3f}',
                ha='center', va='bottom' if r >= 0 else 'top', fontsize=8)
        ax.text(i + w/2, p + (0.01 if p >= 0 else -0.01), f'{p:.3f}',
                ha='center', va='bottom' if p >= 0 else 'top', fontsize=8)
    plt.tight_layout()
    plt.savefig('nataging/SPAREBA_Experiment_progression_mean_slopes.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('nataging/SPAREBA_Experiment_progression_mean_slopes.svg',
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Progression mean slopes barplot saved")

# ── 11c. Real vs Predicted scatter ────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for group in progression_groups:
    gd = slopes_df_progression[slopes_df_progression['progression_status'] == group]
    if len(gd) > 0:
        ax.scatter(gd['Real_BAG_Slope'], gd['Pred_BAG_Slope'],
                   alpha=0.7, s=50, c=progression_colors[group],
                   label=f'{group} (n={len(gd)})',
                   edgecolors='white', linewidth=0.5)
min_val = min(slopes_df_progression['Real_BAG_Slope'].min(),
              slopes_df_progression['Pred_BAG_Slope'].min())
max_val = max(slopes_df_progression['Real_BAG_Slope'].max(),
              slopes_df_progression['Pred_BAG_Slope'].max())
ax.plot([min_val, max_val], [min_val, max_val],
        'k--', alpha=0.5, linewidth=1, label='Perfect prediction')
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
oc, op = scipy.stats.pearsonr(
    slopes_df_progression['Real_BAG_Slope'],
    slopes_df_progression['Pred_BAG_Slope'])
ax.set_xlabel('Real BAG Slope (years/year)', fontsize=10)
ax.set_ylabel('Predicted BAG Slope (years/year)', fontsize=10)
ax.set_title(f'Real vs Predicted BAG Slopes\nr = {oc:.3f},  p = {op:.3e}',
             fontsize=12, fontweight='normal')
ax.legend(frameon=False, fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig('nataging/SPAREBA_Experiment_progression_real_vs_pred_scatter.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('nataging/SPAREBA_Experiment_progression_real_vs_pred_scatter.svg',
            bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Real vs Predicted scatter saved")

# ── 11d. ANOVA across progression groups ──────────────────────────────────────
real_by_prog = {}
pred_by_prog = {}
for group in progression_groups:
    gd = slopes_df_progression[slopes_df_progression['progression_status'] == group]
    if len(gd) > 0:
        real_by_prog[group] = gd['Real_BAG_Slope'].values
        pred_by_prog[group] = gd['Pred_BAG_Slope'].values

for label, slopes_dict in [
    ("Real BAG Slopes",      real_by_prog),
    ("Predicted BAG Slopes", pred_by_prog),
]:
    groups_list = list(slopes_dict.values())
    if len(groups_list) < 2:
        continue
    try:
        f_stat, p_val = f_oneway(*groups_list)
    except Exception as e:
        print(f"ANOVA failed: {e}"); continue
    eta_sq, ss_between, ss_total = compute_eta_squared_from_ss(groups_list)
    total_n = sum(len(g) for g in groups_list)
    df_b = len(groups_list) - 1
    df_w = total_n - len(groups_list)
    print(f"\n{label} — ANOVA across progression groups:")
    print(f"  F({df_b}, {df_w}) = {f_stat:.4f},  p = {p_val:.6e}")
    print(f"  η² (from SS) = {eta_sq:.4f}")
    print(f"  {'SIGNIFICANT' if p_val < 0.05 else 'NOT significant'}")

# Glass's Delta — MCI Stable vs MCI Progressor
if 'MCI Stable' in pred_by_prog and 'MCI Progressor' in pred_by_prog:
    print(f"\n{'='*60}")
    print("GLASS'S DELTA — MCI Stable vs MCI Progressor")
    print(f"{'='*60}")
    compute_glass_delta(
        pred_by_prog['MCI Progressor'], pred_by_prog['MCI Stable'],
        real_by_prog['MCI Progressor'], real_by_prog['MCI Stable'],
        label_a='MCI Progressor', label_b='MCI Stable',
    )


# =============================================================================
# 12. SAVE PROGRESSION ANALYSIS RESULTS
# =============================================================================
print("\n" + "="*60)
print("12. SAVING PROGRESSION ANALYSIS RESULTS")
print("="*60)

slopes_df_progression.to_csv(
    "nataging/SPAREBA_Experiment_progression_individual_slopes.csv",
    index=False)
pd.DataFrame(progression_stats).T.to_csv(
    "nataging/SPAREBA_Experiment_progression_group_statistics.csv")
print("✓ Progression results saved")

print("\n" + "="*60)
print("PROGRESSION ANALYSIS SUMMARY REPORT")
print("="*60)
for group in progression_groups:
    if group in progression_stats:
        st = progression_stats[group]
        print(f"\n{group} (N = {st['N_Subjects']}):")
        print(f"  Real : {st['Real_Slope_Mean']:.4f} ± {st['Real_Slope_Std']:.4f}  "
              f"+ve: {st['Real_Slope_Positive_Pct']:.1f}%")
        print(f"  Pred : {st['Pred_Slope_Mean']:.4f} ± {st['Pred_Slope_Std']:.4f}  "
              f"+ve: {st['Pred_Slope_Positive_Pct']:.1f}%")

print("\n" + "="*80)
print("LONGITUDINAL BAG SLOPE ANALYSIS COMPLETED")
print("="*80)