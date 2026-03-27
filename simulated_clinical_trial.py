#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Strict MCI/mild AD clinical trial simulation (predicted-slope-only)

- Screening:
    Age ∈ [55, 86]
    CDR_Global ∈ {0.5, 1.0}
    MMSE ≥ 20

- Enrichment:
    Top 30% fastest predicted hippocampal atrophy (DKGP or RNN-AD)

- Biomarker window:
    Hippocampal slope = predicted ΔVolume per year over 36 ± 12 months

- Endpoint window:
    MMSE slope = ΔMMSE over 36 ± 12 months

- Treatment effect logic:
    Applied only to true progressors (fast MMSE decliners, median split),
    not uniformly to all enriched subjects.

- Outputs:
    ROC curves with 95% CI
    Power curves with 95% CI
    Required-N curves with 95% CI
    Composite (Nature-Aging-style) figure
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils import resample
from statsmodels.stats.power import TTestIndPower

import warnings
warnings.filterwarnings("ignore")

# =====================================================================
# CONFIG
# =====================================================================

SUBJECTS_PER_ARM = 400
ENRICHMENT_PCT = 30
EFFECT_FACTORS = [0.85, 0.83, 0.81, 0.79, 0.77, 0.75]   # 15–25% treatment benefit

HIPPO_TARGET_MONTHS = 36
HIPPO_TOLERANCE = 12

MMSE_TARGET_MONTHS = 36
MMSE_TOLERANCE = 12

N_BOOTSTRAP_ROC = 1000
N_BOOTSTRAP_POWER = 800
N_BOOTSTRAP_N = 400

COV_PATH = "longitudinal_covariates_subjectsamples_longclean_hmuse_convs_allstudies.csv"
HIPPO_PATH = "../LongGPRegressionBaseline/manuscript1/OldHarmonizedMUSEROIs.csv"
RNN_PATH = "rnn_volume47_results_5folds/all_folds_predictions.csv"
MMSE_PATH = "mmse_clinical_trial_data.csv"

OUTPUT_PREFIX = "strict_trial_predicted36m_claude"

analysis = TTestIndPower()

# =====================================================================
# FIXED-HORIZON SLOPE FUNCTIONS
# =====================================================================

def compute_mmse_delta_36m(mmse_df):
    """Compute ΔMMSE closest to 36±12 months."""
    df = mmse_df[["PTID", "Time", "MMSE_nearest_2.0"]].dropna().copy()
    df = df.sort_values(["PTID", "Time"])

    base = (
        df.groupby("PTID").first()
        .rename(columns={"MMSE_nearest_2.0": "MMSE_baseline"})
    )[["MMSE_baseline"]]

    df["abs_dist"] = (df["Time"] - MMSE_TARGET_MONTHS).abs()
    idx = df.groupby("PTID")["abs_dist"].idxmin()
    follow = df.loc[idx, ["PTID", "Time", "MMSE_nearest_2.0", "abs_dist"]]
    follow = follow[follow["abs_dist"] <= MMSE_TOLERANCE]
    follow = follow.rename(columns={"MMSE_nearest_2.0": "MMSE_followup"})

    out = base.merge(
        follow[["PTID", "MMSE_followup"]],
        left_index=True, right_on="PTID", how="inner"
    )
    out["MMSE_delta_36m"] = out["MMSE_followup"] - out["MMSE_baseline"]
    return out.reset_index(drop=False)


def compute_fixed_change(df, value_col,
                         id_col="id", time_col="time",
                         target_months=36, tolerance=12):
    """Hippocampal Δ (per year) over 36±12 months."""
    df = df[[id_col, time_col, value_col]].dropna().copy()
    df = df.sort_values([id_col, time_col])

    base = (
        df.groupby(id_col).first()
        .rename(columns={value_col: f"{value_col}_base"})
    )[[f"{value_col}_base"]]

    df["abs_dist"] = (df[time_col] - target_months).abs()
    idx = df.groupby(id_col)["abs_dist"].idxmin()
    follow = df.loc[idx, [id_col, time_col, value_col, "abs_dist"]]
    follow = follow[follow["abs_dist"] <= tolerance]

    out = follow.merge(base, left_on=id_col, right_index=True, how="inner")
    out["delta_value"] = out[value_col] - out[f"{value_col}_base"]
    out["delta_per_year"] = out["delta_value"] / (target_months / 12.0)

    return out.rename(columns={id_col: "PTID"})[["PTID", "delta_per_year"]]


# =====================================================================
# SCREENING
# =====================================================================

def screen_strict_trial(baseline):
    """
    Apply standard inclusion criteria.
    Requires columns: Age, CDR_Global, MMSE.
    """
    for col in ["Age", "CDR_Global", "MMSE"]:
        if col not in baseline.columns:
            raise ValueError(
                f"Column '{col}' not found in baseline. "
                f"Available: {baseline.columns.tolist()}"
            )
    age_mask  = (baseline["Age"] >= 55) & (baseline["Age"] <= 86)
    cdr_mask  = baseline["CDR_Global"].isin([0.5, 1.0])
    mmse_mask = baseline["MMSE"] >= 20
    mask = age_mask & cdr_mask & mmse_mask
    n_screened = mask.sum()
    print(f"  Screening: {n_screened}/{len(baseline)} subjects pass inclusion criteria")
    return baseline.loc[mask, "PTID"].values


# =====================================================================
# BOOTSTRAPPING
# =====================================================================

def bootstrap_roc(labels, scores, n_boot=1000):
    """Bootstrap ROC curve with 95% CI bands and AUC CI."""
    labels = np.array(labels)
    scores = np.array(scores)
    n = len(labels)
    tprs, aucs = [], []
    grid = np.linspace(0, 1, 200)

    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        yb, sb = labels[idx], scores[idx]
        try:
            fpr, tpr, _ = roc_curve(yb, sb)
            auc = roc_auc_score(yb, sb)
        except Exception:
            continue
        tprs.append(np.interp(grid, fpr, tpr))
        aucs.append(auc)

    tprs = np.array(tprs)
    return (
        grid,
        tprs.mean(axis=0),
        np.percentile(tprs, 2.5,  axis=0),
        np.percentile(tprs, 97.5, axis=0),
        np.mean(aucs),
        np.percentile(aucs, 2.5),
        np.percentile(aucs, 97.5),
    )


def _cohens_d_progressor_only(mmse_values, fast_mask, eff):
    """
    Compute Cohen's d where the treatment effect is applied only to
    true progressors (fast MMSE decliners), not to all enriched subjects.

    Parameters
    ----------
    mmse_values : array-like  MMSE delta for the enriched cohort
    fast_mask   : boolean array  True = fast decliner (progressor)
    eff         : float  treatment effect factor (e.g. 0.80 = 20% slowing)

    Returns
    -------
    d : float  Cohen's d
    """
    mmse_values = np.array(mmse_values)
    fast_mask   = np.array(fast_mask, dtype=bool)

    mu_c   = mmse_values.mean()
    sigma_c = mmse_values.std(ddof=1)

    if sigma_c == 0 or fast_mask.sum() == 0:
        return 0.0

    # Treatment slows decline only in true progressors
    treated = mmse_values.copy()
    treated[fast_mask] = mmse_values[fast_mask] * eff

    mu_t = treated.mean()
    return abs(mu_c - mu_t) / sigma_c


def bootstrap_power(mmse_values, fast_mask, eff, n_per_arm,
                    n_boot=N_BOOTSTRAP_POWER):
    """
    Bootstrap statistical power.
    Treatment effect applied only to progressors.
    """
    mmse_values = np.array(mmse_values)
    fast_mask   = np.array(fast_mask, dtype=bool)
    n = len(mmse_values)
    outs = []

    for _ in range(n_boot):
        idx  = np.random.choice(n, n, replace=True)
        yb   = mmse_values[idx]
        mb   = fast_mask[idx]
        d    = _cohens_d_progressor_only(yb, mb, eff)
        outs.append(analysis.power(effect_size=d, nobs1=n_per_arm, alpha=0.05))

    return np.mean(outs), np.percentile(outs, 2.5), np.percentile(outs, 97.5)


def bootstrap_required_n(mmse_values, fast_mask, eff,
                          n_boot=N_BOOTSTRAP_N):
    """
    Bootstrap required sample size (80% power).
    Treatment effect applied only to progressors.
    """
    mmse_values = np.array(mmse_values)
    fast_mask   = np.array(fast_mask, dtype=bool)
    n = len(mmse_values)
    outs = []

    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        yb  = mmse_values[idx]
        mb  = fast_mask[idx]
        d   = _cohens_d_progressor_only(yb, mb, eff)
        if d > 0:
            try:
                outs.append(
                    analysis.solve_power(effect_size=d, power=0.8, alpha=0.05)
                )
            except Exception:
                pass

    if not outs:
        return np.nan, np.nan, np.nan
    return np.mean(outs), np.percentile(outs, 2.5), np.percentile(outs, 97.5)


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def main():

    print("\n===== LOADING FILES =====")
    cov  = pd.read_csv(COV_PATH)
    hip  = pd.read_csv(HIPPO_PATH)
    rnn  = pd.read_csv(RNN_PATH)
    mmse = pd.read_csv(MMSE_PATH)

    # ------------------------------------------------------------------
    # BASELINE COVARIATES (one row per subject)
    # ------------------------------------------------------------------
    cov  = cov.sort_values(["PTID", "Time"])
    base = cov.groupby("PTID").first().reset_index()
    base = base.rename(columns={"MMSE_nearest_2.0": "MMSE"})

    # ------------------------------------------------------------------
    # FIXED-HORIZON MMSE DELTA
    # ------------------------------------------------------------------
    mmse_delta = compute_mmse_delta_36m(mmse)

    # ------------------------------------------------------------------
    # FIXED-HORIZON PREDICTED HIPPO SLOPE (DKGP)
    # ------------------------------------------------------------------
    hip_pred = hip[["id", "time", "score_H_MUSE_Volume_47"]].rename(
        columns={"id": "PTID", "score_H_MUSE_Volume_47": "pred"}
    )
    slope_pred = compute_fixed_change(
        hip_pred, "pred",
        id_col="PTID", time_col="time",
        target_months=HIPPO_TARGET_MONTHS,
        tolerance=HIPPO_TOLERANCE,
    ).rename(columns={"delta_per_year": "hippo_pred_slope"})

    # ------------------------------------------------------------------
    # FIXED-HORIZON PREDICTED HIPPO SLOPE (RNN-AD)
    # ------------------------------------------------------------------
    rnn2 = rnn.rename(columns={"subject_id": "PTID", "predicted": "rnn_pred"})
    slope_rnn = compute_fixed_change(
        rnn2[["PTID", "time", "rnn_pred"]],
        "rnn_pred",
        id_col="PTID", time_col="time",
        target_months=HIPPO_TARGET_MONTHS,
        tolerance=HIPPO_TOLERANCE,
    ).rename(columns={"delta_per_year": "rnn_slope"})

    print("Slope shapes — DKGP:", slope_pred.shape, "| RNN-AD:", slope_rnn.shape)

    # ------------------------------------------------------------------
    # MERGE ALL (one row per subject)
    # ------------------------------------------------------------------
    df = base.merge(slope_pred,  on="PTID", how="inner")
    df = df.merge(slope_rnn,     on="PTID", how="inner")
    df = df.merge(mmse_delta[["PTID", "MMSE_delta_36m"]], on="PTID", how="inner")

    print(f"Merged DF shape: {df.shape}  (should be one row per subject)")
    assert df["PTID"].nunique() == len(df), \
        "Duplicate PTIDs detected — check merge keys."

    # ------------------------------------------------------------------
    # APPLY TRIAL SCREENING
    # ------------------------------------------------------------------
    trial_ids = screen_strict_trial(df)
    trial_df  = df[df["PTID"].isin(trial_ids)].copy().reset_index(drop=True)
    print(f"Trial cohort after screening: N={len(trial_df)}")

    # ------------------------------------------------------------------
    # ENRICHMENT SCORES (fast decline → large negative slope → high score)
    # ------------------------------------------------------------------
    trial_df["DKGP_score"] = -trial_df["hippo_pred_slope"]
    trial_df["RNN_score"]  = -trial_df["rnn_slope"]

    # ------------------------------------------------------------------
    # ROC LABELS: fast MMSE decliners (median split)
    # These are the "true progressors" used in both ROC and power analysis
    # ------------------------------------------------------------------
    median_mmse = trial_df["MMSE_delta_36m"].median()
    trial_df["is_fast_decliner"] = (
        trial_df["MMSE_delta_36m"] < median_mmse
    ).astype(int)

    labels = trial_df["is_fast_decliner"].values

    print(f"Fast decliners: {labels.sum()} / {len(labels)} "
          f"({labels.mean()*100:.1f}%)")

    # =================================================================
    # ROC ANALYSIS
    # Computed once and reused in the composite figure
    # =================================================================
    print("\n===== ROC ANALYSIS =====")

    roc_results = {}   # model -> (grid, mtpr, lo, hi, auc_m, auc_lo, auc_hi)
    roc_rows    = []

    for model_name, score_col in [("DKGP", "DKGP_score"), ("RNN-AD", "RNN_score")]:
        scores = trial_df[score_col].values
        result = bootstrap_roc(labels, scores, n_boot=N_BOOTSTRAP_ROC)
        roc_results[model_name] = result
        _, _, _, _, auc_m, auc_lo, auc_hi = result
        roc_rows.append([model_name, auc_m, auc_lo, auc_hi])
        print(f"  {model_name}: AUC = {auc_m:.3f} [{auc_lo:.3f}, {auc_hi:.3f}]")

    roc_df = pd.DataFrame(roc_rows, columns=["Model", "AUC", "Lower", "Upper"])
    roc_df.to_csv(f"{OUTPUT_PREFIX}_roc_auc.csv", index=False)

    # =================================================================
    # POWER & SAMPLE SIZE
    # Treatment effect applied only to true progressors (fast decliners)
    # =================================================================
    print("\n===== POWER & SAMPLE SIZE ANALYSIS =====")

    power_rows = []
    n_rows     = []

    # Models to evaluate: enriched DKGP, enriched RNN-AD, no enrichment
    model_configs = [
        ("DKGP",           "DKGP_score", True),
        ("RNN-AD",         "RNN_score",  True),
        ("No Enrichment",  "DKGP_score", False),  # score col irrelevant when no threshold
    ]

    for model_name, score_col, do_enrich in model_configs:

        if do_enrich:
            scores = trial_df[score_col].values
            thr    = np.percentile(scores, 100 - ENRICHMENT_PCT)
            enriched_df = trial_df[scores >= thr].copy()
        else:
            enriched_df = trial_df.copy()

        Y_en   = enriched_df["MMSE_delta_36m"].values
        fast_m = enriched_df["is_fast_decliner"].values.astype(bool)

        print(f"\n  {model_name}: enriched N = {len(enriched_df)} "
              f"| progressors = {fast_m.sum()} ({fast_m.mean()*100:.1f}%)")

        for eff in EFFECT_FACTORS:
            eff_pct = int((1 - eff) * 100)

            p_mean, p_lo, p_hi = bootstrap_power(
                Y_en, fast_m, eff, SUBJECTS_PER_ARM
            )
            power_rows.append([model_name, eff_pct, p_mean, p_lo, p_hi])

            n_mean, n_lo, n_hi = bootstrap_required_n(Y_en, fast_m, eff)
            n_rows.append([model_name, eff_pct, n_mean, n_lo, n_hi])

            print(f"    Effect {eff_pct}%: Power={p_mean:.3f} [{p_lo:.3f}, {p_hi:.3f}] "
                  f"| Required N={n_mean:.0f} [{n_lo:.0f}, {n_hi:.0f}]")

    power_df = pd.DataFrame(
        power_rows,
        columns=["Model", "EffectPct", "Power_mean", "Power_lo", "Power_hi"]
    )
    power_df.to_csv(f"{OUTPUT_PREFIX}_power.csv", index=False)

    n_df = pd.DataFrame(
        n_rows,
        columns=["Model", "EffectPct", "N_mean", "N_lo", "N_hi"]
    )
    n_df.to_csv(f"{OUTPUT_PREFIX}_sample_size.csv", index=False)

    # =================================================================
    # COMPOSITE FIGURE (NATURE AGING STYLE)
    # =================================================================
    print("\n===== GENERATING COMPOSITE FIGURE =====")

    plt.rcParams.update({
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.size":         16,
        "axes.linewidth":    1.2,
    })

    fig, axes = plt.subplots(1, 3, figsize=(26, 7.5))

    colors = {
        "DKGP":           "#1f77b4",   # blue
        "RNN-AD":         "#d62728",   # red
        "No Enrichment":  "#2ca02c",   # green
    }

    # ------------------------------------------------------------------
    # A. ROC Curve Panel (reuse pre-computed bootstrap results)
    # ------------------------------------------------------------------
    ax = axes[0]

    for model_name in ["DKGP", "RNN-AD"]:
        grid, mtpr, lo, hi, auc_m, auc_lo, auc_hi = roc_results[model_name]
        ax.plot(grid, mtpr, lw=3, color=colors[model_name],
                label=f"{model_name} (AUC: {auc_m:.3f} [{auc_lo:.3f}, {auc_hi:.3f}])")
        ax.fill_between(grid, lo, hi, color=colors[model_name], alpha=0.20)

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=18)
    ax.set_ylabel("True Positive Rate",  fontsize=18)
    ax.set_title("ROC Curves: Hippocampal Enrichment Strategies",
                 fontsize=18, pad=12)
    ax.legend(frameon=False, fontsize=14, loc="lower right")
    ax.text(0.05, 0.95, "a", transform=ax.transAxes,
            fontsize=22, fontweight="bold", va="top")

    # ------------------------------------------------------------------
    # B. Power vs Treatment Effect
    # ------------------------------------------------------------------
    ax = axes[1]

    for model_name in ["DKGP", "RNN-AD", "No Enrichment"]:
        d = power_df[power_df["Model"] == model_name]
        label = f"{model_name} ± 95% CI"
        ax.plot(d["EffectPct"], d["Power_mean"], lw=3, marker="o",
                markersize=6, color=colors[model_name], label=label)
        ax.fill_between(d["EffectPct"], d["Power_lo"], d["Power_hi"],
                        color=colors[model_name], alpha=0.20)

    ax.axhline(0.8, color="gray", ls="--", lw=1, label="0.80 threshold")
    ax.set_xlabel("Treatment Effect (%)", fontsize=18)
    ax.set_ylabel("Statistical Power",    fontsize=18)
    ax.set_title(f"Power vs. Treatment Effect Size\n"
                 f"(Fixed 30% Enrichment, N={SUBJECTS_PER_ARM})",
                 fontsize=18, pad=12)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=14, loc="upper left")
    ax.text(0.05, 0.95, "b", transform=ax.transAxes,
            fontsize=22, fontweight="bold", va="top")

    # ------------------------------------------------------------------
    # C. Required Sample Size vs Treatment Effect
    # ------------------------------------------------------------------
    ax = axes[2]

    for model_name in ["DKGP", "RNN-AD", "No Enrichment"]:
        d = n_df[n_df["Model"] == model_name]
        ax.plot(d["EffectPct"], d["N_mean"], lw=3, marker="o",
                markersize=6, color=colors[model_name],
                label=f"{model_name} ± 95% CI")
        ax.fill_between(d["EffectPct"], d["N_lo"], d["N_hi"],
                        color=colors[model_name], alpha=0.20)

    ax.set_xlabel("Treatment Effect (%)",  fontsize=18)
    ax.set_ylabel("Required N (per group)", fontsize=18)
    ax.set_title("Required Sample Size vs. Treatment Effect",
                 fontsize=18, pad=12)
    ax.legend(frameon=False, fontsize=14, loc="upper right")
    ax.text(0.05, 0.95, "c", transform=ax.transAxes,
            fontsize=22, fontweight="bold", va="top")

    # ------------------------------------------------------------------
    # Layout and Save
    # ------------------------------------------------------------------
    plt.tight_layout()

    plt.savefig(f"{OUTPUT_PREFIX}_composite_fig.png", dpi=450, bbox_inches="tight")
    plt.savefig(f"{OUTPUT_PREFIX}_composite_fig.svg", dpi=450, bbox_inches="tight")
    plt.savefig(f"{OUTPUT_PREFIX}_composite_fig.pdf", dpi=450, bbox_inches="tight")
    plt.close()

    print("\n===== DONE. All outputs saved. =====")
    print(f"\nSummary:")
    print(f"  Trial cohort: N={len(trial_df)}")
    print(f"  Fast decliners: {labels.sum()} ({labels.mean()*100:.1f}%)")
    print(f"\nAUC Results:")
    print(roc_df.to_string(index=False))
    print(f"\nPower Results:")
    print(power_df.to_string(index=False))
    print(f"\nSample Size Results:")
    print(n_df.to_string(index=False))


# =====================================================================
# RUN
# =====================================================================
if __name__ == "__main__":
    main()