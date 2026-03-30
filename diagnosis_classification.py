# ==============================================================================
# Experiment: Alzheimer's Disease Diagnosis Classification
#             Using Real vs. Predicted Brain ROI Volumes
# ==============================================================================
#
# OVERVIEW
# --------
# This script evaluates whether GP-predicted longitudinal brain ROI volumes
# (from the singletask DKGP model) are diagnostically equivalent to real
# (harmonized MUSE) ROI volumes for classifying Alzheimer's Disease (AD),
# Mild Cognitive Impairment (MCI), and Cognitively Normal (CN) subjects.
#
# MOTIVATION
# ----------
# The DKGP model predicts future brain trajectories from a subject's observed
# history. A key validation question is: do predicted volumes preserve enough
# diagnostic signal to support downstream clinical classification? If a
# classifier trained and tested on predicted data performs comparably to one
# trained and tested on real data, this supports the clinical utility of the
# predicted trajectories.
#
# EXPERIMENTAL DESIGN
# -------------------
# Four train/test configurations are evaluated systematically:
#
#   1. Real  → Real  : trained on real volumes,      tested on real volumes
#   2. Real  → Pred  : trained on real volumes,      tested on predicted volumes
#   3. Pred  → Real  : trained on predicted volumes, tested on real volumes
#   4. Pred  → Pred  : trained on predicted volumes, tested on predicted volumes
#
# Configurations 2 and 3 (cross-domain) reveal whether the real and predicted
# feature spaces are interchangeable. If cross-domain performance is close to
# the in-domain baselines (configs 1 and 4), this indicates high fidelity of
# the predicted volumes.
#
# DATA
# ----
# - Input ROI features  : 145 harmonized MUSE volumetric brain regions
# - Subjects            : multi-study longitudinal cohort (ADNI + BLSA)
# - Sampling            : one non-baseline timepoint per subject (random,
#                         seed=42), ensuring real and predicted features are
#                         always drawn from the same visit
# - Diagnosis labels    : CN=0, MCI=1, AD=2
#
# CLASSIFIER
# ----------
# Support Vector Machine (SVM) with RBF kernel and balanced class weighting
# to handle class imbalance. A fresh SVM instance is created at each CV fold
# to prevent any state leakage between evaluations.
#
# EVALUATION
# ----------
# - Primary metric      : weighted F1-score (also reports precision and recall)
# - Cross-validation    : 5-fold stratified (preserves class proportions)
# - Confidence interval : 95% CI via t-distribution (df = 4)
# - Significance test   : paired t-test across folds (real vs. predicted data)
#
# OUTPUTS
# -------
# - ./nataging/diagnosis_classification_results.{png,pdf,svg}
#       Combined bar chart: CV mean ± 95% CI for all 4 configurations × 3
#       metrics, with paired t-test significance brackets
# - ./nataging/svm_performance_comparison.csv
#       Single-split (80/20) performance table
# - ./nataging/svm_cross_validation_results.csv
#       Per-fold CV scores for all configurations
# - ./nataging/svm_{precision,recall,f1}_cv_boxplot.{png,pdf}
#       Supplementary per-metric CV boxplots
# ==============================================================================

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.svm import SVC
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import seaborn as sns

# ==============================================================================
# LOAD DATA
# ==============================================================================
print("=" * 80)
print("ALZHEIMER'S DISEASE DIAGNOSIS CLASSIFICATION ANALYSIS")
print("=" * 80)

roi_df = pd.read_csv("./manuscript1/HarmonizedROIVolumes.csv")
covariate_df = pd.read_csv(
    "../LongGPClustering/data1/"
    "longitudinal_covariates_subjectsamples_longclean_hmuse_convs_allstudies.csv"
)

# ==============================================================================
# PREPROCESS AND MERGE
# ==============================================================================
roi_df = roi_df.rename(columns={"id": "PTID", "time": "Time"})
roi_df["PTID"] = roi_df["PTID"].astype(str)
roi_df["Time"] = roi_df["Time"].astype(int)
covariate_df["PTID"] = covariate_df["PTID"].astype(str)
covariate_df["Time"] = covariate_df["Time"].astype(int)

df = pd.merge(
    roi_df,
    covariate_df[["PTID", "Time", "Diagnosis"]],
    on=["PTID", "Time"],
    how="left",
)

# Strip "y_" prefix from real volume columns
real_cols = [c for c in df.columns if c.startswith("y_H_MUSE_Volume_")]
df = df.rename(columns={c: c.replace("y_", "") for c in real_cols})

# Predicted columns already carry the "score_" prefix — keep as-is
# (the original script had a no-op rename here; intentionally removed)

df = df.dropna(subset=["Diagnosis"])
df = df[df["Diagnosis"].isin([0, 1, 2])]

label_map = {
    0: "CN  (Cognitively Normal)",
    1: "MCI (Mild Cognitive Impairment)",
    2: "AD  (Alzheimer's Disease)",
}

print(f"\nDATASET OVERVIEW:")
print(f"  Total samples   : {len(df):,}")
print(f"  Unique subjects : {df['PTID'].nunique():,}")
print(f"  ROI features    : {len([c for c in df.columns if c.startswith('H_MUSE_Volume_')])}")
print(f"\nDIAGNOSIS DISTRIBUTION (full dataset):")
for diag, cnt in df["Diagnosis"].value_counts().items():
    print(f"  {label_map[diag]}: {cnt:,}  ({cnt/len(df):.1%})")

# ==============================================================================
# SAMPLE ONE NON-BASELINE TIMEPOINT PER SUBJECT
# ==============================================================================
df = df[df["Time"] != 0]
sampled_df = (
    df.groupby("PTID")
    .apply(lambda x: x.sample(1, random_state=42))
    .reset_index(drop=True)
)

print(f"\nAFTER SAMPLING (one non-baseline timepoint per subject):")
print(f"  Samples         : {len(sampled_df):,}")
print(f"  Unique subjects : {sampled_df['PTID'].nunique():,}")
print(f"\nFINAL DIAGNOSIS DISTRIBUTION:")
for diag, cnt in sampled_df["Diagnosis"].value_counts().items():
    print(f"  {label_map[diag]}: {cnt:,}  ({cnt/len(sampled_df):.1%})")

# ==============================================================================
# FEATURE MATRICES
# ==============================================================================
hmuse_cols = [c for c in sampled_df.columns if c.startswith("H_MUSE_Volume_")]
score_cols = ["score_" + c for c in hmuse_cols]

X_real = sampled_df[hmuse_cols].values
X_pred = sampled_df[score_cols].values
y = sampled_df["Diagnosis"].values

# ==============================================================================
# TRAIN / TEST SPLIT — shared index array keeps real & predicted rows aligned
# ==============================================================================
indices = np.arange(len(y))
train_idx, test_idx = train_test_split(
    indices, test_size=0.2, stratify=y, random_state=42
)

X_real_train, X_real_test = X_real[train_idx], X_real[test_idx]
X_pred_train, X_pred_test = X_pred[train_idx], X_pred[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"\nTRAIN / TEST SPLIT (80 % / 20 %):")
print(f"  Training : {len(train_idx):,}")
print(f"  Test     : {len(test_idx):,}")
print(f"\nTRAINING SET DISTRIBUTION:")
for diag, cnt in pd.Series(y_train).value_counts().items():
    print(f"  {label_map[diag]}: {cnt:,}  ({cnt/len(y_train):.1%})")
print(f"\nTEST SET DISTRIBUTION:")
for diag, cnt in pd.Series(y_test).value_counts().items():
    print(f"  {label_map[diag]}: {cnt:,}  ({cnt/len(y_test):.1%})")

# ==============================================================================
# HELPERS
# ==============================================================================
def get_metrics(y_true, y_pred):
    return {
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1":        f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Confusion Matrix":      confusion_matrix(y_true, y_pred),
        "Classification Report": classification_report(y_true, y_pred, zero_division=0),
    }


def make_svm():
    """Fresh SVM — prevents state leakage between fits."""
    return SVC(kernel="rbf", class_weight="balanced", random_state=42)

# ==============================================================================
# SINGLE-SPLIT EVALUATION
# ==============================================================================
print("\n" + "=" * 80)
print("SINGLE-SPLIT MODEL EVALUATION")
print("=" * 80)

real_model = make_svm().fit(X_real_train, y_train)
pred_model = make_svm().fit(X_pred_train, y_train)

single_results = {
    "Real Model (Real Data)":      get_metrics(y_test, real_model.predict(X_real_test)),
    "Real Model (Predicted Data)": get_metrics(y_test, real_model.predict(X_pred_test)),
    "Pred Model (Real Data)":      get_metrics(y_test, pred_model.predict(X_real_test)),
    "Pred Model (Predicted Data)": get_metrics(y_test, pred_model.predict(X_pred_test)),
}

rows = []
for model_type in ["Real Model", "Pred Model"]:
    for test_data in ["Real Data", "Predicted Data"]:
        key = f"{model_type} ({test_data})"
        r = single_results[key]
        rows.append({
            "Model Type": model_type,
            "Test Data":  test_data,
            "Precision":  r["Precision"],
            "Recall":     r["Recall"],
            "F1":         r["F1"],
        })
comparison_df = pd.DataFrame(rows)

print("\nSINGLE-SPLIT PERFORMANCE COMPARISON:")
print(comparison_df.round(3).to_string(index=False))
comparison_df.to_csv("./nataging/svm_performance_comparison.csv", index=False)
print("\n✓ Saved: svm_performance_comparison.csv")

print("\nPERFORMANCE DIFFERENCES (Real Data - Predicted Data):")
for model_type in ["Real Model", "Pred Model"]:
    print(f"\n  {model_type}:")
    r_real = single_results[f"{model_type} (Real Data)"]
    r_pred = single_results[f"{model_type} (Predicted Data)"]
    for metric in ["Precision", "Recall", "F1"]:
        diff = r_real[metric] - r_pred[metric]
        direction = "better" if diff > 0 else "worse" if diff < 0 else "same"
        print(f"    {metric}: {diff:+.3f}  ({direction} with real data)")

print("\n" + "=" * 80)
print("PER-CLASS CLASSIFICATION REPORTS")
print("=" * 80)
for key, res in single_results.items():
    print(f"\n{key.upper()}:")
    print("-" * 50)
    print(res["Classification Report"])

print("\n" + "=" * 80)
print("CONFUSION MATRICES  (rows = true, cols = predicted)")
print("=" * 80)
for key, res in single_results.items():
    cm = res["Confusion Matrix"]
    correct = cm.diagonal()
    totals = cm.sum(axis=1)
    print(f"\n{key.upper()}:")
    print(cm)
    for i, cls in enumerate(["CN", "MCI", "AD"]):
        acc = correct[i] / totals[i] if totals[i] > 0 else float("nan")
        print(f"  {cls}: {acc:.3f}  ({correct[i]}/{totals[i]})")
    print(f"  Overall: {correct.sum()/cm.sum():.3f}  ({correct.sum()}/{cm.sum()})")

# ==============================================================================
# 5-FOLD STRATIFIED CROSS-VALIDATION
# ==============================================================================
print("\n" + "=" * 80)
print("5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 80)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_rows = []

for fold, (fold_train_idx, fold_test_idx) in enumerate(skf.split(X_real, y), 1):
    print(f"\n  Fold {fold}/5 — train: {len(fold_train_idx):,}  test: {len(fold_test_idx):,}")

    # fold_* names prevent overwriting the original 80/20 split variables
    fold_X_real_tr, fold_X_real_te = X_real[fold_train_idx], X_real[fold_test_idx]
    fold_X_pred_tr, fold_X_pred_te = X_pred[fold_train_idx], X_pred[fold_test_idx]
    fold_y_tr, fold_y_te = y[fold_train_idx], y[fold_test_idx]

    fold_real_model = make_svm().fit(fold_X_real_tr, fold_y_tr)
    fold_pred_model = make_svm().fit(fold_X_pred_tr, fold_y_tr)

    combos = {
        ("Real Model", "Real Data"):      get_metrics(fold_y_te, fold_real_model.predict(fold_X_real_te)),
        ("Real Model", "Predicted Data"): get_metrics(fold_y_te, fold_real_model.predict(fold_X_pred_te)),
        ("Pred Model", "Real Data"):      get_metrics(fold_y_te, fold_pred_model.predict(fold_X_real_te)),
        ("Pred Model", "Predicted Data"): get_metrics(fold_y_te, fold_pred_model.predict(fold_X_pred_te)),
    }

    for (model_type, test_data), res in combos.items():
        fold_rows.append({
            "Model Type": model_type,
            "Test Data":  test_data,
            "Fold":       fold,
            "Precision":  res["Precision"],
            "Recall":     res["Recall"],
            "F1":         res["F1"],
        })

cv_df = pd.DataFrame(fold_rows)
cv_df.to_csv("./nataging/svm_cross_validation_results.csv", index=False)
print("\n✓ Saved: svm_cross_validation_results.csv")

# ==============================================================================
# CV SUMMARY + PAIRED T-TESTS
# ==============================================================================
print("\n" + "=" * 80)
print("CROSS-VALIDATION SUMMARY  (mean +/- std, n=5 folds)")
print("=" * 80)
stats_summary = (
    cv_df.groupby(["Model Type", "Test Data"])[["Precision", "Recall", "F1"]]
    .agg(["mean", "std"])
    .round(3)
)
print(stats_summary)

print("\nPAIRED T-TESTS  (Real Data vs Predicted Data, n=5 folds):")
for model_type in ["Real Model", "Pred Model"]:
    for metric in ["Precision", "Recall", "F1"]:
        rv = cv_df.loc[
            (cv_df["Model Type"] == model_type) & (cv_df["Test Data"] == "Real Data"), metric
        ].values
        pv = cv_df.loc[
            (cv_df["Model Type"] == model_type) & (cv_df["Test Data"] == "Predicted Data"), metric
        ].values
        t_stat, p_val = stats.ttest_rel(rv, pv)
        flag = " *" if p_val < 0.05 else ""
        print(f"  {model_type} | {metric:9s}: t={t_stat:+.3f}, p={p_val:.4f}{flag}")

# ==============================================================================
# AGGREGATE CV STATS FOR PLOTTING
# ==============================================================================
agg = (
    cv_df.groupby(["Model Type", "Test Data"])[["Precision", "Recall", "F1"]]
    .agg(["mean", "std"])
    .reset_index()
)
agg.columns = [
    "Model Type", "Test Data",
    "Precision Mean", "Precision Std",
    "Recall Mean",    "Recall Std",
    "F1 Mean",        "F1 Std",
]


def cv_mean(model_type, test_data, metric):
    return agg.loc[
        (agg["Model Type"] == model_type) & (agg["Test Data"] == test_data),
        f"{metric} Mean",
    ].values[0]


def cv_std(model_type, test_data, metric):
    return agg.loc[
        (agg["Model Type"] == model_type) & (agg["Test Data"] == test_data),
        f"{metric} Std",
    ].values[0]


N_FOLDS = 5
T_CRIT  = stats.t.ppf(0.975, df=N_FOLDS - 1)   # ~2.776 for df=4


def cv_ci(model_type, test_data, metric):
    """Half-width of the 95% CI: t_(0.975, df=4) * std / sqrt(n)."""
    std = cv_std(model_type, test_data, metric)
    return T_CRIT * std / np.sqrt(N_FOLDS)


# Pre-compute p-values for all 6 brackets (2 model types x 3 metrics)
pvals = {}
for model_type in ["Real Model", "Pred Model"]:
    for metric in ["Precision", "Recall", "F1"]:
        rv = cv_df.loc[
            (cv_df["Model Type"] == model_type) & (cv_df["Test Data"] == "Real Data"), metric
        ].values
        pv = cv_df.loc[
            (cv_df["Model Type"] == model_type) & (cv_df["Test Data"] == "Predicted Data"), metric
        ].values
        _, p = stats.ttest_rel(rv, pv)
        pvals[(model_type, metric)] = p

# ==============================================================================
# COMBINED SINGLE-PANEL PUBLICATION PLOT
# Layout  : 3 metric groups (Precision, Recall, F1) x 4 bars each
# Encoding: color  -> model type  (blue = Real Model, red = Pred Model)
#           hatch  -> test data   (solid = Real Data, /// = Predicted Data)
# Error bars: CV mean +/- std (n = 5 folds)
# Brackets  : paired t-test  "ns" p>=0.05  |  "*" p<0.05
# ==============================================================================
print("\n" + "=" * 80)
print("CREATING COMBINED SINGLE-PANEL PUBLICATION PLOT")
print("=" * 80)

plt.rcdefaults()
sns.set_style("white")
plt.rcParams.update({
    "font.size":       13,
    "axes.labelsize":  15,
    "axes.titlesize":  14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 10,
    "figure.dpi":      150,
})

METRICS    = ["Precision", "Recall", "F1"]
CONDITIONS = [
    ("Real Model", "Real Data"),       # bar 0 — blue solid
    ("Real Model", "Predicted Data"),  # bar 1 — blue hatched
    ("Pred Model", "Real Data"),       # bar 2 — red solid
    ("Pred Model", "Predicted Data"),  # bar 3 — red hatched
]
BAR_W      = 0.17
GROUP_GAP  = 0.32
GROUP_W    = len(CONDITIONS) * BAR_W
GROUP_CTRS = np.array([i * (GROUP_W + GROUP_GAP) for i in range(len(METRICS))])
OFFSETS    = np.array([-1.5, -0.5, 0.5, 1.5]) * BAR_W

BAR_COLORS  = ["#2E86C1", "#2E86C1", "#C0392B", "#C0392B"]
BAR_HATCHES = ["",        "///",     "",        "///"]

fig, ax = plt.subplots(figsize=(10, 6.5))

bar_xpos = {}   # (metric_idx, cond_idx) -> x centre

for mi, metric in enumerate(METRICS):
    for ci, (model_type, test_data) in enumerate(CONDITIONS):
        xc      = GROUP_CTRS[mi] + OFFSETS[ci]
        mean    = cv_mean(model_type, test_data, metric)
        half_ci = cv_ci(model_type, test_data, metric)

        # Hatched bars get black edges so the hatch lines are solid black;
        # solid bars keep their own colour for a clean border.
        edge_color = "black" if BAR_HATCHES[ci] else BAR_COLORS[ci]
        ax.bar(
            xc, mean, BAR_W,
            color=BAR_COLORS[ci],
            hatch=BAR_HATCHES[ci],
            edgecolor=edge_color,
            linewidth=0.6,
            alpha=0.85,
            zorder=2,
        )
        ax.errorbar(
            xc, mean, yerr=half_ci,
            fmt="none",
            ecolor="black",
            elinewidth=1.1,
            capsize=3.5,
            capthick=1.1,
            zorder=3,
        )
        bar_xpos[(mi, ci)] = xc

# Significance brackets
# Two per metric group:
#   cond 0 vs 1 — Real Model (real data vs predicted data)
#   cond 2 vs 3 — Pred Model (real data vs predicted data)
BRACKET_PAIRS = [
    (0, 1, "Real Model"),
    (2, 3, "Pred Model"),
]
BRACKET_LIFT = 0.022
TICK_H       = 0.008
LABEL_PAD    = 0.004

for mi, metric in enumerate(METRICS):
    for ci_a, ci_b, model_type in BRACKET_PAIRS:
        xa = bar_xpos[(mi, ci_a)]
        xb = bar_xpos[(mi, ci_b)]

        top_a = cv_mean(model_type, CONDITIONS[ci_a][1], metric) + \
                cv_ci(model_type,  CONDITIONS[ci_a][1], metric)
        top_b = cv_mean(model_type, CONDITIONS[ci_b][1], metric) + \
                cv_ci(model_type,  CONDITIONS[ci_b][1], metric)
        y_bracket = max(top_a, top_b) + BRACKET_LIFT

        p_val = pvals[(model_type, metric)]
        label = "ns" if p_val >= 0.05 else "*"

        ax.plot(
            [xa, xa, xb, xb],
            [y_bracket - TICK_H, y_bracket, y_bracket, y_bracket - TICK_H],
            lw=0.9, color="black", zorder=4,
        )
        ax.text(
            (xa + xb) / 2, y_bracket + LABEL_PAD,
            label,
            ha="center", va="bottom",
            fontsize=15, color="black", zorder=4,
        )

# Axes
ax.set_xticks(GROUP_CTRS)
ax.set_xticklabels(METRICS, fontsize=18)
ax.set_xlim(GROUP_CTRS[0]  - GROUP_W * 0.80,
            GROUP_CTRS[-1] + GROUP_W * 0.80)

all_tops = [
    cv_mean(mt, td, m) + cv_ci(mt, td, m)
    for m in METRICS for mt, td in CONDITIONS
]
y_max = max(all_tops) + BRACKET_LIFT + TICK_H + LABEL_PAD + 0.06
y_min = min(
    cv_mean(mt, td, m) - cv_ci(mt, td, m)
    for m in METRICS for mt, td in CONDITIONS
) - 0.04
ax.set_ylim(max(0.0, y_min), min(1.0, y_max))

ax.set_ylabel("Score (CV mean \u00b1 95% CI)", fontsize=18)
ax.set_xlabel("Metric", fontsize=18)
ax.set_title(
    "Diagnosis classification using real and predicted ROI volumes\n"
    "SVM  -  5-fold stratified cross-validation",
    fontsize=15, pad=14,
)

sns.despine(ax=ax)
ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
ax.set_axisbelow(True)

legend_handles = [
    mpatches.Patch(facecolor="#2E86C1", edgecolor="#2E86C1",
                   linewidth=0.6, alpha=0.85,
                   label="Real model - real test data"),
    mpatches.Patch(facecolor="#2E86C1", edgecolor="black",
                   hatch="///", linewidth=0.6, alpha=0.85,
                   label="Real model - predicted test data"),
    mpatches.Patch(facecolor="#C0392B", edgecolor="#C0392B",
                   linewidth=0.6, alpha=0.85,
                   label="Pred model - real test data"),
    mpatches.Patch(facecolor="#C0392B", edgecolor="black",
                   hatch="///", linewidth=0.6, alpha=0.85,
                   label="Pred model - predicted test data"),
    mpatches.Patch(facecolor="none", edgecolor="none",
                   label="Brackets: ns p>=0.05  |  * p<0.05"),
]
ax.legend(
    handles=legend_handles,
    loc="upper right",
    fontsize=12,
    frameon=True,
    framealpha=0.92,
    edgecolor="#cccccc",
)

plt.tight_layout()
plt.savefig("./nataging/diagnosis_classification_results.png",
            dpi=600, bbox_inches="tight")
plt.savefig("./nataging/diagnosis_classification_results.pdf",
            bbox_inches="tight")
plt.savefig("./nataging/diagnosis_classification_results.svg",
            bbox_inches="tight")
plt.close()
print("✓ Saved: diagnosis_classification_results  (.png / .pdf / .svg)")

# ==============================================================================
# SUPPLEMENTARY: PER-METRIC CV BOXPLOTS
# ==============================================================================
print("\nCreating supplementary CV boxplots...")

for metric in ["Precision", "Recall", "F1"]:
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(
        data=cv_df,
        x="Model Type", y=metric, hue="Test Data",
        palette={"Real Data": "#2E86C1", "Predicted Data": "#C0392B"},
        ax=ax,
    )
    for i, model_type in enumerate(["Real Model", "Pred Model"]):
        rv = cv_df.loc[
            (cv_df["Model Type"] == model_type) & (cv_df["Test Data"] == "Real Data"), metric
        ].values
        pv = cv_df.loc[
            (cv_df["Model Type"] == model_type) & (cv_df["Test Data"] == "Predicted Data"), metric
        ].values
        _, p_val = stats.ttest_rel(rv, pv)
        if p_val < 0.05:
            y_ann = max(rv.max(), pv.max()) + 0.015
            ax.text(i, y_ann, "*", ha="center", va="bottom", fontsize=14)

    ax.set_title(f"{metric}  -  5-fold stratified CV", pad=10)
    ax.set_ylim(0.45, 1.0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(f"./nataging/svm_{metric.lower()}_cv_boxplot.pdf",
                dpi=300, bbox_inches="tight")
    plt.savefig(f"./nataging/svm_{metric.lower()}_cv_boxplot.png",
                dpi=300, bbox_inches="tight")
    plt.close()

print("Saved: svm_{precision,recall,f1}_cv_boxplot  (.pdf / .png)")
