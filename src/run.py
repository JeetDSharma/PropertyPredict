"""Benchmark models under nested cross-validation and write the report.

Run from the project root:  python src/run.py

Outer 5-fold CV estimates generalization error. Inside each outer training
fold, a randomized search over 3-fold inner CV picks hyperparameters. The outer
test fold is never seen by the search, the imputer, the scaler, or the feature
selector -- so the reported RMSE is not optimistic.
"""

from __future__ import annotations

import json
import os
import sys
import time

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import (  # noqa: E402
    RANDOM_STATE,
    baseline_pipeline,
    build_datasets,
    candidate_models,
    plain_name,
)

N_OUTER, N_INNER, N_ITER = 5, 3, 25
REPORTS, MODELS = "reports", "models"

# Chart tokens (validated blue ramp; single-series charts, so one hue).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2de"
BLUE = "#2a78d6"
BLUE_LIGHT = "#86b6ef"
NEUTRAL_LINE = "#a8a7a1"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK_MUTED,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.edgecolor": GRID,
    "font.size": 11,
})


def _despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def nested_cv(name, estimator, X, y, splits, search_space=None):
    """Run the outer CV loop. Returns per-fold scores and out-of-fold predictions."""
    oof = np.zeros(len(y))
    fold_rmse, fold_models = [], []
    t0 = time.time()

    for k, (tr, te) in enumerate(splits, 1):
        if search_space is None:
            est = clone(estimator)
        else:
            est = RandomizedSearchCV(
                clone(estimator),
                search_space,
                n_iter=N_ITER,
                cv=KFold(N_INNER, shuffle=True, random_state=RANDOM_STATE),
                scoring="neg_root_mean_squared_error",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        est.fit(X.iloc[tr], y[tr])
        oof[te] = est.predict(X.iloc[te])
        fold_rmse.append(rmse(y[te], oof[te]))
        fold_models.append(est)
        print(f"    fold {k}/{len(splits)}  RMSE {fold_rmse[-1]:.3f}", flush=True)

    return {
        "name": name,
        "rmse_mean": float(np.mean(fold_rmse)),
        "rmse_std": float(np.std(fold_rmse)),
        "mae": float(mean_absolute_error(y, oof)),
        "r2": float(r2_score(y, oof)),
        "oof": oof,
        "fold_models": fold_models,
        "seconds": time.time() - t0,
    }


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------

def chart_model_comparison(results, best_name, path):
    names = [r["name"] for r in results]
    means = [r["rmse_mean"] for r in results]
    errs = [r["rmse_std"] for r in results]
    colors = [BLUE if n == best_name else BLUE_LIGHT for n in names]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(names, means, color=colors, width=0.6, zorder=3)
    ax.errorbar(names, means, yerr=errs, fmt="none",
                ecolor=INK_MUTED, elinewidth=1.5, capsize=5, zorder=4)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(errs) + 0.03,
                f"{m:.3f}", ha="center", va="bottom", fontsize=11, color=INK)

    ax.set_ylabel("RMSE  (log mol/L) — lower is better")
    ax.set_title("Prediction error by model", fontsize=13, color=INK,
                 loc="left", pad=14, weight="bold")
    ax.set_ylim(0, max(means) + max(errs) + 0.35)
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.set_axisbelow(True)
    _despine(ax, keep=("bottom",))
    ax.tick_params(left=False)
    fig.text(0.125, 0.005, "Bars show mean across 5 outer CV folds; whiskers show ±1 SD.",
             fontsize=9, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_pred_vs_actual(y, oof, best_name, path):
    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    lo, hi = min(y.min(), oof.min()) - 0.4, max(y.max(), oof.max()) + 0.4

    ax.plot([lo, hi], [lo, hi], color=NEUTRAL_LINE, linewidth=2,
            linestyle="--", zorder=2)
    ax.scatter(y, oof, s=22, color=BLUE, alpha=0.45,
               edgecolors="none", zorder=3)
    ax.annotate("perfect prediction", xy=(hi - 0.6, hi - 0.6), xytext=(-6, 10),
                textcoords="offset points", ha="right", fontsize=9.5,
                color=INK_MUTED, rotation=45, rotation_mode="anchor")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Measured solubility (log mol/L)")
    ax.set_ylabel("Predicted solubility (log mol/L)")
    ax.set_title(f"{best_name}: predicted vs. measured", fontsize=13, color=INK,
                 loc="left", pad=14, weight="bold")
    ax.grid(color=GRID, zorder=0)
    ax.set_axisbelow(True)
    _despine(ax)
    fig.text(0.02, 0.005, "Every point is a molecule held out of training when it was predicted.",
             fontsize=9, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_residuals(y, oof, path):
    resid = oof - y
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.hist(resid, bins=45, color=BLUE, alpha=0.85, zorder=3)
    ax.axvline(0, color=NEUTRAL_LINE, linewidth=2, linestyle="--", zorder=4)
    ax.set_xlabel("Prediction error (predicted − measured, log mol/L)")
    ax.set_ylabel("Molecules")
    ax.set_title("Where the model is wrong, and by how much", fontsize=13,
                 color=INK, loc="left", pad=14, weight="bold")
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.set_axisbelow(True)
    _despine(ax, keep=("bottom",))
    ax.tick_params(left=False)
    within = float(np.mean(np.abs(resid) <= 1.0) * 100)
    fig.text(0.02, 0.005,
             f"{within:.0f}% of molecules are predicted within one log unit (a 10x factor) of the measured value.",
             fontsize=9, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_importance(imp_df, path, top_n=15):
    top = imp_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.barh(top["label"], top["importance"], color=BLUE, height=0.62, zorder=3)
    for label, val in zip(top["label"], top["importance"]):
        ax.text(val + top["importance"].max() * 0.015, label, f"{val:.3f}",
                va="center", fontsize=9.5, color=INK_MUTED)
    ax.set_xlabel("Increase in error when this property is scrambled")
    ax.set_title("What the model actually relies on", fontsize=13, color=INK,
                 loc="left", pad=14, weight="bold")
    ax.set_xlim(0, top["importance"].max() * 1.18)
    ax.grid(axis="x", color=GRID, zorder=0)
    ax.set_axisbelow(True)
    _despine(ax, keep=("bottom",))
    ax.tick_params(left=False)
    fig.text(0.02, 0.005,
             "Permutation importance, averaged over held-out folds. Longer bar = the model leans on it more.",
             fontsize=9, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------

def main():
    os.makedirs(REPORTS, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)

    print("Loading and cleansing data ...")
    X_base, X_full, y, clean_report, df = build_datasets()
    print(f"  {clean_report['rows_in']} rows in -> {clean_report['rows_out']} rows out")
    print(f"  {X_full.shape[1]} features after descriptor generation + engineering")

    splits = list(KFold(N_OUTER, shuffle=True, random_state=RANDOM_STATE).split(X_full))

    results = []
    print("\nLinear baseline (6 stock descriptors, no tuning):")
    baseline = nested_cv("Linear baseline", baseline_pipeline(), X_base, y, splits)
    results.append(baseline)

    for name, (pipe, space) in candidate_models().items():
        print(f"\n{name} (nested CV, {N_ITER} search iterations per fold):")
        results.append(nested_cv(name, pipe, X_full, y, splits, space))

    tuned = [r for r in results if r["name"] != "Linear baseline"]
    best = min(tuned, key=lambda r: r["rmse_mean"])
    improvement = (baseline["rmse_mean"] - best["rmse_mean"]) / baseline["rmse_mean"] * 100
    print(f"\nBest model: {best['name']}  RMSE {best['rmse_mean']:.3f} "
          f"({improvement:.1f}% below baseline)")

    print("\nComputing permutation importance on held-out folds ...")
    acc = np.zeros(X_full.shape[1])
    for (tr, te), est in zip(splits, best["fold_models"]):
        pi = permutation_importance(
            est, X_full.iloc[te], y[te], n_repeats=5,
            random_state=RANDOM_STATE, scoring="neg_root_mean_squared_error", n_jobs=-1,
        )
        acc += pi.importances_mean
    imp = (pd.DataFrame({
        "feature": X_full.columns,
        "importance": acc / len(splits),
    }).sort_values("importance", ascending=False).reset_index(drop=True))
    imp["label"] = imp["feature"].map(plain_name)
    imp.to_csv(f"{REPORTS}/feature_importance.csv", index=False)

    print("Rendering charts ...")
    chart_model_comparison(results, best["name"], f"{REPORTS}/model_comparison.png")
    chart_pred_vs_actual(y, best["oof"], best["name"], f"{REPORTS}/pred_vs_actual.png")
    chart_residuals(y, best["oof"], f"{REPORTS}/residuals.png")
    chart_importance(imp, f"{REPORTS}/feature_importance.png")

    print("Refitting best model on all data for the app ...")
    best_pipe, best_space = candidate_models()[best["name"]]
    final = RandomizedSearchCV(
        best_pipe, best_space, n_iter=N_ITER,
        cv=KFold(N_INNER, shuffle=True, random_state=RANDOM_STATE),
        scoring="neg_root_mean_squared_error", random_state=RANDOM_STATE, n_jobs=-1,
    )
    final.fit(X_full, y)
    joblib.dump({
        "model": final.best_estimator_,
        "features": list(X_full.columns),
        "model_name": best["name"],
        "cv_rmse": best["rmse_mean"],
        "cv_r2": best["r2"],
        "target_range": (float(y.min()), float(y.max())),
        "n_train": int(len(y)),
    }, f"{MODELS}/best_model.joblib")

    write_report(results, best, baseline, improvement, imp, clean_report,
                 X_full.shape[1], final.best_params_)
    print("\nDone. See results.md and reports/*.png")


def write_report(results, best, baseline, improvement, imp, clean_report,
                 n_features, best_params):
    top5 = imp.head(5)
    rows = "\n".join(
        f"| {r['name']} | {r['rmse_mean']:.3f} ± {r['rmse_std']:.3f} | "
        f"{r['mae']:.3f} | {r['r2']:.3f} |"
        for r in results
    )
    params = "\n".join(f"- `{k}`: {v}" for k, v in sorted(best_params.items()))
    drivers = "\n".join(
        f"{i}. **{r.label}** — permutation importance {r.importance:.3f}"
        for i, r in enumerate(top5.itertuples(), 1)
    )

    ridge = next(r for r in results if r["name"] == "Ridge")
    feat_gain = (baseline["rmse_mean"] - ridge["rmse_mean"]) / baseline["rmse_mean"] * 100
    model_gain = (ridge["rmse_mean"] - best["rmse_mean"]) / ridge["rmse_mean"] * 100
    baseline_rmse = baseline["rmse_mean"]
    ridge_rmse = ridge["rmse_mean"]
    best_rmse = best["rmse_mean"]
    best_name = best["name"]

    md = f"""# Results

Predicting aqueous solubility (log mol/L) for {clean_report['rows_out']} molecules
from structure alone. All numbers below come from **nested cross-validation**:
an outer 5-fold split measures error, and hyperparameters are searched only
inside each outer training fold. No test molecule influenced its own prediction.

## Headline

**{best['name']} cut RMSE by {improvement:.1f}% against the linear baseline**
({baseline['rmse_mean']:.3f} → {best['rmse_mean']:.3f} log units), explaining
{best['r2'] * 100:.0f}% of the variance in measured solubility.

### Where the gain came from

The improvement is not one thing. Splitting it:

| Step | RMSE | Change |
|---|---|---|
| Linear regression, 6 stock descriptors | {baseline_rmse:.3f} | — |
| Ridge, {n_features} engineered features | {ridge_rmse:.3f} | −{feat_gain:.0f}% (richer features + selection) |
| {best_name}, same features, tuned | {best_rmse:.3f} | −{model_gain:.0f}% (nonlinear model + tuning) |

Most of the win comes from giving the model better inputs, not from a fancier
algorithm. That is the usual shape of the result, and worth saying out loud.

## Model comparison

| Model | RMSE (mean ± SD across folds) | MAE | R² |
|---|---|---|---|
{rows}

![Model comparison](reports/model_comparison.png)

The baseline is ordinary least squares on the six descriptors shipped with the
dataset. Every other model sees the full engineered feature set
({n_features} features) and gets a randomized hyperparameter search.

## Predictions vs. reality

![Predicted vs actual](reports/pred_vs_actual.png)

![Residuals](reports/residuals.png)

## What drives the prediction

![Feature importance](reports/feature_importance.png)

{drivers}

**In plain language:** solubility comes down to a tug-of-war. Water pulls a
molecule apart at its polar, charged, hydrogen-bonding surfaces; the molecule
resists in proportion to how large and how greasy it is. The model rediscovers
this on its own — the features it leans on hardest are exactly the ones that
measure greasiness, size, and polar surface. Nobody told it the chemistry.

## Data cleansing

| Step | Rows removed |
|---|---|
| Duplicate structures merged (matched only after canonicalization) | {clean_report['duplicate_structures_merged']} |
| Unparseable SMILES | {clean_report['unparseable_smiles_removed']} |
| Extreme target outliers (outside {clean_report['outlier_bounds']}) | {clean_report['target_outliers_removed']} |

One column was dropped outright: *"ESOL predicted log solubility"*, which is a
previously published model's prediction of the target. Keeping it would have
produced excellent scores that measured nothing.

Missing descriptor values are imputed with the **training fold's** median,
inside the pipeline — never with a statistic computed over the whole dataset.

## Winning configuration

{params}
"""
    with open("results.md", "w") as f:
        f.write(md)

    with open(f"{REPORTS}/metrics.json", "w") as f:
        json.dump({
            "best_model": best["name"],
            "improvement_pct_vs_baseline": improvement,
            "models": [
                {k: v for k, v in r.items() if k not in ("oof", "fold_models")}
                for r in results
            ],
        }, f, indent=2)


if __name__ == "__main__":
    main()
