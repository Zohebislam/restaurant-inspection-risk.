"""
Step 2b: Measure the baseline and train the first models.

Run from the project folder (after features.py):
    python src/train.py
or, after step 3 (external_features.py), with the outside-data features:
    python src/train.py --features features_plus.csv

Time-based split: the models learn from earlier years and are tested on later
years, just like they would be used in real life.

Compares three approaches on the test years:
  1. Baseline: flag a restaurant if it failed its last routine inspection
  2. Logistic regression
  3. Gradient boosting (scikit-learn's HistGradientBoostingClassifier)

The fair comparison: give each model the SAME number of flags as the baseline
(same inspector workload) and count how many failures each one catches.

Outputs (with "_plus" added to each name when using features_plus.csv):
  reports/model_results.md
  reports/figures/09_gains_curve.png
  reports/figures/10_feature_importance.png
  models/gradient_boosting.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"

ID_COLS = ["inspection_id", "license_number", "dba_name", "inspection_date", "latitude", "longitude"]
TARGET = "failed"
BLUE, RED, GRAY, GREEN = "#2b6cb0", "#c53030", "#718096", "#2f855a"
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 12})


def precision_recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> tuple[float, float]:
    """Flag the k highest-scoring inspections; return (precision, recall)."""
    top = np.argsort(-scores, kind="stable")[:k]
    caught = y_true[top].sum()
    return caught / k, caught / y_true.sum()


def gains(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Share of inspections flagged (x) vs. share of all failures caught (y)."""
    order = np.argsort(-scores, kind="stable")
    caught = np.cumsum(y_true[order]) / y_true.sum()
    flagged = np.arange(1, len(y_true) + 1) / len(y_true)
    return np.r_[0, flagged], np.r_[0, caught]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-start", default="2024-01-01",
                        help="inspections on/after this date are the test set (default 2024-01-01)")
    parser.add_argument("--features", default="features.csv",
                        help="feature file in data/processed (default features.csv)")
    args = parser.parse_args()
    test_start = pd.Timestamp(args.test_start)
    features_path = PROCESSED_DIR / args.features
    stem = Path(args.features).stem
    tag = "" if stem == "features" else "_" + stem.replace("features_", "")
    results_path = REPORTS_DIR / f"model_results{tag}.md"
    gains_name, imp_name = f"09_gains_curve{tag}.png", f"10_feature_importance{tag}.png"
    model_path = MODEL_DIR / f"gradient_boosting{tag}.joblib"

    df = pd.read_csv(features_path, parse_dates=["inspection_date"])
    feature_cols = [c for c in df.columns if c not in ID_COLS + [TARGET]]
    train = df[df["inspection_date"] < test_start]
    test = df[df["inspection_date"] >= test_start]
    if train.empty or test.empty:
        raise SystemExit(f"Not enough data on both sides of {test_start:%Y-%m-%d}. Try a different --test-start.")

    X_train, y_train = train[feature_cols], train[TARGET].to_numpy()
    X_test, y_test = test[feature_cols], test[TARGET].to_numpy()
    print(f"Train: {len(train):,} inspections ({train['inspection_date'].min():%b %Y} to "
          f"{train['inspection_date'].max():%b %Y}), {y_train.mean():.1%} failed")
    print(f"Test:  {len(test):,} inspections ({test['inspection_date'].min():%b %Y} to "
          f"{test['inspection_date'].max():%b %Y}), {y_test.mean():.1%} failed\n")

    # 1. Baseline: failed last routine inspection
    base_flag = (test["last_routine_failed"] == 1).to_numpy()
    k = int(base_flag.sum())
    base_precision = y_test[base_flag].mean() if k else 0.0
    base_recall = y_test[base_flag].sum() / y_test.sum()

    # 2. Logistic regression (needs missing values filled and features scaled)
    logreg = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                           LogisticRegression(max_iter=2000, class_weight="balanced"))
    logreg.fit(X_train, y_train)
    lr_scores = logreg.predict_proba(X_test)[:, 1]

    # 3. Gradient boosting (handles missing values on its own)
    gb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                        min_samples_leaf=50, l2_regularization=1.0,
                                        class_weight="balanced", early_stopping=True,
                                        validation_fraction=0.15, random_state=42)
    gb.fit(X_train, y_train)
    gb_scores = gb.predict_proba(X_test)[:, 1]

    # Results table: every approach gets the same number of flags (k)
    rows = [("Baseline: failed last routine", base_precision, base_recall, np.nan, np.nan)]
    for name, s in [("Logistic regression", lr_scores), ("Gradient boosting", gb_scores)]:
        p, r = precision_recall_at_k(y_test, s, k) if k else (np.nan, np.nan)
        rows.append((name, p, r, roc_auc_score(y_test, s), average_precision_score(y_test, s)))
    results = pd.DataFrame(rows, columns=["Model", "Precision", "Recall", "ROC AUC", "Avg precision"])

    best_name, best_scores = max([("Logistic regression", lr_scores), ("Gradient boosting", gb_scores)],
                                 key=lambda t: average_precision_score(y_test, t[1]))
    best_p, best_r = precision_recall_at_k(y_test, best_scores, k) if k else (np.nan, np.nan)
    extra_caught = int(round((best_r - base_recall) * y_test.sum()))
    top20_k = max(1, int(0.2 * len(y_test)))
    _, top20_recall = precision_recall_at_k(y_test, best_scores, top20_k)

    # Gains chart
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, s, color in [("Gradient boosting", gb_scores, GREEN), ("Logistic regression", lr_scores, BLUE)]:
        x, y = gains(y_test, s)
        ax.plot(x, y, color=color, label=name)
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle="--", label="Random order")
    ax.scatter([k / len(y_test)], [base_recall], color=RED, zorder=5, label="Baseline")
    ax.set_title("Failures caught vs. share of restaurants inspected")
    ax.set_xlabel("Share of inspections flagged (riskiest first)")
    ax.set_ylabel("Share of all failures caught")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / gains_name)
    plt.close(fig)

    # Feature importance (permutation importance on a sample of the test set)
    sample = test.sample(min(len(test), 15_000), random_state=42)
    imp = permutation_importance(gb, sample[feature_cols], sample[TARGET], scoring="average_precision",
                                 n_repeats=3, random_state=42, n_jobs=-1)
    importance = pd.Series(imp.importances_mean, index=feature_cols).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(importance.index[::-1], importance.values[::-1], color=GREEN)
    ax.set_title("Most important features (gradient boosting)")
    ax.set_xlabel("Drop in average precision when the feature is shuffled")
    fig.tight_layout()
    fig.savefig(FIG_DIR / imp_name)
    plt.close(fig)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": gb, "features": feature_cols}, model_path)

    # Write results
    def fmt(x: float) -> str:
        return "—" if pd.isna(x) else f"{x:.1%}"

    table = ["| Model | Precision | Recall | ROC AUC | Avg precision |", "|---|---|---|---|---|"]
    table += [f"| {r.Model} | {fmt(r.Precision)} | {fmt(r.Recall)} | "
              f"{'—' if pd.isna(r[3]) else f'{r[3]:.3f}'} | {fmt(r[4])} |" for r in results.itertuples(index=False)]
    lines = [
        "# Model Results: Predicting Restaurant Inspection Failures", "",
        f"- **Train:** {len(train):,} routine inspections, {train['inspection_date'].min():%b %Y} to "
        f"{train['inspection_date'].max():%b %Y}",
        f"- **Test:** {len(test):,} routine inspections, {test['inspection_date'].min():%b %Y} to "
        f"{test['inspection_date'].max():%b %Y} ({y_test.sum():,} failures, {y_test.mean():.1%})",
        f"- **Features:** {len(feature_cols)} (from `{args.features}`)", "",
        "## Same workload comparison",
        f"Each approach flags the same **{k:,}** inspections the baseline flags.", "",
        *table, "",
        (f"**{best_name}** catches **{fmt(best_r)}** of failures vs. **{fmt(base_recall)}** for the baseline, "
         f"about **{extra_caught:,} more failing restaurants** found with the same number of flags."
         if extra_caught > 0 else
         f"**{best_name}** catches **{fmt(best_r)}** of failures vs. **{fmt(base_recall)}** for the baseline, "
         "so it does not beat the baseline yet. Revisit the features before moving on."), "",
        f"If inspectors visited the riskiest 20% of restaurants first, the model would catch "
        f"**{fmt(top20_recall)}** of all failures (random order would catch about 20%).", "",
        "## Gains curve", f"![gains](figures/{gains_name})", "",
        "## Most important features", f"![importance](figures/{imp_name})", "",
        *[f"{i + 1}. `{name}` ({val:.4f})" for i, (name, val) in enumerate(importance.items())], "",
    ]
    results_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved results to {results_path} and model to {model_path}")


if __name__ == "__main__":
    main()
