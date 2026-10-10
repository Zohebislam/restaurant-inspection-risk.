"""
Step 4: Business impact simulation.

Run from the project folder (after train.py):
    python src/simulation.py
or, to use the model trained on the outside-data features:
    python src/simulation.py --features features_plus.csv

The question: if inspectors had visited restaurants in the model's risk order
instead of the order they actually did, how much sooner would failing
restaurants have been found?

How it works, for each month in the test period (2024 onward):
  - take the routine inspections that actually happened that month
  - keep the same inspection dates (same workload, same staff)
  - hand those dates out in risk order: the riskiest restaurant gets the
    earliest date, and so on
  - for every restaurant that failed, compare its actual date to its new date

Simplification: it assumes the same restaurants would have been inspected that
month and that a restaurant's result wouldn't change if it were visited a few
days earlier or later.

Outputs:
  reports/simulation_results.md
  reports/figures/11_simulation.png
  data/processed/scored_inspections.csv   (used by the Streamlit app)
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

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
CLEAN_PATH = PROCESSED_DIR / "inspections_clean.csv"
SCORED_PATH = PROCESSED_DIR / "scored_inspections.csv"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
RESULTS_PATH = ROOT / "reports" / "simulation_results.md"

NAVY, FLAG_RED, GRAY = "#10263B", "#E4002B", "#8AA0B4"
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 12})

# Columns kept for the app (explain each restaurant's score in plain terms)
APP_COLS = ["prior_inspections", "prior_fail_rate", "prior_avg_violations",
            "last_routine_failed", "last_routine_cond", "last_routine_violations",
            "days_since_last_routine", "prior_complaints", "risk_level"]


def simulate(test: pd.DataFrame) -> pd.DataFrame:
    """Reassign each month's actual inspection dates in risk order."""
    out = []
    for _, g in test.groupby(test["inspection_date"].dt.to_period("M")):
        actual = g.sort_values(["inspection_date", "inspection_id"])
        by_risk = g.sort_values("risk_score", ascending=False, kind="stable")
        n = len(g)
        g = g.copy()
        g["sim_date"] = pd.Series(actual["inspection_date"].to_numpy(), index=by_risk.index)
        g["actual_position"] = pd.Series(np.arange(1, n + 1) / n, index=actual.index)
        g["sim_position"] = pd.Series(np.arange(1, n + 1) / n, index=by_risk.index)
        out.append(g)
    sim = pd.concat(out)
    sim["days_earlier"] = (sim["inspection_date"] - sim["sim_date"]).dt.days
    return sim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="features.csv",
                        help="feature file in data/processed (default features.csv)")
    parser.add_argument("--test-start", default="2024-01-01",
                        help="same test start date used in train.py (default 2024-01-01)")
    args = parser.parse_args()

    stem = Path(args.features).stem
    tag = "" if stem == "features" else "_" + stem.replace("features_", "")
    model_path = MODEL_DIR / f"gradient_boosting{tag}.joblib"
    if not model_path.exists():
        cmd = "python src/train.py" + ("" if not tag else f" --features {args.features}")
        raise SystemExit(f"Model not found at {model_path}. Run this first:\n    {cmd}")

    bundle = joblib.load(model_path)
    model, feature_cols = bundle["model"], bundle["features"]
    df = pd.read_csv(PROCESSED_DIR / args.features, parse_dates=["inspection_date"])
    test = df[df["inspection_date"] >= pd.Timestamp(args.test_start)].copy()
    if test.empty:
        raise SystemExit("No inspections in the test period. Check --test-start.")

    # Scores come from a model trained only on earlier years, so these are honest predictions
    test["risk_score"] = model.predict_proba(test[feature_cols])[:, 1]
    sim = simulate(test)
    fails = sim[sim["failed"] == 1]

    # --- Headline numbers ---
    n_fail = len(fails)
    avg_days = fails["days_earlier"].mean()
    median_days = fails["days_earlier"].median()
    pct_sooner = (fails["days_earlier"] > 0).mean()
    pct_later = (fails["days_earlier"] < 0).mean()
    total_days = int(fails["days_earlier"].sum())
    first_half_actual = (fails["actual_position"] <= 0.5).mean()
    first_half_model = (fails["sim_position"] <= 0.5).mean()
    n_months = sim["inspection_date"].dt.to_period("M").nunique()

    # Same-workload check against the baseline (failed last routine inspection)
    base_flag = sim["last_routine_failed"] == 1
    k = int(base_flag.sum())
    top_k = sim.nlargest(k, "risk_score")
    base_caught, model_caught = int(sim.loc[base_flag, "failed"].sum()), int(top_k["failed"].sum())

    # --- Chart: share of each month's failures found as the month goes on ---
    grid = np.linspace(0, 1, 101)
    actual_curve = [(fails["actual_position"] <= x).mean() for x in grid]
    model_curve = [(fails["sim_position"] <= x).mean() for x in grid]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(grid, model_curve, color=FLAG_RED, linewidth=2.2, label="Model's risk order")
    ax.plot(grid, actual_curve, color=NAVY, linewidth=1.6, label="Actual order")
    ax.axvline(0.5, color=GRAY, linestyle=":", linewidth=1)
    ax.set_title("Failing restaurants found as each month goes on")
    ax.set_xlabel("Share of the month's inspections completed")
    ax.set_ylabel("Share of the month's failures found")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_simulation.png")
    plt.close(fig)

    # --- Save scored inspections for the app ---
    clean = pd.read_csv(CLEAN_PATH, usecols=["inspection_id", "address", "zip", "results"],
                        dtype={"zip": str})
    keep = (["inspection_id", "license_number", "dba_name", "inspection_date", "latitude", "longitude",
             "risk_score", "failed", "sim_date", "days_earlier", "actual_position", "sim_position"]
            + [c for c in APP_COLS if c in sim.columns])
    scored = sim[keep].merge(clean, on="inspection_id", how="left")
    scored.to_csv(SCORED_PATH, index=False)

    # --- Report ---
    def pct(x: float) -> str:
        return f"{x:.1%}"

    lines = [
        "# Business Impact Simulation", "",
        f"- **Period:** {sim['inspection_date'].min():%b %Y} to {sim['inspection_date'].max():%b %Y} "
        f"({n_months} months, {len(sim):,} routine inspections, {n_fail:,} failures)",
        f"- **Model:** `{model_path.name}` (trained only on inspections before {args.test_start})",
        "- **Setup:** each month keeps its real inspection dates; the model only changes which "
        "restaurant gets visited first.", "",
        "## Headline results",
        f"- Failing restaurants would have been found **{avg_days:.1f} days sooner on average** "
        f"(median {median_days:.0f} days).",
        f"- **{pct(pct_sooner)}** of failing restaurants would have been found sooner, "
        f"{pct(pct_later)} later.",
        f"- In total, that's **{total_days:,} fewer days** that failing kitchens stayed open "
        "before an inspector arrived.",
        f"- By the middle of each month, inspectors would have found **{pct(first_half_model)}** of "
        f"that month's failures, compared to **{pct(first_half_actual)}** in the actual order.", "",
        "## Same workload as the simple rule",
        f"Flagging the same {k:,} inspections as the baseline (\"failed last routine inspection\"), "
        f"the model finds **{model_caught:,}** failing restaurants vs. **{base_caught:,}** for the baseline "
        f"(**{model_caught - base_caught:+,}**).", "",
        "## Chart", "![simulation](figures/11_simulation.png)", "",
        "## Caveats",
        "- Assumes the same restaurants would have been inspected each month, and that a restaurant's "
        "result wouldn't change if it were visited a few days earlier or later.",
        "- Only routine inspections are reordered; complaint inspections happen when complaints come in.",
        "",
    ]
    RESULTS_PATH.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"Saved report to {RESULTS_PATH} and app data to {SCORED_PATH}")


if __name__ == "__main__":
    main()
