"""
Step 1c: First look at the data (exploratory data analysis).

Run from the project folder (after clean.py):
    python src/eda.py

Saves charts to reports/figures/ and a written summary to reports/eda_summary.md.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "data" / "processed" / "inspections_clean.csv"
FIG_DIR = ROOT / "reports" / "figures"
SUMMARY_PATH = ROOT / "reports" / "eda_summary.md"

BLUE, RED, GRAY = "#2b6cb0", "#c53030", "#718096"
plt.rcParams.update({
    "figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 12,
})


def save(fig, name: str) -> str:
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return f"figures/{name}"


def pct(x: float) -> str:
    return f"{x:.1%}"


def bar_rates(rates: pd.Series, title: str, xlabel: str, name: str, highlight=None) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [RED if highlight is not None and i == highlight else BLUE for i in rates.index]
    ax.bar(rates.index.astype(str), rates.values, color=colors)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Failure rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    for i, v in enumerate(rates.values):
        ax.text(i, v, pct(v), ha="center", va="bottom", fontsize=8)
    return save(fig, name)


def main() -> None:
    df = pd.read_csv(CLEAN_PATH, parse_dates=["inspection_date"])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# Exploratory Data Analysis: Chicago Restaurant Inspections", ""]

    overall = df["failed"].mean()
    lines += [
        "## Overview",
        f"- **{len(df):,}** restaurant inspections of **{df['license_number'].nunique():,}** restaurants",
        f"- Dates: {df['inspection_date'].min():%b %Y} to {df['inspection_date'].max():%b %Y}",
        f"- Overall failure rate: **{pct(overall)}**",
        "",
    ]

    # 1. Result distribution
    counts = df["results"].value_counts().reindex(["Pass", "Pass w/ Conditions", "Fail"]).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(counts.index, counts.values, color=[BLUE, GRAY, RED])
    ax.set_title("Inspection results")
    ax.set_ylabel("Inspections")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=8)
    lines += ["## Results", f"![results]({save(fig, '01_results.png')})", ""]
    lines += [f"- {k}: {int(v):,} ({pct(v / len(df))})" for k, v in counts.items()] + [""]

    # 2. By inspection category
    by_cat = df.groupby("inspection_category")["failed"].agg(["mean", "size"]).sort_values("mean", ascending=False)
    img = bar_rates(by_cat["mean"], "Failure rate by inspection type", "Inspection type", "02_by_category.png")
    lines += ["## By inspection type", f"![category]({img})", ""]
    lines += [f"- {c}: {pct(r['mean'])} of {int(r['size']):,} inspections" for c, r in by_cat.iterrows()]
    lines += ["", "Routine (canvass) inspections are the ones a risk model would help schedule. "
              "Complaint and re-inspections behave differently, so model them separately or as a feature.", ""]

    # 3. By year
    by_year = df.groupby("year")["failed"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(by_year.index, by_year.values, marker="o", color=BLUE)
    if by_year.index.min() <= 2018 <= by_year.index.max():
        ax.axvline(2018.5, color=GRAY, linestyle="--", linewidth=1)
        ax.text(2018.6, ax.get_ylim()[1] * 0.95, "New food code\n(July 2018)", fontsize=8, color=GRAY, va="top")
    ax.set_title("Failure rate by year")
    ax.set_ylabel("Failure rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    lines += ["## Over time", f"![year]({save(fig, '03_by_year.png')})", "",
              "Watch for a shift around July 2018, when Chicago adopted a new food code. "
              "Consider training only on post-2018 data, or adding a `new_food_code` feature.", ""]

    # 4. By month (seasonality)
    by_month = df.groupby("month")["failed"].mean()
    img = bar_rates(by_month, "Failure rate by month", "Month", "04_by_month.png", highlight=by_month.idxmax())
    lines += ["## Seasonality", f"![month]({img})",
              f"- Highest month: {by_month.idxmax()} ({pct(by_month.max())}); "
              f"lowest: {by_month.idxmin()} ({pct(by_month.min())})", ""]

    # 5. By risk level
    by_risk = df.dropna(subset=["risk_level"]).groupby("risk_level")["failed"].mean()
    by_risk.index = by_risk.index.astype(int).map({1: "1 (High)", 2: "2 (Medium)", 3: "3 (Low)"})
    img = bar_rates(by_risk, "Failure rate by city-assigned risk level", "Risk level", "05_by_risk.png")
    lines += ["## By city risk level", f"![risk]({img})", ""]
    lines += [f"- Risk {k}: {pct(v)}" for k, v in by_risk.items()] + [""]

    # 6. Most common violations (new food code only, so numbers are consistent)
    recent = df[df["new_food_code"] == 1]
    codes = recent["violation_codes"].dropna().astype(str).str.split(";").explode()
    codes = codes[codes != ""]
    if not codes.empty:
        names = (recent["violations"].dropna().str.split("|").explode().str.strip()
                 .str.extract(r"^(\d+)\.\s*(.*?)(?:\s*-\s*Comments:.*)?$").dropna()
                 .drop_duplicates(0).set_index(0)[1])
        top = codes.value_counts().head(15)
        def short(text: str, n: int = 50) -> str:
            text = str(text).title()
            return text if len(text) <= n else text[: n - 1].rstrip() + "…"

        labels = [f"#{c}: {short(names.get(c, ''))}" for c in top.index]
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(labels[::-1], top.values[::-1], color=BLUE)
        ax.set_title("Most common violations (since July 2018)")
        ax.set_xlabel("Times cited")
        lines += ["## Most common violations", f"![violations]({save(fig, '06_top_violations.png')})", ""]
        lines += [f"{i + 1}. {lab} ({n:,})" for i, (lab, n) in enumerate(zip(labels, top.values))] + [""]

    # 7. Previous result vs this result (the baseline model preview)
    order = ["Pass", "Pass w/ Conditions", "Fail"]
    has_prev = df.dropna(subset=["prev_result"])
    by_prev = has_prev.groupby("prev_result")["failed"].mean().reindex(order).dropna()
    img = bar_rates(by_prev, "Failure rate by previous result (any inspection type)", "Previous result",
                    "07_by_previous_result.png", highlight="Fail")
    lines += ["## Does the past predict the future?", f"![previous]({img})", ""]
    lines += [f"- After a previous **{k}**: {pct(v)} fail" for k, v in by_prev.items()]
    lines += ["", "Caution: this mixes inspection types. After a fail, the next inspection is usually "
              "a re-inspection, which rarely fails, so this chart understates how much a past "
              "failure matters. The routine-only comparison below is the fair one.", ""]

    # 8. Routine inspections vs. the restaurant's PREVIOUS ROUTINE inspection
    routine = df[df["inspection_category"] == "routine"].sort_values(
        ["license_number", "inspection_date", "inspection_id"]).copy()
    routine["prev_routine_result"] = routine.groupby("license_number")["results"].shift(1)
    has_prev_r = routine.dropna(subset=["prev_routine_result"])
    if not has_prev_r.empty:
        by_prev_r = has_prev_r.groupby("prev_routine_result")["failed"].agg(["mean", "size"]).reindex(order).dropna()
        img = bar_rates(by_prev_r["mean"], "Routine inspections: failure rate by previous routine result",
                        "Previous routine result", "08_routine_by_previous_routine.png", highlight="Fail")
        first_time = routine[routine["prev_routine_result"].isna()]["failed"].mean()
        base_rate = has_prev_r["failed"].mean()
        lines += ["## Fair comparison: routine vs. previous routine", f"![routine]({img})", ""]
        lines += [f"- After a previous routine **{k}**: {pct(r['mean'])} fail ({int(r['size']):,} inspections)"
                  for k, r in by_prev_r.iterrows()]
        lines += [f"- First routine inspection on record: {pct(first_time)} fail",
                  f"- All routine inspections with a previous routine: {pct(base_rate)} fail", "",
                  "Baseline model to beat: flag a restaurant as high risk if it failed its last routine inspection.", ""]

    SUMMARY_PATH.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved charts to {FIG_DIR} and summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
