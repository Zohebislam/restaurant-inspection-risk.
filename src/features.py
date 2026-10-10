"""
Step 2a: Build model features.

Run from the project folder (after clean.py):
    python src/features.py

For every inspection, this builds features from that restaurant's PAST
inspections only (everything is shifted by one inspection, so a row never
sees its own result). It then keeps the rows the model will predict:
routine inspections under the new food code (July 2018 onward).

Output: data/processed/features.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "data" / "processed" / "inspections_clean.csv"
OUT_PATH = ROOT / "data" / "processed" / "features.csv"

CODE_CHANGE_DATE = pd.Timestamp("2018-07-01")
N_TOP_VIOLATIONS = 20  # track history for the 20 most common violation numbers

# Words in inspector comments that may signal a bigger problem.
KEYWORDS = {
    "kw_rodent": r"rodent|mouse|mice|rat |rats|droppings",
    "kw_insect": r"roach|insect|flies|fly |gnat",
    "kw_temperature": r"temperature|temp\.|degrees|cold holding|hot holding",
    "kw_handwash": r"hand ?wash|hand sink|handsink|soap|paper towel",
    "kw_hot_water": r"no hot water|hot water",
    "kw_sewage": r"sewage|backup|back up|drain",
    "kw_mold": r"mold|mildew|slime",
    "kw_cross_contam": r"raw|cross[- ]contam|uncovered",
    "kw_certificate": r"certificate|certified|food handler|manager",
}


def past_cumsum(df: pd.DataFrame, col: str) -> pd.Series:
    """Running total of `col` over each restaurant's EARLIER inspections."""
    return df.groupby("license_number")[col].cumsum() - df[col]


def last_value_before(df: pd.DataFrame, col: str) -> pd.Series:
    """Most recent non-missing value of `col` from an EARLIER inspection of the same restaurant."""
    shifted = df.groupby("license_number")[col].shift(1)
    return shifted.groupby(df["license_number"]).ffill()


def main() -> None:
    df = pd.read_csv(CLEAN_PATH, parse_dates=["inspection_date"], low_memory=False,
                     dtype={"violation_codes": str})
    df = df.sort_values(["license_number", "inspection_date", "inspection_id"]).reset_index(drop=True)
    grp = df.groupby("license_number")

    # --- Outcome indicators for the current row (used ONLY via past sums/shifts) ---
    df["is_fail"] = (df["results"] == "Fail").astype(int)
    df["is_cond"] = (df["results"] == "Pass w/ Conditions").astype(int)
    df["is_complaint"] = (df["inspection_category"] == "complaint").astype(int)
    df["is_routine"] = (df["inspection_category"] == "routine").astype(int)
    df["num_violations"] = df["num_violations"].fillna(0)

    # --- General history ---
    df["prior_inspections"] = grp.cumcount()
    df["prior_fails"] = past_cumsum(df, "is_fail")
    df["prior_conditions"] = past_cumsum(df, "is_cond")
    df["prior_complaints"] = past_cumsum(df, "is_complaint")
    with np.errstate(divide="ignore", invalid="ignore"):
        df["prior_fail_rate"] = df["prior_fails"] / df["prior_inspections"]
        df["prior_avg_violations"] = past_cumsum(df, "num_violations") / df["prior_inspections"]
    df.loc[df["prior_inspections"] == 0, ["prior_fail_rate", "prior_avg_violations"]] = np.nan

    df["prev_failed"] = grp["is_fail"].shift(1)
    df["prev_cond"] = grp["is_cond"].shift(1)
    df["prev_num_violations"] = grp["num_violations"].shift(1)
    df["prev_was_complaint"] = grp["is_complaint"].shift(1)
    df["days_since_prev"] = (df["inspection_date"] - grp["inspection_date"].shift(1)).dt.days
    df["days_since_first"] = (df["inspection_date"] - grp["inspection_date"].transform("min")).dt.days

    # --- Previous ROUTINE inspection (the fair baseline from the EDA) ---
    df["_routine_result"] = df["results"].where(df["is_routine"] == 1)
    df["_routine_date"] = df["inspection_date"].where(df["is_routine"] == 1)
    df["_routine_viol"] = df["num_violations"].where(df["is_routine"] == 1)
    last_routine = last_value_before(df, "_routine_result")
    df["last_routine_failed"] = (last_routine == "Fail").astype(float).where(last_routine.notna())
    df["last_routine_cond"] = (last_routine == "Pass w/ Conditions").astype(float).where(last_routine.notna())
    df["last_routine_violations"] = last_value_before(df, "_routine_viol")
    df["days_since_last_routine"] = (df["inspection_date"] - last_value_before(df, "_routine_date")).dt.days

    # --- Specific violation history (new food code numbering only) ---
    new_code = df["new_food_code"] == 1
    codes = df.loc[new_code, "violation_codes"].dropna().astype(str).str.split(";").explode()
    top_codes = codes[codes != ""].value_counts().head(N_TOP_VIOLATIONS).index.tolist()
    code_sets = df["violation_codes"].fillna("").astype(str).str.split(";").apply(set)
    for code in top_codes:
        col = f"_v{code}"
        df[col] = (code_sets.apply(lambda s, c=code: c in s) & new_code).astype(int)
        df[f"past_viol_{code}"] = past_cumsum(df, col)

    # --- Keywords in past inspector comments ---
    comments = df["inspector_comments"].fillna("").str.lower()
    for name, pattern in KEYWORDS.items():
        df[f"_{name}"] = comments.str.contains(pattern, regex=True).astype(int)
        df[f"past_{name}"] = past_cumsum(df, f"_{name}")

    # --- Keep the rows we want to predict ---
    model_rows = df[(df["is_routine"] == 1) & (df["inspection_date"] >= CODE_CHANGE_DATE)].copy()

    id_cols = ["inspection_id", "license_number", "dba_name", "inspection_date", "latitude", "longitude"]
    feature_cols = (
        ["risk_level", "month", "prior_inspections", "prior_fails", "prior_conditions",
         "prior_complaints", "prior_fail_rate", "prior_avg_violations",
         "prev_failed", "prev_cond", "prev_num_violations", "prev_was_complaint",
         "days_since_prev", "days_since_first",
         "last_routine_failed", "last_routine_cond", "last_routine_violations", "days_since_last_routine"]
        + [f"past_viol_{c}" for c in top_codes]
        + [f"past_{k}" for k in KEYWORDS]
    )
    out = model_rows[id_cols + feature_cols + ["failed"]]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Built {len(feature_cols)} features for {len(out):,} routine inspections "
          f"({out['inspection_date'].min():%b %Y} to {out['inspection_date'].max():%b %Y})")
    print(f"Failure rate in these rows: {out['failed'].mean():.1%}")
    print(f"Violation numbers tracked: {', '.join('#' + c for c in top_codes)}")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
