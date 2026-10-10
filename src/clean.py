"""
Step 1b: Clean the raw inspections data.

Run from the project folder (after download_data.py):
    python src/clean.py

What it does:
  - standardizes column names (works with the API download OR a manual CSV
    export from the data portal website)
  - keeps restaurants only, and only real outcomes (Pass / Pass w/ Conditions / Fail)
  - groups the messy inspection types into a few categories
  - parses the violations text into a count and a list of violation numbers
  - adds each restaurant's PREVIOUS inspection result (using only earlier
    inspections, so there's no data leakage)

Output: data/processed/inspections_clean.csv
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "food_inspections.csv"
OUT_PATH = ROOT / "data" / "processed" / "inspections_clean.csv"

# Chicago switched to a new food code (with renumbered violations) on July 1, 2018.
# Violation numbers before and after this date mean different things.
CODE_CHANGE_DATE = pd.Timestamp("2018-07-01")

KEEP_RESULTS = ["Pass", "Pass w/ Conditions", "Fail"]
VIOLATION_RE = re.compile(r"^\s*(\d+)\.\s*(.*?)(?:\s*-\s*Comments:\s*(.*))?$", re.DOTALL)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make column names lowercase_with_underscores, whatever the download source."""
    cols = (
        df.columns.str.strip()
        .str.lower()
        .str.replace("#", "", regex=False)
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    df.columns = cols
    return df.rename(columns={"license": "license_number"})


def categorize_inspection(raw_type: str) -> str:
    """Collapse dozens of inspection type spellings into a few categories."""
    t = str(raw_type).lower()
    if "re-insp" in t or "reinsp" in t or "re insp" in t:
        return "re-inspection"
    if "canvass" in t:
        return "routine"
    if "complaint" in t or "poison" in t:
        return "complaint"
    if "license" in t:
        return "license"
    return "other"


def parse_violations(text: str) -> tuple[int, str, str]:
    """Return (number of violations, ';'-joined violation numbers, all inspector comments)."""
    if not isinstance(text, str) or not text.strip():
        return 0, "", ""
    codes, comments = [], []
    for piece in text.split("|"):
        match = VIOLATION_RE.match(piece.strip())
        if match:
            codes.append(match.group(1))
            if match.group(3):
                comments.append(match.group(3).strip())
    return len(codes), ";".join(codes), " ".join(comments)


def main() -> None:
    df = standardize_columns(pd.read_csv(RAW_PATH, dtype=str))
    n_raw = len(df)

    # Basic types
    df = df.drop_duplicates(subset="inspection_id")
    df["inspection_date"] = pd.to_datetime(df["inspection_date"], format="mixed", errors="coerce")
    df["license_number"] = pd.to_numeric(df["license_number"], errors="coerce")
    for col in ["latitude", "longitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["inspection_date", "license_number"])
    df = df[df["license_number"] > 0]
    df["license_number"] = df["license_number"].astype("int64")

    # Restaurants only, real outcomes only
    df = df[df["facility_type"].fillna("").str.lower().str.contains("restaurant")]
    df["results"] = df["results"].str.strip()
    df = df[df["results"].isin(KEEP_RESULTS)].copy()
    df["failed"] = (df["results"] == "Fail").astype(int)

    # Inspection category and risk level (1 = high, 3 = low)
    df["inspection_category"] = df["inspection_type"].apply(categorize_inspection)
    df["risk_level"] = pd.to_numeric(
        df["risk"].fillna("").str.extract(r"Risk\s*(\d)", expand=False), errors="coerce"
    )

    # Violations
    parsed = df["violations"].apply(parse_violations)
    df["num_violations"] = parsed.str[0]
    df["violation_codes"] = parsed.str[1]
    df["inspector_comments"] = parsed.str[2]
    df["new_food_code"] = (df["inspection_date"] >= CODE_CHANGE_DATE).astype(int)

    # Time features
    df["year"] = df["inspection_date"].dt.year
    df["month"] = df["inspection_date"].dt.month

    # Previous inspection for the same restaurant (earlier rows only -> no leakage)
    df = df.sort_values(["license_number", "inspection_date", "inspection_id"])
    grp = df.groupby("license_number")
    df["prev_result"] = grp["results"].shift(1)
    df["prev_failed"] = grp["failed"].shift(1)
    df["days_since_prev"] = (df["inspection_date"] - grp["inspection_date"].shift(1)).dt.days
    df["prior_inspections"] = grp.cumcount()

    keep = [
        "inspection_id", "license_number", "dba_name", "aka_name", "facility_type",
        "risk_level", "address", "zip", "latitude", "longitude",
        "inspection_date", "year", "month", "inspection_type", "inspection_category",
        "results", "failed", "num_violations", "violation_codes", "new_food_code",
        "prev_result", "prev_failed", "days_since_prev", "prior_inspections",
        "inspector_comments", "violations",
    ]
    df = df[[c for c in keep if c in df.columns]]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Raw rows: {n_raw:,}")
    print(f"Clean restaurant inspections: {len(df):,}")
    print(f"Unique restaurants: {df['license_number'].nunique():,}")
    print(f"Date range: {df['inspection_date'].min():%Y-%m-%d} to {df['inspection_date'].max():%Y-%m-%d}")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
