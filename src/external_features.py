"""
Step 3b: Add outside-data features to each inspection.

Run from the project folder (after features.py and download_external.py):
    python src/external_features.py

For every routine inspection, counts 311 complaints NEAR the restaurant in the
days BEFORE the inspection (never on or after the inspection date):
  - rodent complaints within 400 m (about a quarter mile): past 30, 90, 365 days
  - sanitation complaints within 400 m: past 90, 365 days
  - restaurant complaints within 50 m (basically the same address): past 90, 365 days
And adds the weather around the inspection date.

If the 311 data doesn't reach far enough back to fill a window (the earliest
inspections), that value is left blank instead of being counted as zero.

Output: data/processed/features_plus.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
REQ_PATH = ROOT / "data" / "raw" / "311_requests.csv"
WEATHER_PATH = ROOT / "data" / "raw" / "weather_chicago.csv"
OUT_PATH = ROOT / "data" / "processed" / "features_plus.csv"

EARTH_RADIUS_M = 6_371_000
# (category, radius in meters, look-back windows in days)
SPECS = [
    ("rodent", 400, [30, 90, 365]),
    ("sanitation", 400, [90, 365]),
    ("restaurant", 50, [90, 365]),
]


def categorize(sr_type: str) -> str | None:
    t = str(sr_type).upper()
    if "RODENT" in t or "RAT " in t:
        return "rodent"
    if "RESTAURANT" in t:
        return "restaurant"
    if "SANITATION" in t:
        return "sanitation"
    return None


def to_days(dates: pd.Series) -> np.ndarray:
    """Dates -> whole days since 1970, for fast comparisons."""
    return dates.dt.normalize().to_numpy().astype("datetime64[D]").astype(np.int64)


def nearby_counts(insp: pd.DataFrame, req: pd.DataFrame, radius_m: int, windows: list[int]) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=insp.index, columns=windows)
    has_loc = insp["latitude"].notna() & insp["longitude"].notna()
    if req.empty or not has_loc.any():
        return out

    tree = BallTree(np.radians(req[["latitude", "longitude"]].to_numpy()), metric="haversine")
    req_days = to_days(req["created_date"])
    data_start = req_days.min()

    loc = insp[has_loc]
    insp_days = to_days(loc["inspection_date"])
    neighbors = tree.query_radius(np.radians(loc[["latitude", "longitude"]].to_numpy()),
                                  r=radius_m / EARTH_RADIUS_M)
    values = np.full((len(loc), len(windows)), np.nan)
    for i, (idx, day) in enumerate(zip(neighbors, insp_days)):
        d = req_days[idx]
        before = d[d < day]  # strictly before the inspection date
        for j, w in enumerate(windows):
            if day - w >= data_start:
                values[i, j] = np.count_nonzero(before >= day - w)
    out.loc[loc.index] = values
    return out


def main() -> None:
    feat = pd.read_csv(FEATURES_PATH, parse_dates=["inspection_date"])
    n_before = feat.shape[1]

    # 311 complaints
    req = pd.read_csv(REQ_PATH, dtype=str)
    req["created_date"] = pd.to_datetime(req["created_date"], format="mixed", errors="coerce")
    for col in ["latitude", "longitude"]:
        req[col] = pd.to_numeric(req[col], errors="coerce")
    req = req.dropna(subset=["created_date", "latitude", "longitude"]).drop_duplicates("sr_number")
    req["category"] = req["sr_type"].apply(categorize)
    print("311 requests used:", req["category"].value_counts().to_dict())
    print(f"311 data covers {req['created_date'].min():%b %d, %Y} to {req['created_date'].max():%b %d, %Y}")

    for category, radius, windows in SPECS:
        print(f"Counting {category} complaints within {radius} m...")
        counts = nearby_counts(feat, req[req["category"] == category], radius, windows)
        for w in windows:
            feat[f"n311_{category}_{radius}m_{w}d"] = counts[w]

    # Weather
    weather = pd.read_csv(WEATHER_PATH, parse_dates=["date"]).sort_values("date")
    weather["temp_max_7d_avg"] = weather["temp_max_f"].rolling(7, min_periods=4).mean()
    weather["precip_7d_total"] = weather["precip_in"].rolling(7, min_periods=4).sum()
    feat["_date"] = feat["inspection_date"].dt.normalize()
    feat = feat.merge(weather.rename(columns={"date": "_date"})[
        ["_date", "temp_max_f", "temp_max_7d_avg", "precip_7d_total"]], on="_date", how="left")
    feat = feat.drop(columns="_date")

    # Keep the target column last
    cols = [c for c in feat.columns if c != "failed"] + ["failed"]
    feat = feat[cols]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(OUT_PATH, index=False)

    new_cols = [c for c in feat.columns if c.startswith("n311_") or c.startswith(("temp_", "precip_"))]
    print(f"\nAdded {feat.shape[1] - n_before} features. Average values:")
    print(feat[new_cols].mean().round(2).to_string())
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
