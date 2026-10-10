"""
Step 1a: Download Chicago's Food Inspections dataset.

Source: Chicago Data Portal, "Food Inspections" (dataset id 4ijn-s7e5)
https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5

Run from the project folder:
    python src/download_data.py

The full dataset is a few hundred thousand rows, so this pulls it in pages.
It takes a few minutes. Re-running skips the download unless you pass --force.
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "food_inspections.csv"
URL = "https://data.cityofchicago.org/resource/4ijn-s7e5.csv"
PAGE_SIZE = 50_000


def fetch_page(offset: int, limit: int, token: str | None) -> pd.DataFrame:
    """Download one page of rows, retrying a few times if the request fails."""
    params = {"$limit": limit, "$offset": offset, "$order": "inspection_id"}
    headers = {"X-App-Token": token} if token else {}
    for attempt in range(5):
        try:
            resp = requests.get(URL, params=params, headers=headers, timeout=180)
            resp.raise_for_status()
            return pd.read_csv(io.StringIO(resp.text), dtype=str)
        except requests.RequestException as err:
            wait = 2 ** attempt
            print(f"  request failed ({err}); retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Could not download rows starting at offset {offset}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if the file exists")
    parser.add_argument("--token", default=None, help="optional Socrata app token (speeds up downloads)")
    args = parser.parse_args()

    if RAW_PATH.exists() and not args.force:
        print(f"{RAW_PATH} already exists. Use --force to download again.")
        return

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    pages, offset = [], 0
    while True:
        print(f"Downloading rows {offset:,} to {offset + PAGE_SIZE:,}...")
        page = fetch_page(offset, PAGE_SIZE, args.token)
        if page.empty:
            break
        pages.append(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    df = pd.concat(pages, ignore_index=True)
    df.to_csv(RAW_PATH, index=False)
    print(f"Saved {len(df):,} rows to {RAW_PATH}")


if __name__ == "__main__":
    main()
