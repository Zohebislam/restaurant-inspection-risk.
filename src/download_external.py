"""
Step 3a: Download outside data that might signal food safety problems.

Run from the project folder:
    python src/download_external.py

Downloads:
  1. Chicago 311 service requests about rodents, sanitation, and restaurants
     (Chicago Data Portal, "311 Service Requests", dataset id v6vf-nfxy)
     -> data/raw/311_requests.csv
  2. Daily Chicago weather (Open-Meteo historical weather API, free, no key)
     -> data/raw/weather_chicago.csv

The 311 data downloads one month at a time. If it stops partway (a slow
server, closed laptop), just run it again: finished months are skipped.
Re-running skips files that already exist unless you pass --force.
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REQ_PATH = RAW_DIR / "311_requests.csv"
WEATHER_PATH = RAW_DIR / "weather_chicago.csv"

SR_URL = "https://data.cityofchicago.org/resource/v6vf-nfxy.csv"
SR_TYPES = (
    "upper(sr_type) like '%RODENT%' OR "
    "upper(sr_type) like '%SANITATION%' OR "
    "upper(sr_type) like '%RESTAURANT%'"
)
SR_SELECT = "sr_number, sr_type, created_date, latitude, longitude"
SR_START = pd.Timestamp("2018-01-01")
PARTS_DIR = RAW_DIR / "311_parts"  # one file per month, so a stopped download can resume
PAGE_SIZE = 25_000

WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
CHICAGO_LAT, CHICAGO_LON = 41.88, -87.63


def get_with_retries(url: str, params: dict, timeout: int = 120) -> requests.Response:
    for attempt in range(5):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException as err:
            wait = 2 ** attempt
            print(f"  request failed ({err}); retrying in {wait}s...")
            time.sleep(wait)
            continue
        if resp.status_code == 400:
            raise SystemExit(f"The server rejected the request:\n{resp.text[:1000]}\n"
                             "Copy this message to Claude so the query can be fixed.")
        if resp.ok:
            return resp
        wait = 2 ** attempt
        print(f"  server returned {resp.status_code}; retrying in {wait}s...")
        time.sleep(wait)
    raise SystemExit(f"Could not download from {url} after several tries. Check your internet and run again.")


def download_month(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Download one month of matching requests (small, fast queries)."""
    where = (f"created_date >= '{start:%Y-%m-%dT00:00:00}' AND "
             f"created_date < '{end:%Y-%m-%dT00:00:00}' AND ({SR_TYPES})")
    pages, offset = [], 0
    while True:
        params = {"$select": SR_SELECT, "$where": where, "$order": "sr_number",
                  "$limit": PAGE_SIZE, "$offset": offset}
        page = pd.read_csv(io.StringIO(get_with_retries(SR_URL, params).text), dtype=str)
        if not page.empty:
            pages.append(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    cols = SR_SELECT.replace(" ", "").split(",")
    return pd.concat(pages, ignore_index=True) if pages else pd.DataFrame(columns=cols)


def download_311() -> None:
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    this_month = pd.Timestamp.today().normalize().replace(day=1)
    months = pd.date_range(SR_START, this_month, freq="MS")
    for start in months:
        end = start + pd.offsets.MonthBegin(1)
        part = PARTS_DIR / f"{start:%Y-%m}.csv"
        # Always re-download the current month, since it is still filling in
        if part.exists() and start < this_month:
            continue
        print(f"311 requests: {start:%b %Y}...", end=" ", flush=True)
        df = download_month(start, end)
        df.to_csv(part, index=False)
        print(f"{len(df):,} rows")

    parts = [pd.read_csv(p, dtype=str) for p in sorted(PARTS_DIR.glob("*.csv"))]
    df = pd.concat(parts, ignore_index=True).drop_duplicates("sr_number")
    if df.empty:
        raise SystemExit("No 311 requests came back. Copy this message to Claude.")
    df.to_csv(REQ_PATH, index=False)
    print(f"Saved {len(df):,} requests to {REQ_PATH}")
    print("Request types found:")
    print(df["sr_type"].value_counts().to_string())


def download_weather() -> None:
    end = (pd.Timestamp.today().normalize() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    params = {"latitude": CHICAGO_LAT, "longitude": CHICAGO_LON, "start_date": "2018-01-01",
              "end_date": end, "daily": "temperature_2m_max,precipitation_sum",
              "temperature_unit": "fahrenheit", "precipitation_unit": "inch",
              "timezone": "America/Chicago"}
    print("Weather: downloading daily Chicago weather...")
    daily = get_with_retries(WEATHER_URL, params, timeout=120).json()["daily"]
    df = pd.DataFrame({"date": daily["time"], "temp_max_f": daily["temperature_2m_max"],
                       "precip_in": daily["precipitation_sum"]})
    df.to_csv(WEATHER_PATH, index=False)
    print(f"Saved {len(df):,} days of weather to {WEATHER_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = parser.parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.force and PARTS_DIR.exists():
        for part in PARTS_DIR.glob("*.csv"):
            part.unlink()

    if REQ_PATH.exists() and not args.force:
        print(f"{REQ_PATH.name} already exists (use --force to download again).")
    else:
        download_311()

    if WEATHER_PATH.exists() and not args.force:
        print(f"{WEATHER_PATH.name} already exists (use --force to download again).")
    else:
        download_weather()


if __name__ == "__main__":
    main()
