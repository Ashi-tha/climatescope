"""
ClimateScope — Step 2: Data Understanding & Exploration
Inspects the dataset structure, identifies issues, and prints a quality report.
Run: python src/02_data_exploration.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR      = Path("data/raw")
REPORTS_DIR  = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# Expected key columns
KEY_VARS = [
    "temperature_celsius", "humidity",
    "precip_mm", "wind_kph",
    "cloud",     "uv_index",
    "pressure_mb", "visibility_km",
    "country", "location_name", "last_updated",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_raw(nrows: int | None = None) -> pd.DataFrame:
    csvs = list(RAW_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            "No CSV in data/raw/. Run 01_data_acquisition.py first."
        )
    return pd.read_csv(csvs[0], nrows=nrows, low_memory=False)


def print_section(title: str) -> None:
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print('═' * width)


# ── Exploration functions ─────────────────────────────────────────────────────

def schema_overview(df: pd.DataFrame) -> dict:
    print_section("1 · SCHEMA OVERVIEW")
    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {df.shape[1]}")
    print(f"\n  {'Column':<35} {'Dtype':<15} {'Non-null':>8}  {'Sample'}")
    print(f"  {'-'*35} {'-'*15} {'-'*8}  {'-'*20}")
    schema = {}
    for col in df.columns:
        nn    = df[col].notna().sum()
        dtype = str(df[col].dtype)
        samp  = str(df[col].dropna().iloc[0]) if nn > 0 else "—"
        print(f"  {col:<35} {dtype:<15} {nn:>8,}  {samp[:40]}")
        schema[col] = {"dtype": dtype, "non_null": int(nn)}
    return schema


def missing_values_report(df: pd.DataFrame) -> dict:
    print_section("2 · MISSING VALUES")
    total = len(df)
    missing = {}
    for col in df.columns:
        n_miss = df[col].isna().sum()
        if n_miss > 0:
            pct = n_miss / total * 100
            missing[col] = {"count": int(n_miss), "pct": round(pct, 2)}
            bar = "█" * int(pct / 2)
            print(f"  {col:<35} {n_miss:>6,}  ({pct:5.1f}%)  {bar}")
    if not missing:
        print("  ✅  No missing values found.")
    return missing


def numeric_summary(df: pd.DataFrame) -> dict:
    print_section("3 · NUMERIC SUMMARY")
    numerics = df.select_dtypes(include="number")
    summary  = {}
    cols_of_interest = [c for c in KEY_VARS if c in numerics.columns]
    desc = numerics[cols_of_interest].describe().round(2)
    print(desc.to_string())
    for col in cols_of_interest:
        summary[col] = desc[col].to_dict()
    return summary


def geographic_coverage(df: pd.DataFrame) -> dict:
    print_section("4 · GEOGRAPHIC COVERAGE")
    geo = {}
    if "country" in df.columns:
        countries = df["country"].nunique()
        top10     = df["country"].value_counts().head(10)
        print(f"  Unique countries : {countries}")
        print(f"\n  Top 10 by record count:")
        for country, cnt in top10.items():
            print(f"    {country:<30} {cnt:>6,}")
        geo["unique_countries"] = int(countries)
        geo["top_countries"]    = top10.to_dict()
    if "location_name" in df.columns:
        locs = df["location_name"].nunique()
        print(f"\n  Unique locations : {locs:,}")
        geo["unique_locations"] = int(locs)
    return geo


def temporal_coverage(df: pd.DataFrame) -> dict:
    print_section("5 · TEMPORAL COVERAGE")
    date_col = next(
        (c for c in df.columns if "date" in c.lower() or "updated" in c.lower()), None
    )
    temp = {}
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        valid  = df[date_col].dropna()
        start  = valid.min()
        end    = valid.max()
        span   = (end - start).days
        print(f"  Date column : {date_col}")
        print(f"  Earliest    : {start.date()}")
        print(f"  Latest      : {end.date()}")
        print(f"  Span        : {span} days")
        temp = {
            "column":  date_col,
            "start":   str(start.date()),
            "end":     str(end.date()),
            "span_days": span,
        }
    else:
        print("  ⚠️  No date column detected.")
    return temp


def detect_anomalies(df: pd.DataFrame) -> dict:
    """Flag rows where values fall outside physical plausibility."""
    print_section("6 · ANOMALY / OUTLIER FLAGS")
    bounds = {
        "temperature_celsius": (-90, 60),
        "humidity":            (0,  100),
        "precip_mm":           (0, 1000),
        "wind_kph":            (0,  500),
        "pressure_mb":        (870, 1085),
        "uv_index":            (0,   20),
    }
    report = {}
    for col, (lo, hi) in bounds.items():
        if col not in df.columns:
            continue
        mask  = (df[col] < lo) | (df[col] > hi)
        count = mask.sum()
        report[col] = {"out_of_range": int(count), "bounds": [lo, hi]}
        status = "✅" if count == 0 else "⚠️ "
        print(f"  {status}  {col:<30} [{lo:>6}, {hi:>6}]  → {count:>5} anomalies")
    return report


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n🌍  ClimateScope · Data Exploration Report")
    df = load_raw()

    report = {
        "total_rows":    len(df),
        "total_columns": df.shape[1],
    }
    report["schema"]      = schema_overview(df)
    report["missing"]     = missing_values_report(df)
    report["numerics"]    = numeric_summary(df)
    report["geography"]   = geographic_coverage(df)
    report["temporal"]    = temporal_coverage(df)
    report["anomalies"]   = detect_anomalies(df)

    # Save report
    out = REPORTS_DIR / "exploration_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n\n💾  Full report saved → {out}")


if __name__ == "__main__":
    main()
