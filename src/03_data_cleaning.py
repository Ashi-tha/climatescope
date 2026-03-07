"""
ClimateScope — Step 3: Data Cleaning & Preprocessing
Cleans raw data and produces processed datasets ready for analysis.
Run: python src/03_data_cleaning.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ── Physical bounds for clamping / flagging ───────────────────────────────────
BOUNDS = {
    "temperature_celsius":   (-90,   60),
    "temperature_fahrenheit":(-130, 140),
    "humidity":              (  0,  100),
    "precip_mm":             (  0, 1000),
    "wind_kph":              (  0,  500),
    "pressure_mb":           (870, 1085),
    "uv_index":              (  0,   20),
    "visibility_km":         (  0,  100),
    "cloud":                 (  0,  100),
}

# Columns that must exist for the row to be useful
REQUIRED_COLS = ["country", "last_updated", "temperature_celsius"]


def load_raw() -> pd.DataFrame:
    csvs = list(RAW_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError("No CSV in data/raw/. Run 01_data_acquisition.py first.")
    df = pd.read_csv(csvs[0], low_memory=False)
    print(f"📂  Loaded raw data  : {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


# ── Cleaning steps ─────────────────────────────────────────────────────────────

def drop_fully_empty(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(how="all")
    print(f"  [drop_fully_empty]       removed {before - len(df):,} fully-empty rows")
    return df


def enforce_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    existing_required = [c for c in REQUIRED_COLS if c in df.columns]
    df = df.dropna(subset=existing_required)
    print(f"  [enforce_required_cols]  removed {before - len(df):,} rows missing required fields")
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the date column and extract useful time features."""
    date_col = next(
        (c for c in df.columns if "updated" in c.lower() or "date" in c.lower()), None
    )
    if date_col:
        df[date_col]  = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df["date"]    = df[date_col].dt.date
        df["year"]    = df[date_col].dt.year
        df["month"]   = df[date_col].dt.month
        df["month_name"] = df[date_col].dt.strftime("%b")
        df["day_of_year"] = df[date_col].dt.dayofyear
        df["season"]  = df["month"].map({
            12: "Winter", 1: "Winter",  2: "Winter",
            3:  "Spring", 4: "Spring",  5: "Spring",
            6:  "Summer", 7: "Summer",  8: "Summer",
            9:  "Autumn", 10: "Autumn", 11: "Autumn",
        })
        print(f"  [parse_dates]            date col '{date_col}' → year/month/season added")
    return df


def remove_out_of_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Replace physically impossible values with NaN (don't drop rows)."""
    total_replaced = 0
    for col, (lo, hi) in BOUNDS.items():
        if col not in df.columns:
            continue
        mask = (df[col] < lo) | (df[col] > hi)
        n    = mask.sum()
        if n:
            df.loc[mask, col] = np.nan
            total_replaced += n
    print(f"  [remove_out_of_bounds]   replaced {total_replaced:,} impossible values with NaN")
    return df


def fill_missing_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each numeric column, fill NaN with the median of that country.
    Remaining NaN (whole country missing) filled with global median.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    filled = 0
    for col in numeric_cols:
        before = df[col].isna().sum()
        if before == 0:
            continue
        if "country" in df.columns:
            df[col] = df.groupby("country")[col].transform(
                lambda s: s.fillna(s.median())
            )
        df[col] = df[col].fillna(df[col].median())
        filled += before - df[col].isna().sum()
    print(f"  [fill_missing_numerics]  imputed {filled:,} NaN values (country-median → global-median)")
    return df


def standardise_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip and title-case string columns."""
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
    print(f"  [standardise_strings]    cleaned {len(str_cols)} string columns")
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create analysis-friendly derived columns."""
    # Heat index (simplified Steadman formula)
    if "temperature_celsius" in df.columns and "humidity" in df.columns:
        T = df["temperature_celsius"]
        H = df["humidity"]
        df["heat_index_celsius"] = (
            -8.78469475556
            + 1.61139411 * T
            + 2.33854883889 * H
            - 0.14611605 * T * H
            - 0.012308094 * T**2
            - 0.0164248277778 * H**2
            + 0.002211732 * T**2 * H
            + 0.00072546 * T * H**2
            - 0.000003582 * T**2 * H**2
        ).round(2)

    # Wind categories (Beaufort scale approximate)
    if "wind_kph" in df.columns:
        bins   = [0, 1, 6, 12, 20, 29, 39, 50, 62, 75, 89, 103, 118, np.inf]
        labels = [
            "Calm","Light air","Light breeze","Gentle breeze",
            "Moderate breeze","Fresh breeze","Strong breeze",
            "Near gale","Gale","Strong gale","Storm","Violent storm","Hurricane",
        ]
        df["wind_category"] = pd.cut(
            df["wind_kph"], bins=bins, labels=labels, right=False
        )

    # Precipitation category
    if "precip_mm" in df.columns:
        df["precip_category"] = pd.cut(
            df["precip_mm"],
            bins=[-0.001, 0, 2.5, 7.6, 50, np.inf],
            labels=["None", "Light", "Moderate", "Heavy", "Violent"],
        )

    print("  [add_derived_features]   heat_index, wind_category, precip_category added")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    subset = [c for c in ["location_name", "last_updated"] if c in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=subset if subset else None)
    print(f"  [remove_duplicates]      removed {before - len(df):,} duplicate rows")
    return df


# ── Aggregation ────────────────────────────────────────────────────────────────

AGG_METRICS = {
    "temperature_celsius": ["mean", "min", "max"],
    "humidity":            ["mean"],
    "precip_mm":           ["sum", "mean"],
    "wind_kph":            ["mean", "max"],
    "pressure_mb":         ["mean"],
    "uv_index":            ["mean", "max"],
    "heat_index_celsius":  ["mean"],
}


def build_monthly_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to (country, year, month) granularity."""
    if "country" not in df.columns or "year" not in df.columns:
        return pd.DataFrame()

    agg_spec = {
        col: funcs
        for col, funcs in AGG_METRICS.items()
        if col in df.columns
    }
    monthly = (
        df.groupby(["country", "year", "month", "month_name", "season"])
        .agg(agg_spec)
        .reset_index()
    )
    # Flatten multi-level columns
    monthly.columns = [
        "_".join(c).rstrip("_") if isinstance(c, tuple) else c
        for c in monthly.columns
    ]
    print(f"  [build_monthly_agg]      monthly agg : {monthly.shape[0]:,} rows")
    return monthly


def build_country_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to country level (all-time averages)."""
    if "country" not in df.columns:
        return pd.DataFrame()

    agg_spec = {
        col: ["mean"]
        for col, _ in AGG_METRICS.items()
        if col in df.columns
    }
    country = (
        df.groupby("country")
        .agg(agg_spec)
        .reset_index()
    )
    country.columns = [
        "_".join(c).rstrip("_") if isinstance(c, tuple) else c
        for c in country.columns
    ]
    print(f"  [build_country_agg]      country agg : {country.shape[0]:,} rows")
    return country


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n🌍  ClimateScope · Data Cleaning & Preprocessing\n")
    df = load_raw()

    print("\n── Cleaning pipeline ──────────────────────────────────────────")
    df = drop_fully_empty(df)
    df = enforce_required_columns(df)
    df = remove_duplicates(df)
    df = parse_dates(df)
    df = standardise_strings(df)
    df = remove_out_of_bounds(df)
    df = fill_missing_numerics(df)
    df = add_derived_features(df)

    print(f"\n✅  Clean dataset   : {df.shape[0]:,} rows × {df.shape[1]} cols")

    print("\n── Aggregating ────────────────────────────────────────────────")
    monthly = build_monthly_agg(df)
    country = build_country_agg(df)

    # ── Save outputs ──────────────────────────────────────────────────────────
    df.to_parquet(PROCESSED_DIR / "weather_clean.parquet", index=False)
    df.to_csv(PROCESSED_DIR / "weather_clean.csv", index=False)

    if not monthly.empty:
        monthly.to_parquet(PROCESSED_DIR / "weather_monthly.parquet", index=False)
        monthly.to_csv(PROCESSED_DIR / "weather_monthly.csv", index=False)

    if not country.empty:
        country.to_parquet(PROCESSED_DIR / "weather_country.parquet", index=False)
        country.to_csv(PROCESSED_DIR / "weather_country.csv", index=False)

    print(f"\n💾  Saved to data/processed/")
    print(f"    weather_clean.parquet   ({df.shape[0]:,} rows)")
    if not monthly.empty:
        print(f"    weather_monthly.parquet ({monthly.shape[0]:,} rows)")
    if not country.empty:
        print(f"    weather_country.parquet ({country.shape[0]:,} rows)")


if __name__ == "__main__":
    main()
