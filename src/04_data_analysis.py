"""
ClimateScope — Step 4: Data Analysis
Statistical analysis, seasonal patterns, extreme events, correlations.
Run: python src/04_data_analysis.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR   = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

WEATHER_VARS = [
    "temperature_celsius", "humidity", "precip_mm",
    "wind_kph", "pressure_mb", "uv_index",
]


def load_clean() -> pd.DataFrame:
    p = PROCESSED_DIR / "weather_clean.parquet"
    if p.exists():
        return pd.read_parquet(p)
    c = PROCESSED_DIR / "weather_clean.csv"
    if c.exists():
        return pd.read_csv(c, low_memory=False)
    raise FileNotFoundError("Run 03_data_cleaning.py first.")


def section(title: str) -> None:
    print(f"\n{'═'*60}\n  {title}\n{'═'*60}")


# ── 1. Distribution analysis ──────────────────────────────────────────────────

def distribution_analysis(df: pd.DataFrame) -> dict:
    section("1 · DISTRIBUTIONS")
    cols   = [c for c in WEATHER_VARS if c in df.columns]
    result = {}
    print(f"  {'Variable':<30} {'Mean':>8} {'Std':>8} {'Skew':>7} {'Kurt':>7}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*7} {'-'*7}")
    for col in cols:
        s    = df[col].dropna()
        mean = s.mean()
        std  = s.std()
        skew = s.skew()
        kurt = s.kurtosis()
        print(f"  {col:<30} {mean:>8.2f} {std:>8.2f} {skew:>7.2f} {kurt:>7.2f}")
        result[col] = {"mean": round(mean,3), "std": round(std,3),
                       "skew": round(skew,3), "kurtosis": round(kurt,3)}
    return result


# ── 2. Correlation matrix ─────────────────────────────────────────────────────

def correlation_analysis(df: pd.DataFrame) -> dict:
    section("2 · CORRELATIONS")
    cols  = [c for c in WEATHER_VARS if c in df.columns]
    corr  = df[cols].corr().round(3)
    print(corr.to_string())
    # Find top correlations
    pairs = []
    for i, c1 in enumerate(cols):
        for c2 in cols[i+1:]:
            r = corr.loc[c1, c2]
            pairs.append({"var1": c1, "var2": c2, "r": round(float(r), 3)})
    pairs.sort(key=lambda x: abs(x["r"]), reverse=True)
    print(f"\n  Top 5 correlations:")
    for p in pairs[:5]:
        sign = "+" if p["r"] > 0 else ""
        print(f"    {p['var1']:<30} ↔  {p['var2']:<25} r = {sign}{p['r']}")
    return {"matrix": corr.to_dict(), "top_pairs": pairs[:10]}


# ── 3. Seasonal patterns ──────────────────────────────────────────────────────

def seasonal_analysis(df: pd.DataFrame) -> dict:
    section("3 · SEASONAL PATTERNS")
    if "season" not in df.columns or "month" not in df.columns:
        print("  ⚠️  No season/month columns. Run cleaning first.")
        return {}
    result = {}
    order  = ["Spring", "Summer", "Autumn", "Winter"]
    for col in [c for c in ["temperature_celsius", "precip_mm", "humidity"] if c in df.columns]:
        seasonal = df.groupby("season")[col].agg(["mean","std","min","max"]).round(2)
        seasonal = seasonal.reindex([s for s in order if s in seasonal.index])
        print(f"\n  {col}:")
        print(seasonal.to_string())
        result[col] = seasonal.to_dict()

    # Month-by-month temperature trend
    if "temperature_celsius" in df.columns:
        monthly_temp = df.groupby("month")["temperature_celsius"].mean().round(2)
        peak_month   = int(monthly_temp.idxmax())
        trough_month = int(monthly_temp.idxmin())
        print(f"\n  Peak temp month   : {peak_month} ({monthly_temp[peak_month]:.1f}°C avg)")
        print(f"  Trough temp month : {trough_month} ({monthly_temp[trough_month]:.1f}°C avg)")
        result["monthly_temp"] = monthly_temp.to_dict()
    return result


# ── 4. Extreme weather events ─────────────────────────────────────────────────

def extreme_events(df: pd.DataFrame) -> dict:
    section("4 · EXTREME WEATHER EVENTS")
    extremes = {}
    id_cols  = [c for c in ["country", "location_name", "date", "last_updated"] if c in df.columns]

    def show_extremes(col: str, label: str, n: int = 10) -> list:
        if col not in df.columns:
            return []
        top = df.nlargest(n, col)[id_cols + [col]]
        print(f"\n  🌡️  {label} (top {n}):")
        print(top.to_string(index=False))
        return top.to_dict(orient="records")

    extremes["hottest"]    = show_extremes("temperature_celsius",  "Hottest")
    extremes["coldest"]    = (
        df.nsmallest(10, "temperature_celsius")[id_cols + ["temperature_celsius"]].to_dict(orient="records")
        if "temperature_celsius" in df.columns else []
    )
    extremes["wettest"]    = show_extremes("precip_mm",            "Highest Precipitation")
    extremes["windiest"]   = show_extremes("wind_kph",             "Highest Wind Speed")
    extremes["highest_uv"] = show_extremes("uv_index",             "Highest UV Index")

    # z-score anomaly detection on temperature
    if "temperature_celsius" in df.columns and "country" in df.columns:
        df["temp_zscore"] = df.groupby("country")["temperature_celsius"].transform(
            lambda x: stats.zscore(x, nan_policy="omit")
        )
        anomalies = df[df["temp_zscore"].abs() > 3][id_cols + ["temperature_celsius", "temp_zscore"]]
        n_anom    = len(anomalies)
        print(f"\n  🔴  Temperature anomalies (|z| > 3) : {n_anom:,} events")
        if n_anom:
            print(anomalies.head(10).to_string(index=False))
        extremes["temp_anomaly_count"] = int(n_anom)

    return extremes


# ── 5. Regional comparison ────────────────────────────────────────────────────

def regional_comparison(df: pd.DataFrame) -> dict:
    section("5 · REGIONAL / COUNTRY COMPARISON")
    if "country" not in df.columns:
        return {}
    metrics = [c for c in ["temperature_celsius", "precip_mm", "wind_kph", "humidity"] if c in df.columns]
    cdf     = df.groupby("country")[metrics].mean().round(2)

    result = {}
    for col in metrics:
        if col not in cdf.columns:
            continue
        top5    = cdf[col].nlargest(5)
        bottom5 = cdf[col].nsmallest(5)
        print(f"\n  {col}:")
        print(f"    Highest: {', '.join(f'{c} ({v:.1f})' for c, v in top5.items())}")
        print(f"    Lowest : {', '.join(f'{c} ({v:.1f})' for c, v in bottom5.items())}")
        result[col] = {
            "highest": top5.to_dict(),
            "lowest":  bottom5.to_dict(),
        }
    return result


# ── 6. Trend analysis (year over year) ───────────────────────────────────────

def trend_analysis(df: pd.DataFrame) -> dict:
    section("6 · YEAR-OVER-YEAR TRENDS")
    if "year" not in df.columns:
        return {}
    result = {}
    for col in [c for c in ["temperature_celsius", "precip_mm"] if c in df.columns]:
        yearly = df.groupby("year")[col].mean().dropna()
        if len(yearly) < 2:
            continue
        slope, intercept, r, p, se = stats.linregress(yearly.index, yearly.values)
        trend_dir = "↑ warming" if slope > 0 else "↓ cooling"
        sig       = "significant" if p < 0.05 else "not significant"
        print(f"\n  {col}:")
        print(f"    Slope  : {slope:+.4f} per year  ({trend_dir})")
        print(f"    R²     : {r**2:.3f}   p={p:.4f}  ({sig})")
        result[col] = {
            "slope": round(float(slope), 5),
            "r2":    round(float(r**2), 4),
            "p":     round(float(p), 4),
            "significant": p < 0.05,
        }
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n🌍  ClimateScope · Data Analysis")
    df = load_clean()

    report = {}
    report["distributions"] = distribution_analysis(df)
    report["correlations"]  = correlation_analysis(df)
    report["seasonal"]      = seasonal_analysis(df)
    report["extremes"]      = extreme_events(df)
    report["regional"]      = regional_comparison(df)
    report["trends"]        = trend_analysis(df)

    out = REPORTS_DIR / "analysis_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n\n💾  Full analysis saved → {out}")


if __name__ == "__main__":
    main()
