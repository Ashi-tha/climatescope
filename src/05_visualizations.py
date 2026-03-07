"""
ClimateScope — Step 5: Visualization Builder
Creates all standalone Plotly charts and saves them as HTML files.
Run: python src/05_visualizations.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROCESSED_DIR = Path("data/processed")
VIZ_DIR       = Path("reports/charts")
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared theme ──────────────────────────────────────────────────────────────
THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#0a0f1a",
    plot_bgcolor="#0d2b4e",
    font=dict(family="DM Sans, sans-serif", color="#e8f4fd"),
    title_font=dict(size=18, color="#a8e6f0"),
)
COLOR_SEQ = px.colors.sequential.Blues_r + px.colors.sequential.Teal


def load(name: str) -> pd.DataFrame | None:
    p = PROCESSED_DIR / f"{name}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    c = PROCESSED_DIR / f"{name}.csv"
    if c.exists():
        return pd.read_csv(c, low_memory=False)
    return None


def save(fig: go.Figure, name: str) -> None:
    path = VIZ_DIR / f"{name}.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    print(f"  💾  {name}.html")


# ── 1. Choropleth — avg temperature by country ───────────────────────────────

def chart_choropleth_temp(df_country: pd.DataFrame) -> None:
    col = next(
        (c for c in df_country.columns if "temperature_celsius_mean" in c or c == "temperature_celsius"),
        None,
    )
    if col is None:
        print("  ⚠️  Skipping choropleth: no temperature column found.")
        return

    fig = px.choropleth(
        df_country,
        locations="country",
        locationmode="country names",
        color=col,
        color_continuous_scale="RdYlBu_r",
        title="🌡️  Average Temperature by Country (°C)",
        labels={col: "Avg Temp (°C)"},
    )
    fig.update_layout(**THEME, geo=dict(bgcolor="#0a0f1a", showframe=False))
    save(fig, "01_choropleth_temperature")


# ── 2. Choropleth — avg precipitation by country ─────────────────────────────

def chart_choropleth_precip(df_country: pd.DataFrame) -> None:
    col = next(
        (c for c in df_country.columns if "precip_mm" in c),
        None,
    )
    if col is None:
        return
    fig = px.choropleth(
        df_country,
        locations="country",
        locationmode="country names",
        color=col,
        color_continuous_scale="Blues",
        title="🌧️  Average Precipitation by Country (mm)",
        labels={col: "Avg Precip (mm)"},
    )
    fig.update_layout(**THEME, geo=dict(bgcolor="#0a0f1a", showframe=False))
    save(fig, "02_choropleth_precipitation")


# ── 3. Time-series — global avg temperature trend ────────────────────────────

def chart_temperature_trend(df: pd.DataFrame) -> None:
    if "year" not in df.columns or "temperature_celsius" not in df.columns:
        return
    yearly = df.groupby("year")["temperature_celsius"].agg(["mean","min","max"]).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly["year"], y=yearly["max"],
        fill=None, mode="lines", line=dict(width=0),
        showlegend=False, name="Max",
    ))
    fig.add_trace(go.Scatter(
        x=yearly["year"], y=yearly["min"],
        fill="tonexty", mode="lines", line=dict(width=0),
        fillcolor="rgba(58,123,213,0.2)", name="Min–Max Range",
    ))
    fig.add_trace(go.Scatter(
        x=yearly["year"], y=yearly["mean"],
        mode="lines+markers",
        line=dict(color="#a8e6f0", width=2.5),
        marker=dict(size=6, color="#f5c842"),
        name="Global Mean",
    ))
    fig.update_layout(
        **THEME,
        title="📈  Global Average Temperature Trend (Year-over-Year)",
        xaxis_title="Year", yaxis_title="Temperature (°C)",
    )
    save(fig, "03_timeseries_temperature_trend")


# ── 4. Seasonal heatmap — avg temp by month & country (top 20) ───────────────

def chart_seasonal_heatmap(df: pd.DataFrame) -> None:
    if not all(c in df.columns for c in ["month", "country", "temperature_celsius"]):
        return
    top_countries = df["country"].value_counts().head(20).index.tolist()
    pivot = (
        df[df["country"].isin(top_countries)]
        .groupby(["country", "month"])["temperature_celsius"]
        .mean()
        .unstack("month")
        .round(1)
    )
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns),
        y=list(pivot.index),
        colorscale="RdYlBu_r",
        colorbar=dict(title="°C"),
        hoverongaps=False,
    ))
    fig.update_layout(
        **THEME,
        title="🗓️  Seasonal Temperature Heatmap (°C) — Top 20 Countries",
        xaxis_title="Month", yaxis_title="Country",
        height=600,
    )
    save(fig, "04_heatmap_seasonal_temperature")


# ── 5. Scatter — temperature vs humidity, coloured by country ────────────────

def chart_scatter_temp_humidity(df: pd.DataFrame) -> None:
    if not all(c in df.columns for c in ["temperature_celsius", "humidity", "country"]):
        return
    sample = df.dropna(subset=["temperature_celsius","humidity"]).sample(
        min(5000, len(df)), random_state=42
    )
    fig = px.scatter(
        sample,
        x="temperature_celsius", y="humidity",
        color="country",
        opacity=0.6, size_max=6,
        title="🔵  Temperature vs Humidity by Country",
        labels={"temperature_celsius": "Temperature (°C)", "humidity": "Humidity (%)"},
        trendline="ols",
        trendline_scope="overall",
    )
    fig.update_layout(**THEME)
    save(fig, "05_scatter_temp_humidity")


# ── 6. Scatter — wind speed vs pressure ──────────────────────────────────────

def chart_scatter_wind_pressure(df: pd.DataFrame) -> None:
    if not all(c in df.columns for c in ["wind_kph", "pressure_mb"]):
        return
    sample = df.dropna(subset=["wind_kph","pressure_mb"]).sample(
        min(5000, len(df)), random_state=42
    )
    colour_col = "temperature_celsius" if "temperature_celsius" in df.columns else None
    fig = px.scatter(
        sample,
        x="pressure_mb", y="wind_kph",
        color=colour_col,
        color_continuous_scale="RdYlBu_r",
        opacity=0.5,
        title="💨  Wind Speed vs Atmospheric Pressure",
        labels={"wind_kph": "Wind Speed (kph)", "pressure_mb": "Pressure (mb)"},
    )
    fig.update_layout(**THEME)
    save(fig, "06_scatter_wind_pressure")


# ── 7. Bar — top 15 windiest countries ───────────────────────────────────────

def chart_windiest_countries(df_country: pd.DataFrame) -> None:
    col = next((c for c in df_country.columns if "wind_kph" in c), None)
    if col is None:
        return
    top = df_country.nlargest(15, col).sort_values(col)
    fig = go.Figure(go.Bar(
        x=top[col], y=top["country"],
        orientation="h",
        marker=dict(color=top[col], colorscale="Blues", showscale=False),
    ))
    fig.update_layout(
        **THEME,
        title="💨  Top 15 Windiest Countries (avg kph)",
        xaxis_title="Avg Wind Speed (kph)", yaxis_title="",
        height=500,
    )
    save(fig, "07_bar_windiest_countries")


# ── 8. Precipitation distribution — box plot by season ───────────────────────

def chart_precip_by_season(df: pd.DataFrame) -> None:
    if not all(c in df.columns for c in ["season", "precip_mm"]):
        return
    order = ["Spring","Summer","Autumn","Winter"]
    df_plot = df[df["season"].isin(order)].copy()
    fig = px.box(
        df_plot, x="season", y="precip_mm",
        category_orders={"season": order},
        color="season",
        color_discrete_sequence=["#2ecc87","#f5c842","#ff6b35","#a8e6f0"],
        title="🌧️  Precipitation Distribution by Season",
        labels={"precip_mm": "Precipitation (mm)", "season": "Season"},
    )
    fig.update_layout(**THEME)
    save(fig, "08_boxplot_precip_by_season")


# ── 9. Extreme events — top 20 hottest locations ─────────────────────────────

def chart_hottest_locations(df: pd.DataFrame) -> None:
    if "temperature_celsius" not in df.columns:
        return
    id_cols = [c for c in ["location_name","country"] if c in df.columns]
    hottest = df.nlargest(20, "temperature_celsius")[id_cols + ["temperature_celsius"]]
    label   = hottest.get("location_name", hottest.get("country", pd.Series(range(20))))
    fig = go.Figure(go.Bar(
        x=hottest["temperature_celsius"].values,
        y=(hottest["location_name"] if "location_name" in hottest.columns else hottest["country"]).values,
        orientation="h",
        marker=dict(color=hottest["temperature_celsius"].values, colorscale="Reds", showscale=True),
    ))
    fig.update_layout(
        **THEME,
        title="🔥  Top 20 Hottest Recorded Locations",
        xaxis_title="Temperature (°C)", height=550,
    )
    save(fig, "09_bar_hottest_locations")


# ── 10. Multi-panel overview dashboard ───────────────────────────────────────

def chart_dashboard_overview(df: pd.DataFrame, df_country: pd.DataFrame) -> None:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Temperature Distribution",
            "Humidity Distribution",
            "Precipitation Distribution",
            "Wind Speed Distribution",
        ),
    )
    pairs = [
        ("temperature_celsius", "#a8e6f0", 1, 1),
        ("humidity",            "#f5c842", 1, 2),
        ("precip_mm",           "#3a7bd5", 2, 1),
        ("wind_kph",            "#2ecc87", 2, 2),
    ]
    for col, colour, row, c in pairs:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        fig.add_trace(
            go.Histogram(x=vals, marker_color=colour, opacity=0.75,
                         name=col.replace("_"," ").title(), nbinsx=60),
            row=row, col=c,
        )
    fig.update_layout(
        **THEME,
        title="📊  Global Weather Variable Distributions",
        showlegend=False, height=600,
    )
    save(fig, "10_overview_distributions")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n🌍  ClimateScope · Building Visualizations\n")
    df         = load("weather_clean")
    df_country = load("weather_country")

    if df is None:
        raise FileNotFoundError("Run 03_data_cleaning.py first.")

    chart_temperature_trend(df)
    chart_seasonal_heatmap(df)
    chart_scatter_temp_humidity(df)
    chart_scatter_wind_pressure(df)
    chart_precip_by_season(df)
    chart_hottest_locations(df)
    chart_dashboard_overview(df, df_country)

    if df_country is not None:
        chart_choropleth_temp(df_country)
        chart_choropleth_precip(df_country)
        chart_windiest_countries(df_country)

    print(f"\n✅  All charts saved to reports/charts/")


if __name__ == "__main__":
    main()
