"""
ClimateScope — Streamlit Dashboard
Interactive global weather visualization platform.
Run: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from live_weather import render_live_weather_page

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClimateScope",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&family=Space+Mono&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .main { background: #0a0f1a; }

  /* Header */
  .cs-header {
    display: flex; align-items: baseline; gap: 12px;
    padding: 8px 0 24px;
    border-bottom: 1px solid rgba(168,230,240,0.15);
    margin-bottom: 28px;
  }
  .cs-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 48px; letter-spacing: 4px;
    color: #e8f4fd; margin: 0;
  }
  .cs-title span { color: #a8e6f0; }
  .cs-sub {
    font-family: 'Space Mono', monospace;
    font-size: 11px; letter-spacing: 2px; text-transform: uppercase;
    color: #3a7bd5; margin: 0;
  }

  /* KPI cards */
  .kpi-card {
    background: rgba(13,43,78,0.5);
    border: 1px solid rgba(168,230,240,0.12);
    border-radius: 4px; padding: 20px 24px; text-align: center;
  }
  .kpi-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 40px; color: #a8e6f0; line-height: 1;
  }
  .kpi-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px; letter-spacing: 2px; text-transform: uppercase;
    color: #b8cfe0; margin-top: 6px;
  }
  .kpi-delta { font-size: 12px; margin-top: 4px; }

  /* Section headers */
  .section-hdr {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 28px; letter-spacing: 2px; color: #e8f4fd;
    border-left: 3px solid #3a7bd5; padding-left: 12px;
    margin: 32px 0 16px;
  }

  div[data-testid="stSidebar"] { background: #0d1a2e; }
  div[data-testid="stSidebar"] .css-1d391kg { padding: 16px; }
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,43,78,0.4)",
    font=dict(family="DM Sans, sans-serif", color="#e8f4fd"),
    margin=dict(t=50, b=40, l=40, r=20),
)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = Path("data/processed")

    def read(name: str) -> pd.DataFrame | None:
        for ext in ("parquet", "csv"):
            p = base / f"{name}.{ext}"
            if p.exists():
                return pd.read_parquet(p) if ext == "parquet" else pd.read_csv(p, low_memory=False)
        return None

    df         = read("weather_clean")
    df_monthly = read("weather_monthly")
    df_country = read("weather_country")
    if df is None:
        st.error("❌  No processed data found. Run the pipeline first: `python src/03_data_cleaning.py`")
        st.stop()
    return df, df_monthly, df_country


df, df_monthly, df_country = load_data()

WEATHER_VARS = {
    "temperature_celsius": "Temperature (°C)",
    "humidity":            "Humidity (%)",
    "precip_mm":           "Precipitation (mm)",
    "wind_kph":            "Wind Speed (kph)",
    "pressure_mb":         "Pressure (mb)",
    "uv_index":            "UV Index",
    "heat_index_celsius":  "Heat Index (°C)",
}
AVAILABLE_VARS = {k: v for k, v in WEATHER_VARS.items() if k in df.columns}
ALL_COUNTRIES  = sorted(df["country"].dropna().unique().tolist()) if "country" in df.columns else []
ALL_YEARS      = sorted(df["year"].dropna().unique().astype(int).tolist()) if "year" in df.columns else []


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🌍 ClimateScope")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Overview", "🗺️ World Map", "📈 Trends", "🔥 Extremes", "🔬 Correlations", "📋 Data Explorer", "🌐 Live Weather"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Filters**")

    sel_countries = st.multiselect(
        "Countries",
        ALL_COUNTRIES,
        default=ALL_COUNTRIES[:10] if len(ALL_COUNTRIES) >= 10 else ALL_COUNTRIES,
        help="Leave empty to include all",
    )

    if ALL_YEARS and int(min(ALL_YEARS)) != int(max(ALL_YEARS)):
        year_range = st.slider("Year range", int(min(ALL_YEARS)), int(max(ALL_YEARS)),
                               (int(min(ALL_YEARS)), int(max(ALL_YEARS))))
    else:
        year_range = (int(ALL_YEARS[0]), int(ALL_YEARS[0])) if ALL_YEARS else (0, 9999)
        st.caption(f"📅 Year: {ALL_YEARS[0] if ALL_YEARS else 'N/A'}")

    sel_var = st.selectbox(
        "Primary variable",
        list(AVAILABLE_VARS.keys()),
        format_func=lambda k: AVAILABLE_VARS[k],
    )

    st.markdown("---")
    st.markdown(
        '<span style="font-family:Space Mono;font-size:10px;color:#3a7bd5;">'
        'DATA: Global Weather Repository<br>SOURCE: Kaggle · Daily updated</span>',
        unsafe_allow_html=True,
    )


# ── Filter data ───────────────────────────────────────────────────────────────

@st.cache_data
def filter_df(countries: list, yr_min: int, yr_max: int) -> pd.DataFrame:
    mask = pd.Series([True] * len(df))
    if countries:
        mask &= df["country"].isin(countries)
    if "year" in df.columns:
        mask &= df["year"].between(yr_min, yr_max)
    return df[mask]


fdf = filter_df(sel_countries, year_range[0], year_range[1])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.markdown("""
    <div class="cs-header">
      <h1 class="cs-title">Climate<span>Scope</span></h1>
      <p class="cs-sub">Global Weather Visualization Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-value">{len(fdf):,}</div>
          <div class="kpi-label">Records</div></div>""", unsafe_allow_html=True)
    with c2:
        n_countries = fdf["country"].nunique() if "country" in fdf.columns else "—"
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-value">{n_countries}</div>
          <div class="kpi-label">Countries</div></div>""", unsafe_allow_html=True)
    with c3:
        avg_temp = f"{fdf['temperature_celsius'].mean():.1f}°" if "temperature_celsius" in fdf.columns else "—"
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-value">{avg_temp}</div>
          <div class="kpi-label">Avg Temp</div></div>""", unsafe_allow_html=True)
    with c4:
        max_wind = f"{fdf['wind_kph'].max():.0f}" if "wind_kph" in fdf.columns else "—"
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-value">{max_wind}</div>
          <div class="kpi-label">Max Wind kph</div></div>""", unsafe_allow_html=True)
    with c5:
        max_precip = f"{fdf['precip_mm'].max():.0f}" if "precip_mm" in fdf.columns else "—"
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-value">{max_precip}</div>
          <div class="kpi-label">Max Precip mm</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">Variable Distributions</div>', unsafe_allow_html=True)

    cols_to_show = list(AVAILABLE_VARS.keys())[:4]
    subplot_titles = [AVAILABLE_VARS[c] for c in cols_to_show]
    fig = make_subplots(rows=1, cols=len(cols_to_show), subplot_titles=subplot_titles)
    colours = ["#a8e6f0", "#f5c842", "#3a7bd5", "#2ecc87"]
    for i, (col, colour) in enumerate(zip(cols_to_show, colours), 1):
        fig.add_trace(
            go.Histogram(x=fdf[col].dropna(), marker_color=colour,
                         opacity=0.8, nbinsx=50, showlegend=False),
            row=1, col=i,
        )
    fig.update_layout(**PLOTLY_THEME, height=300, title="")
    st.plotly_chart(fig, use_container_width=True)

    # Seasonal breakdown
    if "season" in fdf.columns and "temperature_celsius" in fdf.columns:
        st.markdown('<div class="section-hdr">Seasonal Overview</div>', unsafe_allow_html=True)
        order = ["Spring", "Summer", "Autumn", "Winter"]
        seasonal = fdf.groupby("season")["temperature_celsius"].agg(["mean","min","max"]).reindex(
            [s for s in order if s in fdf["season"].unique()]
        ).reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=seasonal["season"], y=seasonal["mean"],
                              name="Mean", marker_color="#a8e6f0"))
        fig2.add_trace(go.Scatter(x=seasonal["season"], y=seasonal["max"],
                                  name="Max", mode="lines+markers",
                                  line=dict(color="#ff6b35", width=2)))
        fig2.add_trace(go.Scatter(x=seasonal["season"], y=seasonal["min"],
                                  name="Min", mode="lines+markers",
                                  line=dict(color="#3a7bd5", width=2)))
        fig2.update_layout(**PLOTLY_THEME, height=350,
                           yaxis_title="Temperature (°C)", xaxis_title="Season")
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: WORLD MAP
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🗺️ World Map":
    st.markdown('<div class="section-hdr">🗺️ Global Choropleth Map</div>', unsafe_allow_html=True)

    if df_country is None:
        st.warning("Country-level aggregation not found. Run `python src/03_data_cleaning.py`.")
    else:
        map_var = st.selectbox(
            "Variable to map",
            list(AVAILABLE_VARS.keys()),
            format_func=lambda k: AVAILABLE_VARS[k],
        )
        col = next((c for c in df_country.columns if map_var in c), None)
        if col:
            scale_map = {
                "temperature_celsius": "RdYlBu_r",
                "precip_mm":           "Blues",
                "wind_kph":            "Purples",
                "humidity":            "Teal",
                "uv_index":            "YlOrRd",
            }
            scale = scale_map.get(map_var, "Viridis")
            fig = px.choropleth(
                df_country,
                locations="country",
                locationmode="country names",
                color=col,
                color_continuous_scale=scale,
                title=f"Global {AVAILABLE_VARS[map_var]}",
                labels={col: AVAILABLE_VARS[map_var]},
            )
            fig.update_layout(**PLOTLY_THEME, height=550,
                              geo=dict(bgcolor="rgba(0,0,0,0)", showframe=False,
                                       showcoastlines=True, coastlinecolor="#3a7bd5"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No aggregated column found for {map_var}.")

    # Scatter-geo of individual readings
    if "latitude" in df.columns and "longitude" in df.columns and sel_var in df.columns:
        st.markdown("**Individual Station Readings**")
        sample = fdf.dropna(subset=["latitude","longitude",sel_var]).sample(
            min(2000, len(fdf)), random_state=42
        )
        fig2 = px.scatter_geo(
            sample, lat="latitude", lon="longitude",
            color=sel_var, size_max=6, opacity=0.7,
            color_continuous_scale="RdYlBu_r",
            hover_name="location_name" if "location_name" in sample.columns else None,
            title=f"Station-level {AVAILABLE_VARS.get(sel_var, sel_var)}",
        )
        fig2.update_layout(**PLOTLY_THEME, height=450,
                           geo=dict(bgcolor="rgba(0,0,0,0)", showframe=False))
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TRENDS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Trends":
    st.markdown('<div class="section-hdr">📈 Time-Series Trends</div>', unsafe_allow_html=True)

    trend_var  = st.selectbox("Variable", list(AVAILABLE_VARS.keys()),
                              format_func=lambda k: AVAILABLE_VARS[k])
    group_by   = st.radio("Granularity", ["Monthly", "Yearly", "Seasonal"], horizontal=True)
    compare_countries = st.multiselect("Compare countries", ALL_COUNTRIES[:5],
                                       default=ALL_COUNTRIES[:3] if len(ALL_COUNTRIES) >= 3 else ALL_COUNTRIES)

    if trend_var in fdf.columns:
        if group_by == "Yearly" and "year" in fdf.columns:
            gdf = fdf.groupby(["year","country"])[trend_var].mean().reset_index() \
                      if "country" in fdf.columns else \
                      fdf.groupby("year")[trend_var].mean().reset_index()
            xcol = "year"
        elif group_by == "Monthly" and "month" in fdf.columns:
            gdf = fdf.groupby(["month","country"])[trend_var].mean().reset_index() \
                      if "country" in fdf.columns else \
                      fdf.groupby("month")[trend_var].mean().reset_index()
            xcol = "month"
        else:  # Seasonal
            gdf  = fdf.groupby(["season","country"])[trend_var].mean().reset_index() \
                       if "country" in fdf.columns else \
                       fdf.groupby("season")[trend_var].mean().reset_index()
            xcol = "season"

        if compare_countries and "country" in gdf.columns:
            gdf = gdf[gdf["country"].isin(compare_countries)]
            fig = px.line(gdf, x=xcol, y=trend_var, color="country",
                          markers=True,
                          title=f"{AVAILABLE_VARS[trend_var]} — {group_by} Trend",
                          labels={trend_var: AVAILABLE_VARS[trend_var]})
        else:
            fig = px.line(gdf, x=xcol, y=trend_var, markers=True,
                          title=f"Global {AVAILABLE_VARS[trend_var]} — {group_by} Trend",
                          labels={trend_var: AVAILABLE_VARS[trend_var]})

        fig.update_layout(**PLOTLY_THEME, height=420)
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap
    if "month" in fdf.columns and "country" in fdf.columns and trend_var in fdf.columns:
        st.markdown("**Heatmap: Monthly × Country**")
        top_c = fdf["country"].value_counts().head(20).index.tolist()
        pivot = (
            fdf[fdf["country"].isin(top_c)]
            .groupby(["country","month"])[trend_var]
            .mean().unstack("month").round(2)
        )
        if not pivot.empty:
            fig2 = go.Figure(go.Heatmap(
                z=pivot.values, x=[str(m) for m in pivot.columns],
                y=list(pivot.index), colorscale="RdYlBu_r",
                colorbar=dict(title=AVAILABLE_VARS[trend_var]),
            ))
            fig2.update_layout(**PLOTLY_THEME, height=500,
                               xaxis_title="Month", yaxis_title="Country")
            st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXTREMES
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔥 Extremes":
    st.markdown('<div class="section-hdr">🔥 Extreme Weather Events</div>', unsafe_allow_html=True)

    ext_var = st.selectbox("Variable", list(AVAILABLE_VARS.keys()),
                           format_func=lambda k: AVAILABLE_VARS[k])
    n_top   = st.slider("Number of records", 5, 50, 20)

    id_cols = [c for c in ["location_name","country","date","last_updated"] if c in fdf.columns]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**🔺 Top {n_top} Highest**")
        top = fdf.nlargest(n_top, ext_var)[id_cols + [ext_var]].reset_index(drop=True)
        fig = go.Figure(go.Bar(
            x=top[ext_var].values,
            y=(top["location_name"] if "location_name" in top.columns else top.index.astype(str)).values,
            orientation="h",
            marker=dict(color=top[ext_var].values, colorscale="Reds"),
        ))
        fig.update_layout(**PLOTLY_THEME, height=450,
                          xaxis_title=AVAILABLE_VARS[ext_var])
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(top, use_container_width=True, height=200)

    with col2:
        st.markdown(f"**🔻 Top {n_top} Lowest**")
        bot = fdf.nsmallest(n_top, ext_var)[id_cols + [ext_var]].reset_index(drop=True)
        fig2 = go.Figure(go.Bar(
            x=bot[ext_var].values,
            y=(bot["location_name"] if "location_name" in bot.columns else bot.index.astype(str)).values,
            orientation="h",
            marker=dict(color=bot[ext_var].values, colorscale="Blues_r"),
        ))
        fig2.update_layout(**PLOTLY_THEME, height=450,
                           xaxis_title=AVAILABLE_VARS[ext_var])
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(bot, use_container_width=True, height=200)

    # Anomaly detection with z-scores
    if ext_var in fdf.columns and "country" in fdf.columns:
        st.markdown("**🚨 Statistical Anomalies (|z-score| > 2.5)**")
        from scipy import stats as scipy_stats
        adf = fdf.dropna(subset=[ext_var]).copy()
        adf["zscore"] = adf.groupby("country")[ext_var].transform(
            lambda x: scipy_stats.zscore(x, nan_policy="omit")
        )
        anomalies = adf[adf["zscore"].abs() > 2.5].nlargest(50, "zscore")
        if len(anomalies):
            fig3 = px.scatter(
                anomalies,
                x=ext_var, y="zscore",
                color="country",
                hover_data=id_cols,
                title=f"Anomalous {AVAILABLE_VARS[ext_var]} readings (z > 2.5σ)",
            )
            fig3.add_hline(y=2.5, line_dash="dash", line_color="#ff6b35")
            fig3.add_hline(y=-2.5, line_dash="dash", line_color="#3a7bd5")
            fig3.update_layout(**PLOTLY_THEME, height=380)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No anomalies found in current selection.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔬 Correlations":
    st.markdown('<div class="section-hdr">🔬 Correlation Analysis</div>', unsafe_allow_html=True)

    num_cols = [c for c in AVAILABLE_VARS if c in fdf.columns]

    # Correlation matrix heatmap
    st.markdown("**Correlation Matrix**")
    corr = fdf[num_cols].corr().round(3)
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale="RdBu",
        zmid=0,
        text=corr.values.round(2),
        texttemplate="%{text}",
        colorbar=dict(title="r"),
    ))
    fig.update_layout(**PLOTLY_THEME, height=450,
                      xaxis=dict(tickangle=-30))
    st.plotly_chart(fig, use_container_width=True)

    # Scatter with trendline
    st.markdown("**Scatterplot Explorer**")
    sc1, sc2 = st.columns(2)
    with sc1:
        x_var = st.selectbox("X axis", num_cols, index=0, format_func=lambda k: AVAILABLE_VARS.get(k, k))
    with sc2:
        y_var = st.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1),
                             format_func=lambda k: AVAILABLE_VARS.get(k, k))

    sample = fdf.dropna(subset=[x_var, y_var]).sample(min(4000, len(fdf)), random_state=42)
    colour_col = "country" if "country" in sample.columns else None
    fig2 = px.scatter(
        sample, x=x_var, y=y_var,
        color=colour_col, opacity=0.6,
        trendline="ols", trendline_scope="overall",
        trendline_color_override="#ff6b35",
        labels={x_var: AVAILABLE_VARS.get(x_var, x_var),
                y_var: AVAILABLE_VARS.get(y_var, y_var)},
        title=f"{AVAILABLE_VARS.get(x_var, x_var)} vs {AVAILABLE_VARS.get(y_var, y_var)}",
    )
    fig2.update_layout(**PLOTLY_THEME, height=450)
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📋 Data Explorer":
    st.markdown('<div class="section-hdr">📋 Raw Data Explorer</div>', unsafe_allow_html=True)

    st.markdown(f"**{len(fdf):,} records** matching current filters")

    search = st.text_input("Search location / country", "")
    if search:
        mask = pd.Series([False] * len(fdf))
        for col in ["country", "location_name"]:
            if col in fdf.columns:
                mask |= fdf[col].str.contains(search, case=False, na=False)
        fdf = fdf[mask]

    st.dataframe(fdf.head(500), use_container_width=True, height=450)

    st.download_button(
        "⬇️  Download filtered data (CSV)",
        data=fdf.to_csv(index=False).encode(),
        file_name="climatescope_filtered.csv",
        mime="text/csv",
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE WEATHER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🌐 Live Weather":
    render_live_weather_page()
