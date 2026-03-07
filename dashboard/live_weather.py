"""
ClimateScope — Live Weather Page (Open-Meteo)
Realtime + forecast data. No API key required.
Used inside dashboard/app.py as the '🌐 Live Weather' page.
"""

import time
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st
from plotly.subplots import make_subplots

# ── Shared plot theme (matches app.py) ───────────────────────────────────────
PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,43,78,0.4)",
    font=dict(family="DM Sans, sans-serif", color="#e8f4fd"),
    margin=dict(t=50, b=40, l=40, r=20),
)

# ── Default city list ─────────────────────────────────────────────────────────
DEFAULT_CITIES = [
    {"name": "Mumbai",        "lat": 19.076,  "lon": 72.878},
    {"name": "Delhi",         "lat": 28.614,  "lon": 77.209},
    {"name": "London",        "lat": 51.507,  "lon": -0.128},
    {"name": "New York",      "lat": 40.713,  "lon": -74.006},
    {"name": "Tokyo",         "lat": 35.689,  "lon": 139.692},
    {"name": "Sydney",        "lat": -33.869, "lon": 151.209},
    {"name": "Dubai",         "lat": 25.204,  "lon": 55.270},
    {"name": "Paris",         "lat": 48.857,  "lon": 2.347},
    {"name": "Singapore",     "lat": 1.352,   "lon": 103.820},
    {"name": "Cairo",         "lat": 30.033,  "lon": 31.233},
]

# WMO weather code → description + emoji
WMO_CODES = {
    0:  ("Clear sky",            "☀️"),
    1:  ("Mainly clear",         "🌤️"),
    2:  ("Partly cloudy",        "⛅"),
    3:  ("Overcast",             "☁️"),
    45: ("Foggy",                "🌫️"),
    48: ("Icy fog",              "🌫️"),
    51: ("Light drizzle",        "🌦️"),
    53: ("Moderate drizzle",     "🌦️"),
    55: ("Dense drizzle",        "🌧️"),
    61: ("Slight rain",          "🌧️"),
    63: ("Moderate rain",        "🌧️"),
    65: ("Heavy rain",           "🌧️"),
    71: ("Slight snow",          "🌨️"),
    73: ("Moderate snow",        "❄️"),
    75: ("Heavy snow",           "❄️"),
    80: ("Slight showers",       "🌦️"),
    81: ("Moderate showers",     "🌧️"),
    82: ("Violent showers",      "⛈️"),
    95: ("Thunderstorm",         "⛈️"),
    96: ("Thunderstorm + hail",  "⛈️"),
    99: ("Thunderstorm + hail",  "⛈️"),
}


# ── API calls ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)   # cache for 60 seconds = refresh every 1 minute
def fetch_current(lat: float, lon: float) -> dict | None:
    """Fetch current conditions for a single location."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":             lat,
        "longitude":            lon,
        "current":              [
            "temperature_2m", "relative_humidity_2m",
            "apparent_temperature", "weather_code",
            "wind_speed_10m", "wind_direction_10m",
            "precipitation", "surface_pressure",
            "cloud_cover", "uv_index",
        ],
        "wind_speed_unit":      "kmh",
        "timezone":             "auto",
    }
    try:
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=60)
def fetch_forecast(lat: float, lon: float) -> dict | None:
    """Fetch 7-day hourly forecast."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "hourly":     [
            "temperature_2m", "relative_humidity_2m",
            "precipitation_probability", "precipitation",
            "wind_speed_10m", "uv_index", "cloud_cover",
        ],
        "forecast_days": 7,
        "wind_speed_unit": "kmh",
        "timezone": "auto",
    }
    try:
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_all_cities(cities: list[dict]) -> list[dict]:
    """Fetch current conditions for all cities."""
    results = []
    for city in cities:
        data = fetch_current(city["lat"], city["lon"])
        if data and "current" in data:
            c = data["current"]
            code = c.get("weather_code", 0)
            desc, emoji = WMO_CODES.get(code, ("Unknown", "🌡️"))
            results.append({
                "city":        city["name"],
                "lat":         city["lat"],
                "lon":         city["lon"],
                "temp":        c.get("temperature_2m"),
                "feels_like":  c.get("apparent_temperature"),
                "humidity":    c.get("relative_humidity_2m"),
                "wind":        c.get("wind_speed_10m"),
                "wind_dir":    c.get("wind_direction_10m"),
                "precip":      c.get("precipitation"),
                "pressure":    c.get("surface_pressure"),
                "cloud":       c.get("cloud_cover"),
                "uv":          c.get("uv_index"),
                "condition":   desc,
                "emoji":       emoji,
                "code":        code,
            })
    return results


# ── Helper renderers ──────────────────────────────────────────────────────────

def wind_direction_label(degrees: float) -> str:
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    return dirs[round(degrees / 45) % 8] if degrees is not None else "—"


def uv_label(uv: float) -> tuple[str, str]:
    if uv is None:
        return "—", "#888"
    if uv <= 2:   return "Low",       "#2ecc87"
    if uv <= 5:   return "Moderate",  "#f5c842"
    if uv <= 7:   return "High",      "#ff9500"
    if uv <= 10:  return "Very High", "#ff6b35"
    return "Extreme", "#e74c3c"


def render_city_card(city_data: dict) -> None:
    uv_text, uv_color = uv_label(city_data["uv"])
    wind_lbl = wind_direction_label(city_data["wind_dir"])
    st.markdown(f"""
    <div style="
        background: rgba(13,43,78,0.55);
        border: 1px solid rgba(168,230,240,0.13);
        border-radius: 6px; padding: 20px 22px;
        transition: all 0.3s;
    ">
      <div style="display:flex; justify-content:space-between; align-items:flex-start;">
        <div>
          <div style="font-family:'Space Mono',monospace; font-size:10px;
                      letter-spacing:2px; text-transform:uppercase; color:#3a7bd5;">
            {city_data['city']}
          </div>
          <div style="font-family:'Bebas Neue',sans-serif; font-size:52px;
                      color:#e8f4fd; line-height:1; margin: 4px 0;">
            {city_data['temp']:.0f}°<span style="font-size:28px;color:#a8e6f0;">C</span>
          </div>
          <div style="font-size:13px; color:#b8cfe0;">
            {city_data['emoji']} {city_data['condition']}
          </div>
        </div>
        <div style="text-align:right; font-size:12px; color:#b8cfe0; line-height:2;">
          <div>Feels like <b style="color:#e8f4fd">{city_data['feels_like']:.0f}°C</b></div>
          <div>Humidity <b style="color:#e8f4fd">{city_data['humidity']}%</b></div>
          <div>Wind <b style="color:#e8f4fd">{city_data['wind']} kph {wind_lbl}</b></div>
          <div>Pressure <b style="color:#e8f4fd">{city_data['pressure']:.0f} mb</b></div>
          <div>UV <b style="color:{uv_color}">{city_data['uv']} — {uv_text}</b></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Main page renderer (called from app.py) ───────────────────────────────────

def render_live_weather_page() -> None:

    # ── Header ────────────────────────────────────────────────────────────────
    now = datetime.now(timezone.utc).strftime("%d %b %Y · %H:%M UTC")
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center;
                border-bottom:1px solid rgba(168,230,240,0.15); padding-bottom:16px; margin-bottom:24px;">
      <div>
        <div style="font-family:'Bebas Neue',sans-serif; font-size:32px;
                    letter-spacing:3px; color:#e8f4fd;">
          🌐 Live Weather
        </div>
        <div style="font-family:'Space Mono',monospace; font-size:10px;
                    letter-spacing:2px; color:#3a7bd5; text-transform:uppercase;">
          Powered by Open-Meteo · Refreshes every 60 seconds
        </div>
      </div>
      <div style="font-family:'Space Mono',monospace; font-size:11px; color:#b8cfe0;">
        🕐 {now}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── City selector ─────────────────────────────────────────────────────────
    with st.expander("⚙️  Manage Cities", expanded=False):
        st.markdown("**Add a custom city**")
        col_a, col_b, col_c, col_d = st.columns([2, 1, 1, 1])
        with col_a:
            new_name = st.text_input("City name", placeholder="e.g. Kozhikode")
        with col_b:
            new_lat  = st.number_input("Latitude",  value=11.25, format="%.3f")
        with col_c:
            new_lon  = st.number_input("Longitude", value=75.78, format="%.3f")
        with col_d:
            st.markdown("<br>", unsafe_allow_html=True)
            add_btn = st.button("➕ Add")

        if "live_cities" not in st.session_state:
            st.session_state.live_cities = DEFAULT_CITIES.copy()

        if add_btn and new_name:
            st.session_state.live_cities.append(
                {"name": new_name, "lat": new_lat, "lon": new_lon}
            )
            st.success(f"Added {new_name}!")
            st.cache_data.clear()

        st.markdown("**Active cities**")
        names = [c["name"] for c in st.session_state.live_cities]
        selected = st.multiselect("Remove cities by deselecting", names, default=names)
        st.session_state.live_cities = [
            c for c in st.session_state.live_cities if c["name"] in selected
        ]

    cities = st.session_state.get("live_cities", DEFAULT_CITIES)

    # ── Fetch all cities ──────────────────────────────────────────────────────
    with st.spinner("Fetching live weather data..."):
        all_data = fetch_all_cities(cities)

    if not all_data:
        st.error("Could not fetch weather data. Check your internet connection.")
        return

    df_live = pd.DataFrame(all_data)

    # ── Global snapshot KPIs ──────────────────────────────────────────────────
    st.markdown("**Global Snapshot**")
    k1, k2, k3, k4, k5 = st.columns(5)
    hottest  = df_live.loc[df_live["temp"].idxmax()]
    coldest  = df_live.loc[df_live["temp"].idxmin()]
    windiest = df_live.loc[df_live["wind"].idxmax()]

    for col, label, value, sub in [
        (k1, "Hottest",   f"{hottest['temp']:.0f}°C",  hottest['city']),
        (k2, "Coldest",   f"{coldest['temp']:.0f}°C",  coldest['city']),
        (k3, "Windiest",  f"{windiest['wind']:.0f} kph", windiest['city']),
        (k4, "Avg Temp",  f"{df_live['temp'].mean():.1f}°C", "across all cities"),
        (k5, "Avg Humid", f"{df_live['humidity'].mean():.0f}%", "across all cities"),
    ]:
        col.markdown(f"""
        <div style="background:rgba(13,43,78,0.55); border:1px solid rgba(168,230,240,0.12);
                    border-radius:4px; padding:16px; text-align:center;">
          <div style="font-family:'Bebas Neue',sans-serif; font-size:32px;
                      color:#a8e6f0; line-height:1;">{value}</div>
          <div style="font-family:'Space Mono',monospace; font-size:9px;
                      letter-spacing:2px; text-transform:uppercase; color:#b8cfe0;
                      margin-top:4px;">{label}</div>
          <div style="font-size:11px; color:#3a7bd5; margin-top:2px;">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── World map ─────────────────────────────────────────────────────────────
    st.markdown('<div style="font-family:\'Bebas Neue\',sans-serif; font-size:22px; '
                'letter-spacing:2px; color:#e8f4fd; border-left:3px solid #3a7bd5; '
                'padding-left:10px; margin:20px 0 12px;">Live Temperature Map</div>',
                unsafe_allow_html=True)

    fig_map = go.Figure(go.Scattergeo(
        lat=df_live["lat"],
        lon=df_live["lon"],
        text=df_live.apply(
            lambda r: f"{r['city']}<br>{r['emoji']} {r['condition']}<br>"
                      f"{r['temp']:.0f}°C · {r['humidity']}% humidity<br>"
                      f"Wind: {r['wind']:.0f} kph", axis=1
        ),
        mode="markers+text",
        textposition="top center",
        textfont=dict(size=10, color="#e8f4fd"),
        marker=dict(
            size=df_live["temp"].apply(lambda t: max(10, min(30, t + 10))),
            color=df_live["temp"],
            colorscale="RdYlBu_r",
            showscale=True,
            colorbar=dict(title="°C", thickness=12),
            line=dict(color="rgba(168,230,240,0.5)", width=1),
        ),
        hovertemplate="%{text}<extra></extra>",
    ))
    fig_map.update_layout(
        **PLOTLY_THEME, height=420,
        geo=dict(
            bgcolor="rgba(0,0,0,0)",
            showframe=False,
            showcoastlines=True,
            coastlinecolor="#1a3a5c",
            showland=True, landcolor="#0d2b4e",
            showocean=True, oceancolor="#060d18",
            showlakes=True, lakecolor="#060d18",
            showcountries=True, countrycolor="#1a3a5c",
            projection_type="natural earth",
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # ── City cards grid ───────────────────────────────────────────────────────
    st.markdown('<div style="font-family:\'Bebas Neue\',sans-serif; font-size:22px; '
                'letter-spacing:2px; color:#e8f4fd; border-left:3px solid #3a7bd5; '
                'padding-left:10px; margin:24px 0 16px;">City Conditions</div>',
                unsafe_allow_html=True)

    cols_per_row = 3
    rows = [all_data[i:i+cols_per_row] for i in range(0, len(all_data), cols_per_row)]
    for row in rows:
        cols = st.columns(cols_per_row)
        for col, city_data in zip(cols, row):
            with col:
                render_city_card(city_data)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Comparison bar chart ──────────────────────────────────────────────────
    st.markdown('<div style="font-family:\'Bebas Neue\',sans-serif; font-size:22px; '
                'letter-spacing:2px; color:#e8f4fd; border-left:3px solid #3a7bd5; '
                'padding-left:10px; margin:24px 0 16px;">City Comparison</div>',
                unsafe_allow_html=True)

    compare_var = st.selectbox(
        "Compare by",
        ["temp", "humidity", "wind", "pressure", "uv", "precip"],
        format_func=lambda x: {
            "temp": "Temperature (°C)", "humidity": "Humidity (%)",
            "wind": "Wind Speed (kph)", "pressure": "Pressure (mb)",
            "uv": "UV Index", "precip": "Precipitation (mm)",
        }[x],
    )
    df_sorted = df_live.sort_values(compare_var, ascending=True)
    fig_bar = go.Figure(go.Bar(
        x=df_sorted[compare_var],
        y=df_sorted["city"],
        orientation="h",
        marker=dict(
            color=df_sorted[compare_var],
            colorscale="RdYlBu_r" if compare_var == "temp" else "Blues",
            showscale=False,
        ),
        text=df_sorted[compare_var].round(1),
        textposition="outside",
        textfont=dict(color="#e8f4fd", size=11),
    ))
    fig_bar.update_layout(**PLOTLY_THEME, height=380, xaxis_title="", yaxis_title="")
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── 7-day forecast for selected city ─────────────────────────────────────
    st.markdown('<div style="font-family:\'Bebas Neue\',sans-serif; font-size:22px; '
                'letter-spacing:2px; color:#e8f4fd; border-left:3px solid #3a7bd5; '
                'padding-left:10px; margin:24px 0 16px;">7-Day Forecast</div>',
                unsafe_allow_html=True)

    forecast_city = st.selectbox(
        "Select city for forecast",
        [c["name"] for c in cities],
    )
    city_info = next(c for c in cities if c["name"] == forecast_city)

    with st.spinner(f"Loading forecast for {forecast_city}..."):
        fcast = fetch_forecast(city_info["lat"], city_info["lon"])

    if fcast and "hourly" in fcast:
        h = fcast["hourly"]
        df_fcast = pd.DataFrame({
            "time":        pd.to_datetime(h["time"]),
            "temp":        h["temperature_2m"],
            "humidity":    h["relative_humidity_2m"],
            "precip_prob": h["precipitation_probability"],
            "precip":      h["precipitation"],
            "wind":        h["wind_speed_10m"],
            "uv":          h["uv_index"],
            "cloud":       h["cloud_cover"],
        })

        fig_fc = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=("Temperature (°C)", "Precipitation Probability (%)", "Wind Speed (kph)"),
            vertical_spacing=0.08,
        )
        fig_fc.add_trace(go.Scatter(
            x=df_fcast["time"], y=df_fcast["temp"],
            mode="lines", line=dict(color="#a8e6f0", width=2),
            fill="tozeroy", fillcolor="rgba(168,230,240,0.08)",
            name="Temp",
        ), row=1, col=1)
        fig_fc.add_trace(go.Bar(
            x=df_fcast["time"], y=df_fcast["precip_prob"],
            marker_color="rgba(58,123,213,0.6)", name="Precip %",
        ), row=2, col=1)
        fig_fc.add_trace(go.Scatter(
            x=df_fcast["time"], y=df_fcast["wind"],
            mode="lines", line=dict(color="#2ecc87", width=1.5),
            name="Wind",
        ), row=3, col=1)
        fig_fc.update_layout(
            **PLOTLY_THEME, height=560,
            showlegend=False,
            title=f"7-Day Forecast — {forecast_city}",
        )
        st.plotly_chart(fig_fc, use_container_width=True)

    # ── Auto-refresh countdown ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div style="font-family:\'Space Mono\',monospace; font-size:10px; '
        'letter-spacing:2px; color:#3a7bd5; text-align:center;">'
        '🔄 DATA REFRESHES AUTOMATICALLY EVERY 60 SECONDS · OPEN-METEO API</div>',
        unsafe_allow_html=True,
    )
    time.sleep(1)
    st.rerun()
