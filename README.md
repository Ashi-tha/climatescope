<<<<<<< HEAD
# 🌍 ClimateScope
### Visualizing Global Weather Trends and Extreme Events

An end-to-end Python data pipeline and interactive Streamlit dashboard for exploring the [Global Weather Repository](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository) dataset from Kaggle.

---

## 📁 Project Structure

```
climatescope/
├── src/
│   ├── 01_data_acquisition.py    # Download dataset via Kaggle API
│   ├── 02_data_exploration.py    # Schema, missing values, quality report
│   ├── 03_data_cleaning.py       # Clean, preprocess & aggregate
│   ├── 04_data_analysis.py       # Stats, correlations, extremes, trends
│   └── 05_visualizations.py      # Build standalone Plotly HTML charts
├── dashboard/
│   └── app.py                    # Streamlit interactive dashboard
├── data/
│   ├── raw/                      # Raw CSV from Kaggle
│   └── processed/                # Cleaned parquet/CSV files
├── reports/
│   ├── charts/                   # Standalone HTML charts
│   ├── exploration_report.json   # Output of step 2
│   └── analysis_report.json      # Output of step 4
├── run_pipeline.py               # Run all steps in sequence
├── requirements.txt
└── .env.example
```

---

## ⚡ Quick Start

### 1. Clone & install dependencies

```bash
git clone <your-repo-url>
cd climatescope
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Kaggle credentials

```bash
cp .env.example .env
# Edit .env and fill in your KAGGLE_USERNAME and KAGGLE_KEY
# Get your key at: https://www.kaggle.com/settings → API → Create New Token
```

### 3. Run the full pipeline

```bash
python run_pipeline.py
```

Or run individual steps:

```bash
python src/01_data_acquisition.py    # Download data
python src/02_data_exploration.py    # Explore & report
python src/03_data_cleaning.py       # Clean & preprocess
python src/04_data_analysis.py       # Statistical analysis
python src/05_visualizations.py      # Build charts
```

### 4. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Then open **http://localhost:8501** in your browser.

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Overview | KPI cards, variable distributions, seasonal breakdown |
| 🗺️ World Map | Choropleth maps + scatter-geo for any variable |
| 📈 Trends | Time-series by month/year/season, per-country comparison |
| 🔥 Extremes | Top N hottest/coldest/windiest + z-score anomaly detection |
| 🔬 Correlations | Correlation matrix heatmap + interactive scatter explorer |
| 📋 Data Explorer | Searchable table with CSV export |

All pages respect the **sidebar filters**: country selection and year range.

---

## 🗂️ Data Pipeline

```
Kaggle API
    ↓
01_data_acquisition.py   →  data/raw/GlobalWeatherRepository.csv
    ↓
02_data_exploration.py   →  reports/exploration_report.json
    ↓
03_data_cleaning.py      →  data/processed/weather_clean.parquet
                         →  data/processed/weather_monthly.parquet
                         →  data/processed/weather_country.parquet
    ↓
04_data_analysis.py      →  reports/analysis_report.json
    ↓
05_visualizations.py     →  reports/charts/*.html
    ↓
dashboard/app.py         →  Interactive Streamlit app
```

---

## 🔧 Tech Stack

| Layer | Tool |
|-------|------|
| Language | Python 3.10+ |
| Data | pandas, numpy |
| Statistics | scipy |
| Visualization | Plotly, Streamlit |
| Maps | Plotly choropleth, folium (optional) |
| Data source | Kaggle API |
| Storage | CSV + Parquet |
| Config | python-dotenv |

---

## 📈 Key Analyses

- **Distributions**: histogram + skewness/kurtosis for all weather variables
- **Correlations**: Pearson r matrix; temperature↔humidity, wind↔pressure
- **Seasonal patterns**: variable averages broken down by season and month
- **Extreme events**: top N records + z-score anomaly detection per country
- **Regional comparison**: country-level rankings for all variables
- **Year-over-year trends**: linear regression with significance testing

---

## 🚀 Future Enhancements

- [ ] Live weather API integration (OpenWeatherMap / WeatherAPI)
- [ ] Predictive modeling for temperature forecasting (Prophet / ARIMA)
- [ ] Anomaly alert system with email/Slack notifications
- [ ] Folium-based interactive map with clustering
- [ ] Docker containerization for easy deployment
- [ ] Automated daily pipeline via cron / GitHub Actions

---

## 📄 Data Source

**Global Weather Repository** — Kaggle  
https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository  
Daily-updated worldwide weather data: temperature, humidity, precipitation, wind speed, UV index, pressure, visibility, and more.

---

# climatescope
Global weather visualization dashboard
<img width="958" height="471" alt="image" src="https://github.com/user-attachments/assets/0d53d8a9-8d6e-4780-bc48-655f1cf6d418" />
