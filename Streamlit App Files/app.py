import streamlit as st
import plotly.express as px

from modules.fetch import fetch_openaq, fetch_weather, merge_datasets
from modules.preprocessing import clean_data
from modules.features import feature_pipeline
from modules.anomaly import detect_anomalies
from modules.train import train_model
from modules.forecasting import recursive_forecast

st.set_page_config("EcoPulse – PM2.5", "🌫️", layout="wide")
st.title("🌫️ EcoPulse – PM2.5 Forecasting Dashboard")

# Sidebar
city = st.sidebar.text_input("City", "Delhi")
lat = st.sidebar.number_input("Latitude", 28.6139)
lon = st.sidebar.number_input("Longitude", 77.2090)
horizon = st.sidebar.slider("Forecast Hours", 12, 48, 24)

# Fetch Data
pm_df = fetch_openaq(city, 72, lat, lon)
weather_df = fetch_weather(lat, lon)
df = merge_datasets(pm_df, weather_df)

df = clean_data(df)
df, _ = detect_anomalies(df)

# Current PM2.5
st.metric("Current PM2.5", f"{df.iloc[-1]['pm25']:.1f}")

# Historical Plot
fig = px.line(df.tail(48), x="timestamp", y="pm25", title="Last 48 Hours PM2.5")
st.plotly_chart(fig, use_container_width=True)

# Feature Engineering & Model Training
df_feat = feature_pipeline(df)
model = train_model(df_feat)

# Forecast
future = recursive_forecast(model, df_feat, horizon)

fig2 = px.line(future, x="timestamp", y="forecast", title=f"Next {horizon} Hours Forecast")
st.plotly_chart(fig2, use_container_width=True)
