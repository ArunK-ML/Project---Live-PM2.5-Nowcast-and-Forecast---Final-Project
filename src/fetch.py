import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_openaq(city="Delhi", hours=72):
    """Fetch PM2.5 data from OpenAQ API."""
    end = datetime.utcnow()
    start = end - timedelta(hours=hours)
    url = f"https://api.openaq.org/v2/measurements"
    params = {
        "city": city,
        "parameter": "pm25",
        "date_from": start.isoformat(timespec="seconds") + "Z",
        "date_to": end.isoformat(timespec="seconds") + "Z",
        "limit": 10000,
        "sort": "desc"
    }
    r = requests.get(url, params=params)
    if r.status_code != 200:
        raise Exception(f"OpenAQ API failed: {r.status_code}")
    results = r.json().get("results", [])
    df = pd.DataFrame(results)
    if "date" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"].apply(lambda x: x["utc"]))
    df = df[["timestamp", "value"]].rename(columns={"value": "pm25"}).dropna()
    df.sort_values("timestamp", inplace=True)
    return df.reset_index(drop=True)

def fetch_weather(lat=28.6139, lon=77.2090):
    """Fetch hourly weather data from Open-Meteo API."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl",
        "timezone": "UTC"
    }
    r = requests.get(url, params=params)
    if r.status_code != 200:
        raise Exception("Weather API fetch failed")
    data = r.json()["hourly"]
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["time"])
    df.drop(columns=["time"], inplace=True)
    return df

def merge_datasets(pm_df, weather_df):
    """Merge PM2.5 with weather data on timestamp."""
    df = pd.merge_asof(pm_df.sort_values("timestamp"),
                       weather_df.sort_values("timestamp"),
                       on="timestamp")
    return df
