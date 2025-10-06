import pandas as pd

def add_time_features(df):
    """Add time-based features like hour, dayofweek."""
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    return df

def add_lag_features(df, col="pm25", lags=6):
    """Create lag features for time series modeling."""
    for i in range(1, lags + 1):
        df[f"{col}_lag_{i}"] = df[col].shift(i)
    return df

def add_rolling_features(df, col="pm25", windows=[3,6,12]):
    """Add rolling mean and std features."""
    for w in windows:
        df[f"{col}_roll_mean_{w}"] = df[col].rolling(w).mean()
        df[f"{col}_roll_std_{w}"] = df[col].rolling(w).std()
    return df

def feature_pipeline(df):
    """Complete feature engineering pipeline."""
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = df.dropna().reset_index(drop=True)
    return df
