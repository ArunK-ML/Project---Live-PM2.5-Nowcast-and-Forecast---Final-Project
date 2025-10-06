import pandas as pd
import numpy as np

def clean_data(df):
    """Clean and preprocess merged dataset."""
    df = df.copy()
    df = df.ffill().bfill()  # fill missing values
    df = df.drop_duplicates(subset=["timestamp"])
    df = df.set_index("timestamp").resample("1H").mean().interpolate()
    return df.reset_index()

def remove_outliers(df, cols=None, z_thresh=3):
    """Remove statistical outliers using z-score."""
    if cols is None:
        cols = [c for c in df.columns if c != "timestamp"]
    for c in cols:
        z = np.abs((df[c] - df[c].mean()) / df[c].std())
        df.loc[z > z_thresh, c] = np.nan
    return df.ffill().bfill()

def save_processed(df, path="data/processed/pm25_merged.csv"):
    """Save cleaned data to CSV."""
    df.to_csv(path, index=False)
