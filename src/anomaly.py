import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_anomalies(df, col="pm25"):
    """Detect anomalies using IsolationForest."""
    model = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly_flag"] = model.fit_predict(df[[col]])
    df["anomaly_flag"] = df["anomaly_flag"].map({1: 0, -1: 1})
    return df, model
