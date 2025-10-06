import numpy as np
import pandas as pd

def recursive_forecast(model, initial_window, steps, feature_fn=None):
    """
    Perform recursive forecasting for `steps` ahead.
    - model: trained regressor
    - initial_window: recent feature vector (DataFrame)
    - feature_fn: optional function to rebuild features each step
    """
    preds = []
    window = initial_window.copy()
    for _ in range(steps):
        X_last = window.iloc[[-1]].drop(columns=["timestamp", "pm25"])
        y_pred = model.predict(X_last)[0]
        preds.append(y_pred)
        next_row = window.iloc[[-1]].copy()
        next_row["pm25"] = y_pred
        next_row["timestamp"] += pd.Timedelta(hours=1)
        window = pd.concat([window, next_row]).reset_index(drop=True)
        if feature_fn:
            window = feature_fn(window)
    forecast_df = pd.DataFrame({
        "timestamp": pd.date_range(window["timestamp"].iloc[-steps-1], periods=steps+1, freq="H")[1:],
        "forecast": preds
    })
    return forecast_df
