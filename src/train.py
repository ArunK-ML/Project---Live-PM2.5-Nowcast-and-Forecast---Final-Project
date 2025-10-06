import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd

from src.preprocessing import clean_data
from src.features import feature_pipeline
from src.anomaly import detect_anomalies

def train_model(data_path="data/processed/pm25_merged.csv", model_out="models/model.joblib"):
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    df = clean_data(df)
    df, _ = detect_anomalies(df)
    df = feature_pipeline(df)

    X = df.drop(columns=["timestamp", "pm25"])
    y = df["pm25"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    mae = mean_absolute_error(y_val, preds)
    print(f"Validation MAE: {mae:.3f}")

    joblib.dump(model, model_out)
    print(f"✅ Model saved to {model_out}")
    return model, mae
