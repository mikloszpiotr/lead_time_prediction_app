
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def load_scaler(path='models/scaler.pkl'):
    return joblib.load(path)

def preprocess_input(df, scaler):
    X = df[["supplier_id", "product_category", "region", "historical_mean_lead_time", "macro_indicator"]]
    X_scaled = scaler.transform(X)
    return X_scaled
