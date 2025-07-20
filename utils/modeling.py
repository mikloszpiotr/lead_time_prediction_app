
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def train_models(data):
    X = data.drop(columns=["lead_time"])
    y = data["lead_time"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsRegressor(n_neighbors=5)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    knn.fit(X_train_scaled, y_train)
    rf.fit(X_train_scaled, y_train)

    models = {"knn": knn, "rf": rf}
    X_test_scaled_dict = {"X_test": X_test_scaled, "y_test": y_test}
    return models, scaler, X_test_scaled_dict

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return mae, rmse

def predict(model, input_scaled):
    return model.predict(input_scaled)
