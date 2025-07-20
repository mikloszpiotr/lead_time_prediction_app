
import joblib

def load_model(model_name='knn'):
    model_path = f'models/{model_name}_model.pkl'
    return joblib.load(model_path)

def predict(model, X_scaled):
    return model.predict(X_scaled)
