
# Lead Time Prediction App

This Streamlit app uses a K-Nearest Neighbors (KNN) regression model to predict supplier lead time variability.
It helps reduce safety stock and improve order fulfillment.

## 🔍 Business Problem
Supply planners often face lead time variability across suppliers and regions. Estimating future lead times helps in safety stock calculation and proactive planning.

## 🧠 ML Solution
- **Model**: K-Nearest Neighbors (KNN) Regressor
- **Features**: Supplier ID, Product Category, Region, Historical Lead Time, Macroeconomic Indicator
- **Target**: Lead Time in Days

## 📊 Dashboard Features
- Upload new supplier data
- Predict lead time
- Visualize outputs directly in browser

## 📁 Files
- `data/supplier_lead_times.csv` – Example dataset
- `models/knn_model.pkl`, `scaler.pkl` – Trained model and scaler
- `utils/` – Scripts for preprocessing and prediction
- `app.py` – Streamlit app

## 🚀 Run the App
```bash
pip install -r requirements.txt
streamlit run app.py
```
