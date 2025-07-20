
# 🚚 Lead Time Prediction App

A Streamlit ML app that predicts supplier lead time variability using historical data and macroeconomic indicators. It helps supply planners reduce safety stock and improve order fulfillment accuracy.

---

## 🧠 Business Context

In supply planning, **lead time variability** is a major risk factor. Unreliable supplier delivery windows lead to:

- Excess safety stock
- Missed order fulfillment targets
- Increased working capital

**Goal**: Predict lead time variability using supplier behavior and macro factors to enable data-driven inventory planning.

---

## 🔍 Business Questions Answered

- What is the expected lead time for each supplier and product group?
- How does prediction accuracy vary between ML models?
- Which model (KNN or Random Forest) performs better on historical data?
- How much could safety stock be optimized based on more accurate lead time forecasts?

---

## 🧪 ML Solution

We use supervised regression to predict lead times:

### ✨ Features:
- `supplier_id`
- `product_category`
- `region`
- `historical_mean_lead_time`
- `macro_indicator`

### 🎯 Target:
- `lead_time` (in days)

### 🤖 Models Used:
- **K-Nearest Neighbors (KNN)** Regressor
- **Random Forest Regressor**

Both models are trained **dynamically at runtime** (no `.pkl` files used) for full compatibility with Streamlit Cloud.

---

## 📊 Streamlit Dashboard Features

- Upload historical data to train models
- View MAE and RMSE error metrics
- Compare performance of KNN vs Random Forest
- Upload new supplier data to generate real-time lead time predictions

---

## 🛠 Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas
- Matplotlib
- Joblib

---

## 🚀 How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/mikloszpiotr/lead_time_prediction_app.git
   cd lead_time_prediction_app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## 📁 Data Requirements

### For training:
Your uploaded CSV should include:
```
supplier_id, product_category, region, historical_mean_lead_time, macro_indicator, lead_time
```

### For prediction:
Same features, but **without** `lead_time`.

---

## 📈 Example Use Case

> A supply chain analyst uploads supplier delivery history to train the model. They compare model metrics and select the best-performing algorithm. Then, they upload upcoming supplier orders and receive lead time forecasts for safety stock planning.
