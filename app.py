
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils.preprocessing import load_scaler, preprocess_input
from utils.modeling import load_model, predict

st.set_page_config(page_title="Lead Time Prediction", layout="wide")
st.title("🚚 Lead Time Prediction Dashboard")

st.markdown("""
Predict supplier lead time using historical and macroeconomic data.
Compare KNN and Random Forest models using MAE and RMSE metrics.
""")

# Show model performance metrics
st.subheader("📊 Model Evaluation Metrics (on test set)")
metrics_df = pd.read_csv("data/model_metrics.csv")
st.dataframe(metrics_df)

# Plot comparison bar chart
fig, ax = plt.subplots(figsize=(6, 4))
metrics_df.plot(x="Model", y=["MAE", "RMSE"], kind="bar", ax=ax)
plt.title("Model Error Comparison")
plt.ylabel("Error")
st.pyplot(fig)

# Model selection and prediction
st.subheader("🧠 Predict Lead Time")
model_choice = st.selectbox("Choose Model", options=["knn", "rf"], format_func=lambda x: "KNN" if x=="knn" else "Random Forest")

uploaded_file = st.file_uploader("📤 Upload Supplier Data CSV", type=["csv"])
if uploaded_file:
    input_df = pd.read_csv(uploaded_file)
    st.write("✅ Uploaded Data Preview", input_df.head())

    model = load_model(model_choice)
    scaler = load_scaler()
    X_scaled = preprocess_input(input_df, scaler)
    predictions = predict(model, X_scaled)

    input_df[f"Predicted Lead Time ({model_choice.upper()})"] = predictions
    st.write("📈 Predictions", input_df)
