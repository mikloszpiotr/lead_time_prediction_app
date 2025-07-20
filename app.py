
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils.modeling import train_models, evaluate_model, predict
from utils.preprocessing import preprocess_input

st.set_page_config(page_title="Lead Time Prediction", layout="wide")
st.title("🚚 Lead Time Prediction Dashboard")

st.markdown("""
Predict supplier lead time using historical and macroeconomic data.
Compare KNN and Random Forest models trained on-the-fly using uploaded CSV data.
""")

uploaded_file = st.file_uploader("📤 Upload Historical Supplier Lead Time CSV", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Check required columns
    expected_cols = {"supplier_id", "product_category", "region", "historical_mean_lead_time", "macro_indicator", "lead_time"}
    if not expected_cols.issubset(set(data.columns)):
        st.error(f"❌ Uploaded CSV must contain columns: {expected_cols}")
    else:
        st.write("✅ Uploaded Data Preview", data.head())

        # Train models and evaluate
        models, scaler, test_set = train_models(data)
        X_test, y_test = test_set["X_test"], test_set["y_test"]

        metrics = {"Model": [], "MAE": [], "RMSE": []}
        for name, model in models.items():
            mae, rmse = evaluate_model(model, X_test, y_test)
            metrics["Model"].append("KNN" if name == "knn" else "Random Forest")
            metrics["MAE"].append(mae)
            metrics["RMSE"].append(rmse)

        metrics_df = pd.DataFrame(metrics)
        st.subheader("📊 Model Evaluation Metrics (on uploaded dataset)")
        st.dataframe(metrics_df)

        fig, ax = plt.subplots()
        metrics_df.plot(x="Model", y=["MAE", "RMSE"], kind="bar", ax=ax)
        st.pyplot(fig)

        # Choose model for prediction
        st.subheader("🧠 Predict New Lead Times")
        model_choice = st.selectbox("Choose model for prediction", options=["knn", "rf"], format_func=lambda x: "KNN" if x == "knn" else "Random Forest")
        new_file = st.file_uploader("📤 Upload New Supplier Data (Without Lead Time)", type=["csv"], key="new_data")

        if new_file:
            new_data = pd.read_csv(new_file)
            st.write("📄 New Data Preview", new_data.head())

            input_scaled = preprocess_input(new_data, scaler)
            predictions = predict(models[model_choice], input_scaled)

            new_data["Predicted Lead Time"] = predictions
            st.write("📈 Predictions", new_data)
