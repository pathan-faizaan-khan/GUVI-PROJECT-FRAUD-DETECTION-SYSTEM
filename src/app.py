
import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path
from predict import load_model, predict_single
from eda import fraud_distribution, fraud_rate_by_type, fraud_trend_by_hour, correlation_heatmap

st.set_page_config(page_title="Bank Fraud Detector", layout="wide")

st.title("💳 Bank Transaction Fraud Detection Dashboard")

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models" / "fraud_model_rf.joblib"
METRICS_PATH = BASE / "models" / "metrics.json"
DATA_PATH = BASE / "data" / "transactions.csv"

@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH)
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    df = pd.read_csv(DATA_PATH)
    return model, metrics, df

model, metrics, df = load_resources()

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 EDA Dashboard", "📈 Model Metrics"])

with tab1:
    st.sidebar.header("Input Transaction")
    amount = st.sidebar.number_input("Amount (INR)", min_value=0.0, value=1500.0, step=10.0)
    ttype = st.sidebar.selectbox("Transaction Type", df["Type"].unique().tolist())
    location = st.sidebar.selectbox("Location", df["Location"].unique().tolist())
    hour = st.sidebar.slider("Hour of Day", 0, 23, 12)

    if st.sidebar.button("Predict"):
        res = predict_single(model, amount, ttype, location, hour)
        prob = res["probability"]
        pred = res["prediction"]
        st.markdown("---")
        st.subheader("Prediction Result")
        if pred == 1:
            st.error(f"⚠️ **Fraudulent Transaction Detected!**")
        else:
            st.success(f"✅ **Genuine Transaction**")
        st.markdown(f"**Fraud Probability:** {prob*100:.2f}%")
        st.progress(prob)

        # Gauge visualization
        st.markdown("#### Probability Gauge")
        st.write(f"<div style='width:100%;height:30px;background:linear-gradient(90deg,green {100-prob*100}%,red {prob*100}%);border-radius:8px;'><span style='padding-left:10px;color:white;font-weight:bold;'>{prob*100:.2f}%</span></div>", unsafe_allow_html=True)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fraud vs Genuine")
        fig, ax = plt.subplots()
        fraud_distribution(df)
        st.pyplot(fig)
    with col2:
        st.subheader("Fraud Rate by Transaction Type")
        fig, ax = plt.subplots()
        fraud_rate_by_type(df)
        st.pyplot(fig)
    st.subheader("Fraud Trend by Hour of Day")
    fig, ax = plt.subplots()
    fraud_trend_by_hour(df)
    st.pyplot(fig)
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    correlation_heatmap(df)
    st.pyplot(fig)

with tab3:
    st.subheader("Model Performance")
    report = metrics.get("classification_report", {})
    accuracy = report.get("accuracy", None)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("ROC AUC", f"{metrics.get('roc_auc'):.3f}")
    with col2:
        if accuracy:
            st.metric("Accuracy", f"{accuracy:.3f}")
    if report:
        st.dataframe(pd.DataFrame(report).transpose().round(3))
    
    if hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importance")
        importances = pd.Series(model.feature_importances_, index=model.feature_names_in_)
        st.bar_chart(importances.sort_values(ascending=False))
