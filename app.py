import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Gold Price Predictor", layout="wide")

# Load models
@st.cache_resource
def load_models():
    model = joblib.load('best_gold_predictor.pkl')
    scaler = joblib.load('gold_scaler.pkl')
    return model, scaler

model, scaler = load_models()

st.title("=>>>> Gold Price Predictor")
st.markdown("**R²=0.989 | MAE=$3.37 | Linear Regression**")

col1, col2 = st.columns(2)

with col1:
    st.header("=>>>> Input Latest Data")
    gprd = st.slider("GPRD Index", 50, 300, 157)
    gprd_act = st.slider("GPRD_ACT", 50, 300, 157)
    gprd_threat = st.slider("GPRD_THREAT", 50, 300, 157)

with col2:
    silver_yest = st.number_input("Yesterday Silver ($)", 5.0, 50.0, 5.8)
    gold_yest = st.number_input("Yesterday Gold (Dataset $)", 200, 400, 303)

if st.button("=>>>> PREDICT TOMORROW'S GOLD PRICE", type="primary"):
    data = pd.DataFrame({
        'GPRD': [gprd], 'GPRD_ACT': [gprd_act], 
        'GPRD_THREAT': [gprd_threat],
        'SILVER_PRICE_lag1': [silver_yest],
        'GOLD_PRICE_lag1': [gold_yest]
    })
    
    pred = model.predict(scaler.transform(data))[0]
    real_pred = pred * 4.06  # Your scaling factor
    
    st.success(f"**Tomorrow's Gold: ${real_pred:.2f}**")
    st.balloons()
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Dataset Prediction", f"${pred:.2f}")
    with col_b:
        st.metric("Real-World Scale", f"${real_pred:.2f}")
