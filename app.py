import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from src.explanation import ModelExplainer

# Page Config
st.set_page_config(page_title="Credit Risk Engine", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #00d4ff;
        color: black;
        font-weight: bold;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Credit Risk / Loan Default Prediction")
st.markdown("Enter applicant details to assess default risk using our production ML engine.")

# Load Model and Preprocessor
@st.cache_resource
def load_assets():
    model = joblib.load(os.path.join('models', 'xgboost.joblib'))
    preprocessor = joblib.load(os.path.join('models', 'preprocessor.joblib'))
    return model, preprocessor

try:
    model, preprocessor = load_assets()
except Exception:
    st.error("Model files not found. Please run `python main.py` first to train the models.")
    st.stop()

# Sidebar Inputs
st.sidebar.header("Applicant Information")

person_age = st.sidebar.number_input("Age", 18, 100, 30)
person_income = st.sidebar.number_input("Annual Income ($)", 0, 1000000, 50000)
person_home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
person_emp_length = st.sidebar.number_input("Employment Length (Years)", 0, 50, 5)
loan_intent = st.sidebar.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
loan_grade = st.sidebar.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
loan_amnt = st.sidebar.number_input("Loan Amount ($)", 0, 500000, 10000)
loan_int_rate = st.sidebar.number_input("Interest Rate (%)", 0.0, 30.0, 10.0)
loan_percent_income = loan_amnt / person_income if person_income > 0 else 0
cb_person_default_on_file = st.sidebar.selectbox("Historical Default?", ["N", "Y"])
cb_person_cred_hist_length = st.sidebar.number_input("Credit History Length", 0, 50, 5)

# Create input dataframe
input_data = pd.DataFrame({
    'person_age': [person_age],
    'person_income': [person_income],
    'person_home_ownership': [person_home_ownership],
    'person_emp_length': [person_emp_length],
    'loan_intent': [loan_intent],
    'loan_grade': [loan_grade],
    'loan_amnt': [loan_amnt],
    'loan_int_rate': [loan_int_rate],
    'loan_percent_income': [loan_percent_income],
    'cb_person_default_on_file': [cb_person_default_on_file],
    'cb_person_cred_hist_length': [cb_person_cred_hist_length]
})

# Prediction logic
if st.button("Analyze Risk"):
    # Preprocess
    processed_input = preprocessor.transform(input_data)
    
    # Predict
    prob = model.predict_proba(processed_input)[0][1]
    risk_score = round(prob * 100, 2)
    
    # UI Layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Risk Score")
        if risk_score < 30:
            category = "LOW"
            color = "green"
        elif risk_score < 70:
            category = "MEDIUM"
            color = "orange"
        else:
            category = "HIGH"
            color = "red"
            
        st.markdown(f"<h1 style='text-align: center; color: {color};'>{risk_score}%</h1>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>Category: {category}</h3>", unsafe_allow_html=True)
        
        st.metric("Probability of Default", f"{risk_score}%")

    with col2:
        st.subheader("Decision Summary")
        if category == "LOW":
            st.success("Applicant is likely to repay. Approval Recommended.")
        elif category == "MEDIUM":
            st.warning("Moderate risk. Manual review advised.")
        else:
            st.error("High risk of default. Rejection Recommended.")

    # SHAP Explanation
    st.divider()
    st.subheader("Model Interpretability (SHAP)")
    explainer = ModelExplainer(model, preprocessor.get_feature_names())
    
    with st.spinner("Generating SHAP explanation..."):
        fig = explainer.get_local_explanation(processed_input)
        st.pyplot(fig)
        st.info("The waterfall plot above shows how each feature contributed to the final probability. Red bars increase risk, while blue bars decrease it.")

else:
    st.info("Adjust values in the sidebar and click 'Analyze Risk' to begin.")
