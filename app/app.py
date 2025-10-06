import streamlit as st
import pandas as pd
import joblib
import os

# Load model
MODEL_PATH = "../models/credit_risk_model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.error("Model file not found! Please train the model first.")
    st.stop()

st.title("Credit Risk Prediction App💵💰")
st.markdown("Provide borrower details to predict likelihood of default.")

# --- User Inputs ---
age = st.slider("Age", 18, 75, 30)
sex = st.radio("Sex", ["Male", "Female"])
limit_bal = st.number_input("Credit Limit (LIMIT_BAL)", min_value=1000, max_value=1000000, value=50000, step=1000)
pay_0 = st.selectbox("Last Month Payment Status (PAY_0)", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
avg_utilization = st.slider("Average Utilization", 0.0, 5.0, 1.0, step=0.1)
avg_payment_ratio = st.slider("Average Payment Ratio", 0.0, 5.0, 1.0, step=0.1)

education = st.selectbox("Education Level", ["Graduate", "University", "HighSchool", "Others"])
marriage = st.selectbox("Marital Status", ["Married", "Single", "Others"])

# --- Convert to encoded format ---
input_data = {
    "AGE": age,
    "SEX": 1 if sex == "Male" else 0,
    "LIMIT_BAL": limit_bal,
    "PAY_0": pay_0,
    "avg_utilization": avg_utilization,
    "avg_payment_ratio": avg_payment_ratio,
    # Education one-hot
    "EDUCATION_Graduate": 1 if education == "Graduate" else 0,
    "EDUCATION_University": 1 if education == "University" else 0,
    "EDUCATION_HighSchool": 1 if education == "HighSchool" else 0,
    "EDUCATION_Others": 1 if education == "Others" else 0,
    # Marriage one-hot
    "MARRIAGE_Married": 1 if marriage == "Married" else 0,
    "MARRIAGE_Single": 1 if marriage == "Single" else 0,
    "MARRIAGE_Others": 1 if marriage == "Others" else 0,
}

input_df = pd.DataFrame([input_data])

# --- Prediction ---
if st.button("Predict Default Risk"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Default. Probability: {prob:.2f}")
    else:
        st.success(f"✅ Low Risk of Default. Probability: {prob:.2f}")
