import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models and tools
clf = joblib.load("models/attrition_features.pkl")
reg = joblib.load("models/multi_regressor.pkl")
xgb_model = joblib.load("models/attrition_xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/label_encoders.pkl")
features = joblib.load("models/feature_columns.pkl")
attrition_features = joblib.load("models/attrition_features.pkl")

st.set_page_config(page_title="Mental Health Insights", layout="wide")

st.title("Mental Health Insights Dashboard")
st.subheader("Predict burnout, stress, support needs, attrition risk, and more")

# Input form
with st.form("input_form"):
    st.write("### Employee Information")
    input_data = {}

    for col in features:
        if col in encoders:
            options = list(encoders[col].classes_)
            choice = st.selectbox(col, options)
            input_data[col] = encoders[col].transform([choice])[0]
        else:
            value = st.number_input(col, step=1.0)
            input_data[col] = value

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    # Predictions
    class_preds = clf.predict(input_scaled)[0]
    reg_preds = reg.predict(input_scaled)[0]

    attr_input = input_df[attrition_features]
    attr_input_scaled = scaler.transform(attr_input)
    attr_pred = xgb_model.predict(attr_input_scaled)[0]

    st.write("### Classification Predictions")
    st.json({label: int(pred) for label, pred in zip(clf.estimators_[0].classes_, class_preds)})

    st.write("### Regression Predictions")
    for col, pred in zip(reg.feature_names_in_, reg_preds):
        st.write(f"{col}: {pred:.2f}")

    st.write("### Attrition Risk Prediction")
    st.write("Likely to leave" if attr_pred == 1 else "Not likely to leave")

# Custom dark mode styling
st.markdown("""
<style>
    html, body, [class*="css"]  {
        background-color: #0e1117;
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #262730;
        color: white;
        border-radius: 6px;
    }
    .stSelectbox>div>div>div {
        background-color: #262730 !important;
    }
</style>
""", unsafe_allow_html=True)
