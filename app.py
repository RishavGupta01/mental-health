import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === Load models and preprocessing tools ===
clf = joblib.load("models/multi_classifier.pkl")
reg = joblib.load("models/multi_regressor.pkl")
xgb_model = joblib.load("models/attrition_xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/label_encoders.pkl")
features = joblib.load("models/feature_columns.pkl")
attrition_features = joblib.load("models/attrition_features.pkl")

# === Configure Streamlit ===
st.set_page_config(page_title="Mental Health Insights", layout="wide")

st.title("Mental Health Insights Dashboard")
st.markdown("Predict burnout, stress flags, need for support, attrition risk, and productivity indicators.")

# === Input Form ===
# === Input Form with Hints ===
with st.form("input_form"):
    st.subheader("Enter Employee Data")
    input_data = {}

    field_hints = {
        "Gender": "Select gender identity",
        "Department": "Department the employee works in",
        "EducationLevel": "Highest education level attained",
        "HasMentalHealthSupport": "Does the company offer mental health support?",
        "HasTherapyAccess": "Does the employee have access to therapy or counseling?",
        "RemoteWork": "Is the employee working remotely?",
        "CompanySize": "Company size (e.g., 1–10, 11–50...)",
        "MentalHealthCoverage": "Does company cover mental health expenses?",
        "Age": "Age in years (18–65)",
        "WorkHoursPerWeek": "Typical hours worked per week (30–80)",
        "YearsInCompany": "Years spent at current company (0–40)",
        "SleepHours": "Average hours of sleep per day (0–12)",
        "PhysicalActivityHrs": "Exercise hours per week (0–20)",
        "WorkLifeBalanceScore": "Self-rated work-life balance (1–10)",
        "StressLevel": "Self-rated stress level (1 = low, 10 = high)",
        "EngagementScore": "Engagement level at work (1–10)",
        "SocialSupportScore": "Support from coworkers/supervisors (1–10)",
        "CommuteTime": "Daily commute duration in minutes (0–180)",
        "ManagerSupportScore": "Support from manager/supervisor (1 = low, 10 = high)",
        "CareerGrowthScore": "Perceived career growth opportunities (1 = poor, 10 = strong)",
        "BurnoutLevel": "Self-assessed burnout level (1 = low, 10 = high)",
    }

    for col in features:
        label = f"{col} — {field_hints.get(col, '')}"
        if col in encoders:
            options = list(encoders[col].classes_)
            selected = st.selectbox(label, options, key=col)
            input_data[col] = encoders[col].transform([selected])[0]
        else:
            value = st.number_input(label, step=1.0, key=col)
            input_data[col] = value

    submitted = st.form_submit_button("Run Predictions")



# === Make Predictions ===
if submitted:
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    # Classification
    class_preds = clf.predict(input_scaled)[0]
    class_labels = ["BurnoutRisk", "NeedsSupport", "HighStressFlag"]

    # Regression
    reg_preds = reg.predict(input_scaled)[0]
    reg_labels = ["MentalHealthDaysOff", "JobSatisfaction", "ProductivityScore", "WellBeingScore"]

    # Attrition
    attr_input = input_df[attrition_features]
    attr_input_scaled = scaler.transform(attr_input)
    attr_pred = xgb_model.predict(attr_input_scaled)[0]

    # === Display Predictions ===
    st.subheader("Classification Results")
    for i, col in enumerate(class_labels):
        st.write(f"{col}: {'Yes' if class_preds[i] == 1 else 'No'}")

    st.subheader("Regression Estimates")
    for i, col in enumerate(reg_labels):
        st.write(f"{col}: {reg_preds[i]:.2f}")

    st.subheader("Attrition Risk")
    st.write("Likely to leave" if attr_pred == 1 else "Not likely to leave")

# === Custom Styling ===
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
    .stTextInput>div>input {
        background-color: #1a1d21;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)
