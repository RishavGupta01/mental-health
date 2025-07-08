import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === Load models and preprocessing tools ===
clf = joblib.load("models/multi_classifier.pkl")   # Only predicts BurnoutRisk
reg = joblib.load("models/multi_regressor.pkl")    # Predicts 3 regression targets
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/label_encoders.pkl")
features = joblib.load("models/feature_columns.pkl")

# === Configure Streamlit ===
st.set_page_config(page_title="Mental Health Insights", layout="centered")

st.title("🧠 Mental Health Insights")
st.markdown("Analyze burnout risk, mental health days, satisfaction, and productivity based on employee inputs.")

# === Input Form ===
with st.form("input_form"):
    st.subheader("📥 Enter Employee Data")
    input_data = {}

    field_hints = {
        "Gender": "Select gender identity",
        "Country": "Employee's current country",
        "Department": "Employee’s department",
        "JobRole": "Employee’s job role",
        "Age": "Age in years (18–65)",
        "YearsAtCompany": "Years in the current company (0–40)",
        "WorkHoursPerWeek": "Typical weekly work hours (30–80)",
        "RemoteWork": "Is employee working remotely?",
        "BurnoutLevel": "Self-rated burnout level (1–10)",
        "StressLevel": "Self-rated stress (1–10)",
        "JobSatisfaction": "Self-rated satisfaction (1–10)",
        "ProductivityScore": "Internal productivity score (1–10)",
        "SleepHours": "Average daily sleep hours (0–12)",
        "PhysicalActivityHrs": "Weekly exercise hours (0–20)",
        "CommuteTime": "One-way commute in minutes (0–180)",
        "HasMentalHealthSupport": "Is support available in company?",
        "ManagerSupportScore": "Manager support (1–10)",
        "HasTherapyAccess": "Does the employee have therapy access?",
        "SalaryRange": "Salary band (e.g., Low, Mid, High)",
        "WorkLifeBalanceScore": "Rated 1–10",
        "TeamSize": "Size of the immediate team",
        "CareerGrowthScore": "Perceived growth (1–10)",
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

    # === Burnout Classification ===
    burnout_pred = clf.predict(input_scaled)[0]

    # === Regression Predictions ===
    reg_preds = reg.predict(input_scaled)[0]
    reg_labels = ["MentalHealthDaysOff", "JobSatisfaction", "ProductivityScore"]

    # === Display Results ===
    st.subheader("📊 Prediction Results")

    result = "Yes" if burnout_pred == 1 else "No"
    st.write(f"**BurnoutRisk:** {result}")
    if burnout_pred == 1:
        st.caption("Employees with long work hours, little rest, and weak support are more likely to burn out.")

    st.markdown("### 📈 Mental Health & Productivity Estimates")
    explanations = {
        "MentalHealthDaysOff": lambda x: "Over 3 days/month suggests high risk zone." if x > 3 else "Healthy range (≤3 days/month).",
        "JobSatisfaction": lambda x: "Low satisfaction — possible disengagement." if x < 5 else "Fair to good satisfaction.",
        "ProductivityScore": lambda x: "Moderate productivity — check for blockers." if x < 6 else "Healthy productivity trend.",
    }

    for i, col in enumerate(reg_labels):
        val = reg_preds[i]
        st.write(f"**{col}:** {val:.2f}")
        st.caption(explanations[col](val))

    # === Dynamic Recommendations ===
    with st.expander("🛠 Recommendations & Next Steps"):
        if burnout_pred == 1:
            st.markdown("**1. Burnout Risk**")
            st.markdown("- Offer mental health days.\n- Reduce overwork.\n- Encourage mindfulness breaks.")

        if reg_preds[0] > 3:
            st.markdown("**2. High Mental Health Leave**")
            st.markdown("- Provide therapy access.\n- Schedule mental health check-ins.")

        if reg_preds[1] < 5:
            st.markdown("**3. Low Job Satisfaction**")
            st.markdown("- Discuss career growth.\n- Realign responsibilities.")

        if reg_preds[2] < 6:
            st.markdown("**4. Productivity Support**")
            st.markdown("- Reduce task load.\n- Offer skill-building sessions.")

# === Mobile-Friendly Styling ===
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0e1117;
    color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
    font-size: 16px;
    padding: 10px;
}
@media screen and (max-width: 768px) {
    .stSelectbox div[data-baseweb="select"], input, .stButton > button {
        width: 100% !important;
        font-size: 16px !important;
    }
    .stForm, .stMarkdown, .stSubheader, .stCaption {
        padding: 8px;
    }
}
.stButton > button {
    background-color: #262730;
    color: white;
    padding: 12px 24px;
    border-radius: 6px;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)
