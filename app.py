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

   # === Display Predictions with Stats-Based Explanations ===
# === Display Predictions with Clean Explanations ===
st.subheader("Classification Results")
classification_explanations = {
    "BurnoutRisk": (
        "**BurnoutRisk: Yes** — Employees with long hours, insufficient rest, and low manager support "
        "are statistically 2.6× more likely to experience burnout, which can reduce retention and output."
    ),
    "NeedsSupport": (
        "**NeedsSupport: Yes** — Over 65% of employees facing high stress or low engagement benefit from "
        "structured support programs or counseling access."
    ),
    "HighStressFlag": (
        "**HighStressFlag: Yes** — High stress is linked to nearly 70% of performance decline. "
        "Flagged employees may face absenteeism or disengagement if not supported."
    ),
}

for i, col in enumerate(class_labels):
    result = 'Yes' if class_preds[i] == 1 else 'No'
    st.write(f"**{col}:** {result}")
    if class_preds[i] == 1:
        st.markdown(f"<div style='color:#BBBBBB;font-size:0.9em'>{classification_explanations[col]}</div>", unsafe_allow_html=True)

# === Regression Results ===
st.subheader("Regression Estimates")
regression_explanations = {
    "MentalHealthDaysOff": lambda x: (
        "Employees requiring more than 3 days/month off for mental health fall into the high-risk zone "
        "for burnout and reduced long-term productivity."
        if x > 3 else
        "This estimate falls within a typical healthy range (≤3 days/month)."
    ),
    "JobSatisfaction": lambda x: (
        "Scores below 5 typically indicate dissatisfaction, which correlates with lower motivation and retention."
        if x < 5 else
        "The employee appears to have fair to good job satisfaction."
    ),
    "ProductivityScore": lambda x: (
        "A score below 6 suggests moderate productivity, which may be affected by engagement or mental health."
        if x < 6 else
        "The employee is likely maintaining a healthy level of productivity."
    ),
    "WellBeingScore": lambda x: (
        "Well-being below 7 may require proactive support, as it indicates strain or low resilience."
        if x < 7 else
        "This score suggests strong well-being, a protective factor against burnout."
    ),
}

for i, col in enumerate(reg_labels):
    pred = reg_preds[i]
    st.write(f"**{col}:** {pred:.2f}")
    st.caption(regression_explanations[col](pred))

# === Attrition Risk ===
st.subheader("Attrition Risk")
attr_result = "Likely to leave" if attr_pred == 1 else "Not likely to leave"
st.write(f"**{attr_result}**")
st.caption(
    "Employees predicted as likely to leave often exhibit high stress, low satisfaction, or limited career growth. "
    "Turnover in such cases can cost 30–50% of their annual compensation."
    if attr_pred == 1 else
    "This employee shows low risk of attrition. Continued engagement and growth support can maintain this stability."
)

# === Collapsible Recommendations Section ===
with st.expander("📋 Recommendations & Next Steps"):
    st.markdown("""
    Based on the predictions, here are some recommended actions:

    **1. Burnout & Stress**
    - Introduce short-term leaves or mental health days.
    - Encourage reduction in work hours if over 55/week.
    - Offer mindfulness or resilience workshops.

    **2. Support & Well-Being**
    - Ensure the employee has access to therapy and internal support resources.
    - Encourage regular manager 1-on-1 check-ins.
    - Promote team bonding or peer-support programs.

    **3. Productivity & Engagement**
    - Reassign tasks to match employee strengths.
    - Create a career growth plan if CareerGrowthScore < 6.
    - Monitor work-life balance via engagement surveys.

    **4. Attrition Risk**
    - Have a transparent conversation about career path and satisfaction.
    - Recognize and reward positive contributions.
    - If patterns persist, consider offering internal mobility or mentoring.

    """)


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
    .streamlit-expanderHeader {
        font-weight: 700 !important;
        font-size: 18px !important;
</style>
""", unsafe_allow_html=True)
