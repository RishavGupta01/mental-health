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
st.set_page_config(page_title="Mental Health Insights", layout="wide")
st.title("Mental Health Insights Dashboard")
st.markdown("Analyze burnout risk, mental health estimates, and productivity insights based on employee inputs.")

# === Input Form ===
with st.form("input_form"):
    st.subheader("Enter Employee Data")
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

    # === Classification ===
    burnout_pred = clf.predict(input_scaled)[0]

    # === Regression ===
    reg_preds = reg.predict(input_scaled)[0]
    reg_labels = ["MentalHealthDaysOff", "JobSatisfaction", "ProductivityScore"]

    # === Output Section ===
    st.subheader("Burnout Prediction")
    result = "Yes" if burnout_pred == 1 else "No"
    st.write(f"**BurnoutRisk:** {result}")
    if burnout_pred == 1:
        st.caption("Employees with longer hours, low rest, and weak support are 2.6× more prone to burnout.")

    st.subheader("Mental Health & Productivity Estimates")
    regression_explanations = {
        "MentalHealthDaysOff": lambda x: (
            "More than 3 days/month indicates high emotional fatigue." if x > 3
            else "Within the normal range of ≤3 mental health days/month."
        ),
        "JobSatisfaction": lambda x: (
            "Scores under 5 suggest dissatisfaction or disengagement." if x < 5
            else "Satisfaction level is fair or high."
        ),
        "ProductivityScore": lambda x: (
            "Moderate productivity. May need support or motivation." if x < 6
            else "Healthy productivity trend."
        ),
    }

    for i, col in enumerate(reg_labels):
        value = reg_preds[i]
        st.write(f"**{col}:** {value:.2f}")
        st.caption(regression_explanations[col](value))

    # === Recommendations Based on Model Output ===
    with st.expander("Recommendations & Next Steps"):
        st.markdown("These insights are based on the predictions:")

        if burnout_pred == 1:
            st.markdown("**1. Burnout Risk**")
            st.markdown("""
            - Reduce work hours or redistribute workload.
            - Offer short mental health leaves.
            - Promote mindfulness or resilience sessions.
            """)

        if reg_preds[reg_labels.index("MentalHealthDaysOff")] > 3:
            st.markdown("**2. High Mental Health Leave**")
            st.markdown("""
            - Conduct check-ins to understand mental health needs.
            - Provide optional therapy or digital wellness programs.
            """)

        if reg_preds[reg_labels.index("JobSatisfaction")] < 5:
            st.markdown("**3. Low Satisfaction**")
            st.markdown("""
            - Initiate a conversation around career growth and motivation.
            - Offer new project opportunities aligned with interests.
            """)

        if reg_preds[reg_labels.index("ProductivityScore")] < 6:
            st.markdown("**4. Productivity Concerns**")
            st.markdown("""
            - Identify blockers in task flow or team dynamics.
            - Explore flexible deadlines or environment adjustments.
            """)

# === Custom Styling (Dark UI) ===
st.markdown("""
<style>
    html, body, [class*="css"]  {
        background-color: #0e1117;
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
        line-height: 1.6;
    }
    .stMarkdown { margin-bottom: 1.5rem !important; }
    .stCaption { font-size: 0.9rem; color: #BBBBBB; margin-bottom: 20px; }
    .stSubheader { margin-top: 30px !important; margin-bottom: 10px !important; }
    .stExpander { margin-top: 30px; }
    .stButton>button {
        background-color: #262730;
        color: white;
        border-radius: 6px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)
