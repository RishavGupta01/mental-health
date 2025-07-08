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

    # === Display Classification Results ===
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
    # === Dynamic Recommendations Based on Predictions ===
with st.expander("Recommendations & Next Steps"):
    st.markdown("Based on the predictions, here are some recommended actions:")

    # 1. Burnout Risk
    if class_preds[class_labels.index("BurnoutRisk")] == 1:
        st.markdown("**1. Burnout & Stress**")
        st.markdown("""
        - Introduce short-term leaves or mental health days.
        - Encourage reduction in work hours if over 55/week.
        - Offer mindfulness or resilience workshops.
        """)

    # 2. Needs Support
    if class_preds[class_labels.index("NeedsSupport")] == 1:
        st.markdown("**2. Support & Well-Being**")
        st.markdown("""
        - Ensure the employee has access to therapy and internal support resources.
        - Encourage regular manager 1-on-1 check-ins.
        - Promote team bonding or peer-support programs.
        """)

    # 3. High Stress
    if class_preds[class_labels.index("HighStressFlag")] == 1:
        st.markdown("**3. High Stress Management**")
        st.markdown("""
        - Evaluate workload and redistribute tasks if necessary.
        - Consider weekly check-ins to monitor mental state.
        - Provide access to self-paced de-stress activities.
        """)

    # 4. Regression-Based Actions
    if reg_preds[reg_labels.index("JobSatisfaction")] < 5:
        st.markdown("**4. Job Satisfaction & Growth**")
        st.markdown("""
        - Initiate transparent career development discussions.
        - Align work assignments with employee strengths.
        - Set short-term goals and recognize achievements.
        """)

    if reg_preds[reg_labels.index("ProductivityScore")] < 6:
        st.markdown("**5. Productivity Support**")
        st.markdown("""
        - Review task complexity and cognitive load.
        - Adjust deadlines if necessary to reduce pressure.
        - Explore if disengagement is due to work culture or tools.
        """)

    if reg_preds[reg_labels.index("WellBeingScore")] < 7:
        st.markdown("**6. Well-Being Support**")
        st.markdown("""
        - Recommend wellness check-ins or survey feedback.
        - Provide optional health sessions or gym vouchers.
        - Promote good sleep and exercise habits.
        """)

    if attr_pred == 1:
        st.markdown("**7. Attrition Risk**")
        st.markdown("""
        - Have a 1-on-1 conversation about job satisfaction and goals.
        - Highlight future opportunities and learning paths.
        - Consider mentoring or lateral moves before churn.
        """)


# === Custom Styling ===
st.markdown("""
<style>
    html, body, [class*="css"]  {
        background-color: #0e1117;
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
        line-height: 1.6;
    }

    .stMarkdown {
        margin-bottom: 1.5rem !important;
    }

    .stCaption {
        font-size: 0.9rem;
        color: #BBBBBB;
        margin-bottom: 20px;
    }

    .stSubheader {
        margin-top: 30px !important;
        margin-bottom: 10px !important;
    }

    .stExpander {
        margin-top: 30px;
    }

    .stButton>button {
        background-color: #262730;
        color: white;
        border-radius: 6px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

