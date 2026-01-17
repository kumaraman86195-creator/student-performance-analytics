import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model and features
model = joblib.load(r"C:\Users\kumar\student-performance-analytics\model\student_pass_model.pkl")
features = joblib.load(r"C:\Users\kumar\student-performance-analytics\model\feature_columns.pkl")

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Prediction System")
st.markdown("Predict student academic outcome using Machine Learning")

st.divider()

# ================= INPUT FORM =================
st.subheader("📌 Enter Student Details")

studytime_ui = st.selectbox(
    "Study Time (hours)",
    list(range(1, 18))
)

if studytime_ui <= 2:
    studytime = 1
elif studytime_ui <= 5:
    studytime = 2
elif studytime_ui <= 10:
    studytime = 3
else:
    studytime = 4
failures = st.selectbox("Previous Failures", [0, 1, 2, 3])
absences = st.slider("Number of Absences", 0, 100, 5)
schoolsup = st.radio("School Support", ["Yes", "No"])
famsup = st.radio("Family Support", ["Yes", "No"])
internet = st.radio("Internet Access", ["Yes", "No"])
activities = st.radio("Extra-curricular Activities", ["Yes", "No"])
health = st.slider("Health (1 = Poor, 5 = Excellent)", 1, 5, 3)

# Convert categorical inputs
input_data = {
    "studytime": studytime,
    "failures": failures,
    "absences": absences,
    "schoolsup_yes": 1 if schoolsup == "Yes" else 0,
    "famsup_yes": 1 if famsup == "Yes" else 0,
    "internet_yes": 1 if internet == "Yes" else 0,
    "activities_yes": 1 if activities == "Yes" else 0,
    "health": health
}

# Create DataFrame
input_df = pd.DataFrame([input_data])

# Align columns with training features
input_df = input_df.reindex(columns=features, fill_value=0)

# ================= PREDICTION =================
if st.button("🔮 Predict Performance"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df).max() * 100

    st.divider()
    st.subheader("📊 Prediction Result")

    if prediction == 0:
        st.error("❌ Prediction: FAIL")
        st.warning("⚠ Risk Level: HIGH")
        st.markdown("""
        **Recommendations:**
        - Increase daily study time  
        - Reduce absences  
        - Academic counseling advised  
        """)
    else:
        st.success("✅ Prediction: PASS")
        st.info("🎯 Risk Level: LOW")
        st.markdown("""
        **Recommendations:**
        - Maintain study consistency  
        - Participate in activities  
        - Continue healthy habits  
        """)

    st.caption(f"Prediction Confidence: {prob:.2f}%")

st.divider()
st.caption("Developed by Aman Kumar | ML Project")
