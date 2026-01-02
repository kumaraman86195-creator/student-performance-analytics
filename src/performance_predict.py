import joblib
import pandas as pd

# Load model and feature columns
model = joblib.load(r"C:\Users\kumar\student-performance-analytics\model\student_pass_model.pkl")
feature_columns = joblib.load(r"C:\Users\kumar\student-performance-analytics\model\feature_columns.pkl")

print("---- STUDENT PERFORMANCE PREDICTION SYSTEM ----")

# Take inputs
sex = input("Enter Gender (M/F): ")
age = int(input("Enter Age: "))
studytime = int(input("Enter Study Time (1 = Low, 4 = High): "))
failures = int(input("Enter Number of Past Failures: "))

schoolsup = input("School Support (Yes/No): ")
famsup = input("Family Support (Yes/No): ")
paid = input("Extra Paid Classes (Yes/No): ")
activities = input("Extra Curricular Activities (Yes/No): ")
higher = input("Wants Higher Education (Yes/No): ")
internet = input("Internet Access (Yes/No): ")

absences = int(input("Number of Absences: "))

# Create input dictionary
input_data = {
    'age': age,
    'studytime': studytime,
    'failures': failures,
    'absences': absences,
    'sex_M': 1 if sex == 'M' else 0,
    'schoolsup_yes': 1 if schoolsup.lower() == 'yes' else 0,
    'famsup_yes': 1 if famsup.lower() == 'yes' else 0,
    'paid_yes': 1 if paid.lower() == 'yes' else 0,
    'activities_yes': 1 if activities.lower() == 'yes' else 0,
    'higher_yes': 1 if higher.lower() == 'yes' else 0,
    'internet_yes': 1 if internet.lower() == 'yes' else 0
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Ensure all feature columns exist
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns
input_df = input_df[feature_columns]

# Predict
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

print("\n----- RESULT -----")
print("Pass Probability:", round(probability * 100, 2), "%")

if prediction == 1:
    print("Prediction: Student is likely to PASS")
else:
    print("Prediction: Student is likely to FAIL")
