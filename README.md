⁸# Student Performance Analytics & Prediction
 #  Project Overview

This project analyzes student academic data to identify key factors influencing student performance and predicts whether a student is likely to PASS or FAIL using machine learning.

The project combines data analytics, exploratory data analysis (EDA), and a Logistic Regression model, along with a Command-Line Interface (CLI) for real-time prediction.

# Objectives

->Analyze factors affecting student academic performance

->Identify patterns related to pass/fail outcomes

->Build a predictive model to estimate pass probability

->Provide a CLI-based prediction system

# Machine Learning Approach

Algorithm: Logistic Regression

Problem Type: Binary Classification (Pass / Fail)

Target Variable:

Pass = 1 → Final grade (G3) ≥ 10

Pass = 0 → Final grade (G3) < 10

Logistic Regression was chosen for its simplicity, interpretability, and suitability for educational decision-making.

# Project Structure
student-performance-analytics/
│
├── data/
│   └── raw/
│       └── student-mat.csv
│
├── notebooks/
│   └── student_analysis.ipynb
│
├── src/
│   └── performance_predict.py
│
├── model/
│   ├── student_pass_model.pkl
│   └── feature_columns.pkl
│
├── requirements.txt
└── README.md

#  Dataset

Name: Student Performance Dataset

Source: Kaggle

Records: ~395 students

Features Include:

Demographics (age, gender)

Study habits (study time, failures)

Support systems (family, school)

Attendance (absences)

# Exploratory Data Analysis (EDA)

Key insights from the analysis:

Students with higher study time perform better

Previous academic failures strongly increase failure risk

High absenteeism negatively impacts performance

Family and school support improve pass rates

#  Installation & Setup
1️⃣ Clone the Repository
git clone <repository-url>
cd student-performance-analytics

2️⃣ Install Dependencies
pip install -r requirements.txt

#  Model Training

Model training and analysis are performed in the Jupyter notebook:

notebooks/student_analysis.ipynb


This includes:

Data cleaning

Feature selection & encoding

Model training & evaluation

# Prediction Using CLI

Run the CLI prediction script:

cd src
python performance_predict.py

Sample Input
Enter Gender (M/F): F
Enter Age: 17
Enter Study Time (1 = Low, 4 = High): 3
Enter Number of Past Failures: 0
School Support (Yes/No): Yes
Family Support (Yes/No): Yes
Extra Paid Classes (Yes/No): No
Extra Curricular Activities (Yes/No): Yes
Wants Higher Education (Yes/No): Yes
Internet Access (Yes/No): Yes
Number of Absences: 2

Sample Output
Pass Probability: 78.45 %
Prediction: Student is likely to PASS

# Evaluation Metrics

Accuracy

Precision

Recall

Confusion Matrix

The model achieves reasonable accuracy (~70–80%), which is realistic for educational datasets.

# Tools & Technologies

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Joblib

# Future Enhancements

Add more advanced models (Random Forest, SVM)

Include feature importance visualization

Build a web app using Streamlit or Flask

Extend to multi-class grade prediction

#  Author

Aman Kumar
Data Analytics & Machine Learning Project
