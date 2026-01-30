import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

st.title("🏦 Loan Approval Prediction App")
st.write("Enter applicant details to check loan approval chances")

# =====================
# Load Dataset
# =====================
df = pd.read_csv("LP_Train.csv")

# =====================
# Data Cleaning (YOUR LOGIC)
# =====================
df['Dependents'] = df['Dependents'].fillna(0)
df['Dependents'] = df['Dependents'].replace('[+]','', regex=True).astype(int)

df['Gender'] = df['Gender'].fillna('Male')
df['Married'] = df['Married'].fillna('Yes')
df['Self_Employed'] = df['Self_Employed'].fillna('Yes')

df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
df['Credit_History'] = df['Credit_History'].fillna(0)

df = df.rename(columns={'Loan_Status': 'Loan_approval'})

# =====================
# Encoding
# =====================
le = LabelEncoder()
cat_cols = ['Gender','Married','Education','Self_Employed','Property_Area','Loan_approval']

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# =====================
# Model Training
# =====================
X = df[['Gender','Married','Dependents','Education','Self_Employed',
        'ApplicantIncome','CoapplicantIncome','LoanAmount',
        'Loan_Amount_Term','Credit_History','Property_Area']]

y = df['Loan_approval']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# =====================
# USER INPUT
# =====================
st.header("📋 Applicant Details")

name = st.text_input("Applicant Name")

gender = st.selectbox("Gender", ['Male','Female'])
married = st.selectbox("Married", ['Yes','No'])
education = st.selectbox("Education", ['Graduate','Not Graduate'])
self_emp = st.selectbox("Self Employed", ['Yes','No'])
property_area = st.selectbox("Property Area", ['Urban','Semiurban','Rural'])

dependents = st.number_input("Dependents", 0, 5, 0)
app_income = st.number_input("Applicant Income", 0)
co_income = st.number_input("Co-applicant Income", 0)
loan_amt = st.number_input("Loan Amount", 0)
loan_term = st.selectbox("Loan Term", [360, 240, 180, 120])
credit_history = st.selectbox("Credit History", [1, 0])

# =====================
# Convert Input
# =====================
input_data = pd.DataFrame([[gender, married, dependents, education,
                            self_emp, app_income, co_income,
                            loan_amt, loan_term, credit_history, property_area]],
                          columns=X.columns)

for col in input_data.columns:
    if input_data[col].dtype == 'object':
        input_data[col] = le.fit_transform(input_data[col])

# =====================
# Prediction
# =====================
if st.button("🔍 Check Loan Approval"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader(f"Applicant: {name}")

    if prediction == 1:
        st.success(f"✅ Loan Approved (Chance: {probability*100:.2f}%)")
    else:
        st.error(f"❌ Loan Not Approved (Chance: {probability*100:.2f}%)")

# =====================
# EXTRA INSIGHTS
# =====================
st.header("📊 Insights")
st.bar_chart(df.groupby('Credit_History')['Loan_approval'].mean())
