
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ---------------- PAGE ----------------
st.set_page_config("Loan Approval App", "🏦", layout="centered")

st.markdown("""
<style>
.title {font-size:32px; font-weight:bold;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">🏦 Loan Approval Prediction System</p>', unsafe_allow_html=True)

# ---------------- DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("LP_Train.csv")
    df['Dependents'] = df['Dependents'].fillna(0).replace('[+]','',regex=True).astype(int)
    df['Gender'] = df['Gender'].fillna('Male')
    df['Married'] = df['Married'].fillna('Yes')
    df['Self_Employed'] = df['Self_Employed'].fillna('Yes')
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
    df['Credit_History'] = df['Credit_History'].fillna(0).astype(int)
    df = df.rename(columns={'Loan_Status':'Loan_approval'})
    return df

df = load_data()

# ---------------- ENCODE ----------------
encoder = LabelEncoder()
for col in ['Gender','Married','Education','Self_Employed','Property_Area','Loan_approval']:
    df[col] = encoder.fit_transform(df[col])

X = df.drop('Loan_approval', axis=1)
y = df['Loan_approval']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- MODEL ----------------
model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
model.fit(X_train, y_train)

# ---------------- SIDEBAR ----------------
st.sidebar.header("📝 Applicant Details")

name = st.sidebar.text_input("Applicant Name")
Gender = st.sidebar.selectbox("Gender", ["Male","Female"])
Married = st.sidebar.selectbox("Married", ["Yes","No"])
Dependents = st.sidebar.selectbox("Dependents", [0,1,2,3])
Education = st.sidebar.selectbox("Education", ["Graduate","Not Graduate"])
Self_Employed = st.sidebar.selectbox("Self Employed", ["Yes","No"])
ApplicantIncome = st.sidebar.number_input("Applicant Income", 0)
CoapplicantIncome = st.sidebar.number_input("Coapplicant Income", 0)
LoanAmount = st.sidebar.number_input("Loan Amount", 0)
Loan_Amount_Term = st.sidebar.number_input("Loan Term (Months)", 360)
Credit_History = st.sidebar.selectbox("Credit History", [1,0])
Property_Area = st.sidebar.selectbox("Property Area", ["Urban","Semiurban","Rural"])

# ---------------- PREDICT ----------------
if st.sidebar.button("🔍 Predict Loan Status"):
    input_df = pd.DataFrame({
        'Gender':[Gender],'Married':[Married],'Dependents':[Dependents],
        'Education':[Education],'Self_Employed':[Self_Employed],
        'ApplicantIncome':[ApplicantIncome],'CoapplicantIncome':[CoapplicantIncome],
        'LoanAmount':[LoanAmount],'Loan_Amount_Term':[Loan_Amount_Term],
        'Credit_History':[Credit_History],'Property_Area':[Property_Area]
    })

    for col in ['Gender','Married','Education','Self_Employed','Property_Area']:
        input_df[col] = encoder.fit_transform(input_df[col])

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.metric("Approval Probability", f"{proba*100:.2f}%")

    if prediction == 1:
        st.success(f"🎉 Loan Approved for {name}")
    else:
        st.error(f"❌ Loan Not Approved for {name}")

# ---------------- INSIGHTS ----------------
with st.expander("📊 Model Insights"):
    importance = pd.DataFrame({
        'Feature':X.columns,
        'Importance':model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    st.bar_chart(importance.set_index('Feature'))
