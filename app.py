import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Loan Approval Predictor", layout="wide")

st.title("🏦 Loan Approval Prediction App")

# =========================
# Upload Dataset
# =========================
st.sidebar.header("Upload Dataset")
file = st.sidebar.file_uploader("Upload Loan CSV File", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # =========================
    # Data Cleaning
    # =========================
    df['Dependents'] = df['Dependents'].fillna(0)
    df['Dependents'] = df['Dependents'].replace('[+]','', regex=True).astype(int)

    df['Gender'] = df['Gender'].fillna('Male')
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean()).astype(int)
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
    df['Credit_History'] = df['Credit_History'].fillna(0).astype(int)

    df = df.rename(columns={'Loan_Status': 'Loan_approval'})

    # Encode categorical columns
    le = LabelEncoder()
    cat_cols = ['Gender','Married','Education','Self_Employed','Property_Area','Loan_approval']

    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # =========================
    # Model Training
    # =========================
    X = df.drop('Loan_approval', axis=1)
    y = df['Loan_approval']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    st.success("✅ Model trained successfully!")

    # =========================
    # User Input Section
    # =========================
    st.sidebar.header("Enter Applicant Details")

    gender = st.sidebar.selectbox("Gender", ['Male','Female'])
    married = st.sidebar.selectbox("Married", ['Yes','No'])
    education = st.sidebar.selectbox("Education", ['Graduate','Not Graduate'])
    self_employed = st.sidebar.selectbox("Self Employed", ['Yes','No'])
    property_area = st.sidebar.selectbox("Property Area", ['Urban','Semiurban','Rural'])

    applicant_income = st.sidebar.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
    loan_term = st.sidebar.number_input("Loan Amount Term", min_value=0)
    credit_history = st.sidebar.selectbox("Credit History", [0,1])
    dependents = st.sidebar.selectbox("Dependents", [0,1,2,3])

    if st.sidebar.button("Predict Loan Approval"):
        user_data = pd.DataFrame({
            'Gender': [le.fit_transform([gender])[0]],
            'Married': [le.fit_transform([married])[0]],
            'Dependents': [dependents],
            'Education': [le.fit_transform([education])[0]],
            'Self_Employed': [le.fit_transform([self_employed])[0]],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_term],
            'Credit_History': [credit_history],
            'Property_Area': [le.fit_transform([property_area])[0]]
        })

        prediction = model.predict(user_data)[0]
        probability = model.predict_proba(user_data)[0][1]

        st.subheader("🔮 Prediction Result")
        if prediction == 1:
            st.success(f"Loan Approved ✅ (Chance: {probability*100:.2f}%)")
        else:
            st.error(f"Loan Not Approved ❌ (Chance: {probability*100:.2f}%)")

    # =========================
    # Interactive Visualizations
    # =========================
    st.subheader("📊 Data Insights")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sb.boxplot(x=df['ApplicantIncome'], ax=ax)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sb.barplot(x=df['Loan_approval'], y=df['CoapplicantIncome'], ax=ax)
        st.pyplot(fig)

    fig, ax = plt.subplots()
    sb.countplot(x=df['Credit_History'], hue=df['Loan_approval'], ax=ax)
    st.pyplot(fig)

else:
    st.info("👈 Upload a CSV file to get started")
