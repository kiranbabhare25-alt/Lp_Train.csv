
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Loan Approval Prediction App")
st.write("Enter applicant details to check loan approval status.")

# ---------------------------
# LOAD & PREPROCESS DATA
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("../Datasets/LP_Train.csv")

    df['Dependents'] = df['Dependents'].fillna(0)
    df['Dependents'] = df['Dependents'].replace('[+]','', regex=True).astype(int)

    df['Gender'] = df['Gender'].fillna('Male')
    df['Married'] = df['Married'].fillna('Yes')
    df['Self_Employed'] = df['Self_Employed'].fillna('Yes')

    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean()).astype(int)
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())

    df['Credit_History'] = df['Credit_History'].fillna(0).astype(int)

    df = df.rename(columns={'Loan_Status': 'Loan_approval'})
    df['Loan_approval'] = df['Loan_approval'].map({'Y': 1, 'N': 0})

    return df

df = load_data()

# ---------------------------
# ENCODING
# ---------------------------
encoder = LabelEncoder()
categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']

for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

# ---------------------------
# MODEL TRAINING
# ---------------------------
X = df.drop(['Loan_approval', 'Loan_ID'], axis=1)
y = df['Loan_approval']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------------------
# USER INPUT SECTION
# ---------------------------
st.sidebar.header("👤 Applicant Information")

name = st.sidebar.text_input("Applicant Name")

gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
married = st.sidebar.selectbox("Married", ['Yes', 'No'])
dependents = st.sidebar.selectbox("Dependents", [0, 1, 2, 3])
education = st.sidebar.selectbox("Education", ['Graduate', 'Not Graduate'])
self_employed = st.sidebar.selectbox("Self Employed", ['Yes', 'No'])

app_income = st.sidebar.number_input("Applicant Income", min_value=0)
coapp_income = st.sidebar.number_input("Coapplicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
loan_term = st.sidebar.selectbox("Loan Amount Term", [360, 180, 240, 120])
credit_history = st.sidebar.selectbox("Credit History", [1, 0])
property_area = st.sidebar.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])

# ---------------------------
# PREDICTION
# ---------------------------
if st.sidebar.button("🔍 Check Loan Approval"):

    input_data = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [app_income],
        'CoapplicantIncome': [coapp_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area]
    })

    for col in categorical_cols:
        input_data[col] = encoder.fit_transform(
            list(df[col].unique()) + list(input_data[col])
        )[-1:]

    prediction = model.predict(input_data)[0]

    st.subheader(f"📋 Result for {name if name else 'Applicant'}")

    if prediction == 1:
        st.success("🎉 Loan Approved!")
    else:
        st.error("❌ Loan Not Approved")

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Machine Learning")


