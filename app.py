import streamlit as st
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="Loan Approval System", layout="wide")
st.title("🏦 Loan Approval Prediction System")

# -------------------------------------------------
# Load Dataset (NO UPLOAD)
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("LP_Train.csv")   # keep CSV in same folder

df = load_data()

# -------------------------------------------------
# Data Cleaning (your original logic)
# -------------------------------------------------
if 'Loan_ID' in df.columns:
    df.drop(columns=['Loan_ID'], inplace=True)

df['Dependents'] = df['Dependents'].fillna(0)
df['Dependents'] = df['Dependents'].replace('[+]', '', regex=True).astype(int)

df['Gender'] = df['Gender'].fillna('Male')
df['LoanAmount'] = df['LoanAmount'].fillna(146).astype(int)
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(342.0)
df['Credit_History'] = df['Credit_History'].fillna(0).astype(int)

df = df.rename(columns={'Loan_Status': 'Loan_approval'})

# -------------------------------------------------
# Encode categorical columns
# -------------------------------------------------
encoders = {}
cat_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_approval']

for col in cat_cols:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])

# -------------------------------------------------
# Train ML model
# -------------------------------------------------
X = df.drop('Loan_approval', axis=1)
y = df['Loan_approval']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------------------------------
# Sidebar - User Input
# -------------------------------------------------
st.sidebar.header("Applicant Details")

name = st.sidebar.text_input("Applicant Name")
income = st.sidebar.number_input("Applicant Income", min_value=0)
credit_history = st.sidebar.selectbox("Credit History", [0, 1])

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.sidebar.button("Check Loan Approval"):

    user_data = pd.DataFrame({
        'Gender': [encoders['Gender'].transform(['Male'])[0]],
        'Married': [encoders['Married'].transform(['Yes'])[0]],
        'Dependents': [0],
        'Education': [encoders['Education'].transform(['Graduate'])[0]],
        'Self_Employed': [encoders['Self_Employed'].transform(['No'])[0]],
        'ApplicantIncome': [income],
        'CoapplicantIncome': [0],
        'LoanAmount': [150],
        'Loan_Amount_Term': [360],
        'Credit_History': [credit_history],
        'Property_Area': [encoders['Property_Area'].transform(['Urban'])[0]]
    })

    prediction = model.predict(user_data)[0]
    probability = model.predict_proba(user_data)[0][1] * 100

    st.subheader(f"Result for {name if name else 'Applicant'}")

    if prediction == 1:
        st.success(f"✅ Loan Approved\n\nApproval Chance: **{probability:.2f}%**")
    else:
        st.error(f"❌ Loan Not Approved\n\nApproval Chance: **{probability:.2f}%**")

# -------------------------------------------------
# Visualizations
# -------------------------------------------------
st.subheader("📊 Loan Data Analysis")

# Row 1
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sb.boxplot(x=df['ApplicantIncome'], ax=ax)
    ax.set_title("Applicant Income Distribution")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sb.boxplot(x=df['LoanAmount'], ax=ax)
    ax.set_title("Loan Amount Distribution")
    st.pyplot(fig)

# Row 2
col3, col4 = st.columns(2)

with col3:
    fig, ax = plt.subplots()
    sb.countplot(x=df['Loan_approval'], ax=ax)
    ax.set_title("Loan Approval Count")
    st.pyplot(fig)

with col4:
    fig, ax = plt.subplots()
    sb.countplot(x=df['Credit_History'], hue=df['Loan_approval'], ax=ax)
    ax.set_title("Credit History vs Loan Approval")
    st.pyplot(fig)

# Row 3
col5, col6 = st.columns(2)

with col5:
    fig, ax = plt.subplots()
    sb.barplot(x=df['Property_Area'], y=df['LoanAmount'], ax=ax)
    ax.set_title("Property Area vs Loan Amount")
    st.pyplot(fig)

with col6:
    fig, ax = plt.subplots()
    sb.barplot(x=df['Loan_approval'], y=df['Loan_Amount_Term'], ax=ax)
    ax.set_title("Loan Term vs Loan Approval")
    st.pyplot(fig)

# Heatmap
fig, ax = plt.subplots()
sb.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
ax.set_title("Correlation Heatmap")
st.pyplot(fig)
