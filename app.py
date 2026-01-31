import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(
    page_title="🏦 Loan Approval Predictor",
    page_icon="💰",
    layout="centered"
)

st.markdown("""
<style>
h1 {
    color: #4B0082;
}
.stButton>button {
    background-color: #4B0082;
    color: white;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

st.title("💰 Loan Approval Prediction System")
st.markdown("### Predict your loan approval chances instantly!")

# ----------------------------------
# Load dataset (internal)
# ----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("LP_Train.csv")

df = load_data()

# ----------------------------------
# Data cleaning
# ----------------------------------
if 'Loan_ID' in df.columns:
    df.drop(columns=['Loan_ID'], inplace=True)

df['Dependents'] = df['Dependents'].fillna(0)
df['Dependents'] = df['Dependents'].replace('[+]', '', regex=True).astype(int)
df['Gender'] = df['Gender'].fillna('Male')
df['LoanAmount'] = df['LoanAmount'].fillna(146).astype(int)
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(342.0)
df['Credit_History'] = df['Credit_History'].fillna(0).astype(int)
df = df.rename(columns={'Loan_Status': 'Loan_approval'})

# ----------------------------------
# Encode categorical columns
# ----------------------------------
encoders = {}
cat_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_approval']

for col in cat_cols:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])

# ----------------------------------
# Train model
# ----------------------------------
X = df.drop('Loan_approval', axis=1)
y = df['Loan_approval']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ----------------------------------
# User Input
# ----------------------------------
st.markdown("### 📝 Enter Applicant Details")
with st.form(key="loan_form"):
    name = st.text_input("👤 Applicant Name")
    income = st.number_input("💵 Applicant Income", min_value=0)
    credit_history = st.selectbox("📜 Credit History", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    
    submitted = st.form_submit_button("✅ Check Loan Status")

# ----------------------------------
# Prediction
# ----------------------------------
if submitted:
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

    st.markdown("---")
    st.subheader(f"Result for **{name if name else 'Applicant'}**")

    if prediction == 1:
        st.success(f"🎉 Loan Approved!")
        st.progress(int(probability))
        st.metric("Approval Chance", f"{probability:.2f}%")
    else:
        st.error(f"❌ Loan Not Approved")
        st.progress(int(probability))
        st.metric("Approval Chance", f"{probability:.2f}%")

# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.markdown("💡 **Tip:** Provide accurate income and credit history to get better predictions.")
st.caption("Loan Approval Prediction System | Machine Learning Project")
