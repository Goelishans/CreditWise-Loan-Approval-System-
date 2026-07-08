import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="CreditWise Loan Approval System",
    page_icon="🏦",
    layout="wide"
)

# -----------------------------
# Load Saved Files
# -----------------------------
try:
    model = joblib.load("loan_model.pkl")
    encoder = joblib.load("encoder.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.main{
    background-color:#f8f9fa;
}

.title{
    text-align:center;
    color:#1f4e79;
    font-size:40px;
    font-weight:bold;
}

.sub{
    text-align:center;
    color:gray;
    font-size:18px;
    margin-bottom:30px;
}

div.stButton > button{
    width:100%;
    background:#1f77b4;
    color:white;
    border-radius:12px;
    height:50px;
    font-size:18px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🏦 CreditWise Loan Approval System</div>", unsafe_allow_html=True)

st.markdown(
"<div class='sub'>Predict Loan Approval using Machine Learning</div>",
unsafe_allow_html=True
)

st.divider()

st.subheader("📝 Applicant Information")

left, right = st.columns(2)

with left:

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0,
        value=50000
    )

    coapplicant_income = st.number_input(
        "Co-applicant Income",
        min_value=0,
        value=10000
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=80,
        value=30
    )

    dependents = st.selectbox(
        "Dependents",
        [0,1,2,3]
    )

    existing_loans = st.selectbox(
        "Existing Loans",
        [0,1,2,3]
    )

    savings = st.number_input(
        "Savings",
        min_value=0,
        value=50000
    )

    collateral_value = st.number_input(
        "Collateral Value",
        min_value=0,
        value=100000
    )

with right:

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=1000,
        value=200000
    )

    loan_term = st.selectbox(
        "Loan Term (Months)",
        [12,24,36,48,60,72,84]
    )

    education = st.selectbox(
        "Education Level",
        ["Graduate","Not Graduate"]
    )

    employment = st.selectbox(
        "Employment Status",
        ["Salaried","Self-employed","Unemployed"]
    )

    marital = st.selectbox(
        "Marital Status",
        ["Married","Single"]
    )

    purpose = st.selectbox(
        "Loan Purpose",
        ["Car","Education","Home","Personal"]
    )

    property_area = st.selectbox(
        "Property Area",
        ["Rural","Semiurban","Urban"]
    )

    gender = st.selectbox(
        "Gender",
        ["Female","Male"]
    )

    employer_category = st.selectbox(
        "Employer Category",
        ["Government","MNC","Private","Unemployed"]
    )

credit_score = st.slider(
    "Credit Score",
    550,
    800,
    700
)

predict_btn = st.button("Predict Loan Approval")

# ============================================
# Prediction
# ============================================

if predict_btn:

    try:

        # Education Encoding
        education_level = 1 if education == "Graduate" else 0

        # Feature Engineering
        dti_ratio = loan_amount / (applicant_income + coapplicant_income + 1)

        dti_ratio_sq = dti_ratio ** 2
        credit_score_sq = credit_score ** 2

        # Numerical Data
        numerical = pd.DataFrame({

            "Applicant_Income":[applicant_income],
            "Coapplicant_Income":[coapplicant_income],
            "Age":[age],
            "Dependents":[dependents],
            "Existing_Loans":[existing_loans],
            "Savings":[savings],
            "Collateral_Value":[collateral_value],
            "Loan_Amount":[loan_amount],
            "Loan_Term":[loan_term],
            "Education_Level":[education_level]

        })

        # Categorical Data
        categorical = pd.DataFrame({

            "Employment_Status":[employment],
            "Marital_Status":[marital],
            "Loan_Purpose":[purpose],
            "Property_Area":[property_area],
            "Gender":[gender],
            "Employer_Category":[employer_category]

        })

        # One Hot Encoding
        encoded = encoder.transform(categorical)

        encoded = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out()
        )

        # Merge
        final_df = pd.concat(
            [numerical.reset_index(drop=True),
             encoded.reset_index(drop=True)],
            axis=1
        )

        # Feature Engineering Columns
        final_df["DTI_Ratio_sq"] = dti_ratio_sq
        final_df["Credit_Score_sq"] = credit_score_sq

        # Correct Column Order
        final_df = final_df[[
            'Applicant_Income',
            'Coapplicant_Income',
            'Age',
            'Dependents',
            'Existing_Loans',
            'Savings',
            'Collateral_Value',
            'Loan_Amount',
            'Loan_Term',
            'Education_Level',
            'Employment_Status_Salaried',
            'Employment_Status_Self-employed',
            'Employment_Status_Unemployed',
            'Marital_Status_Single',
            'Loan_Purpose_Car',
            'Loan_Purpose_Education',
            'Loan_Purpose_Home',
            'Loan_Purpose_Personal',
            'Property_Area_Semiurban',
            'Property_Area_Urban',
            'Gender_Male',
            'Employer_Category_Government',
            'Employer_Category_MNC',
            'Employer_Category_Private',
            'Employer_Category_Unemployed',
            'DTI_Ratio_sq',
            'Credit_Score_sq'
        ]]

        # Scaling
        final_scaled = scaler.transform(final_df)

        # Prediction
        prediction = model.predict(final_scaled)[0]

        probability = model.predict_proba(final_scaled)[0]

        st.divider()

        st.subheader("Prediction Result")

        if prediction == 1:

            st.success("✅ Loan Approved")

            st.metric(
                "Approval Probability",
                f"{probability[1]*100:.2f}%"
            )

        else:

            st.error("❌ Loan Rejected")

            st.metric(
                "Rejection Probability",
                f"{probability[0]*100:.2f}%"
            )

    except Exception as e:

        st.error(e)



