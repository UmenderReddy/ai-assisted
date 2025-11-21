import pandas as pd
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore') # Suppress scikit-learn warnings for simplicity

# --- 1. Model Training/Loading (Using the provided logic) ---

@st.cache_resource
def load_and_train_model():
    """Loads data, preprocesses it, and trains the Decision Tree model."""
    
    # Create the sample dataset
    data = {
        'Gender': ['Male', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
        'Married': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes'],
        'Credit_History': [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
        'ApplicantIncome': [5000, 4500, 6000, 3500, 7000, 4000, 5500, 6500, 4200, 7500],
        'Loan_Status': ['Y', 'N', 'Y', 'N', 'Y', 'Y', 'N', 'Y', 'N', 'Y']
    }
    df = pd.DataFrame(data)

    # Preprocessing for the Decision Tree
    # Note: In a real scenario, Credit_History would be more reliably handled than a simple fillna(1.0)
    df['Credit_History'] = df['Credit_History'].fillna(1.0) 
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})

    # Define features (X) and target (y)
    X = df[['Gender', 'Married', 'Credit_History', 'ApplicantIncome']]
    y = df['Loan_Status']

    # Train the Decision Tree Model
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)
    
    return model, X.columns.tolist()

# Load the model and feature names
model, feature_names = load_and_train_model()

# --- 2. Streamlit Web Interface Setup ---

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

st.title("🏦 Decision Tree Loan Approval Predictor")
st.markdown("---")
st.write("Use the controls below to input applicant data and predict the loan status.")

# --- Input Widgets ---

with st.sidebar:
    st.header("Applicant Information")

    # 1. Gender input
    gender = st.selectbox(
        'Gender:',
        ('Male', 'Female')
    )
    
    # 2. Marital Status input
    married = st.selectbox(
        'Marital Status:',
        ('Yes', 'No')
    )
    
    # 3. Applicant Income input
    applicant_income = st.number_input(
        'Applicant Income (USD)',
        min_value=1000, 
        max_value=100000, 
        value=5000, 
        step=500
    )

    # 4. Credit History input
    credit_history = st.radio(
        'Credit History (Met Debt Obligations):',
        (1.0, 0.0),
        format_func=lambda x: 'Good (1.0)' if x == 1.0 else 'Poor (0.0)',
        index=0 # Default to 1.0 (Good)
    )

# --- Prediction Logic ---

if st.button('Predict Loan Status'):
    # Convert categorical inputs to the format used by the model
    gender_encoded = 1 if gender == 'Male' else 0
    married_encoded = 1 if married == 'Yes' else 0

    # Prepare data for prediction (must match the training features order/format)
    # The order must be: 'Gender', 'Married', 'Credit_History', 'ApplicantIncome'
    features = np.array([[gender_encoded, married_encoded, credit_history, applicant_income]])

    try:
        # Make prediction
        prediction = model.predict(features)[0]

        # Determine the result message
        if prediction == 'Y':
            st.success("✅ *Prediction: Loan Approved!*")
            st.balloons()
        else:
            st.error("❌ *Prediction: Loan Rejected.*")

        st.markdown("---")
        st.subheader("Input Summary")
        summary_df = pd.DataFrame({
            'Feature': feature_names,
            'Value': [gender_encoded, married_encoded, credit_history, applicant_income]
        })
        st.dataframe(summary_df, hide_index=True, use_container_width=True)
        
        # Display the actual rule used (optional, requires tree export/visualization, but good for interpretability)
        st.info("The prediction was made based on the decision rules learned from the sample data.")

    except Exception as e:
        st.exception(f"An error occurred during prediction: {e}")