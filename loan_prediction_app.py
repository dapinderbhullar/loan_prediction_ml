import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle

# Load and preprocess data
loan_dataset = pd.read_csv('loan_dataset.csv')
loan_dataset = loan_dataset.dropna()
loan_dataset.replace({"Loan_Status": {'N': 0, 'Y': 1}}, inplace=True)
loan_dataset['Dependents'].replace('3+', 4, inplace=True)
loan_dataset.replace({'Married': {'No': 0, 'Yes': 1},'Gender': {'Male': 1, 'Female': 0},'Self_Employed': {'No': 0, 'Yes': 1},'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},'Education': {'Graduate': 1, 'Not Graduate': 0}}, inplace=True)

X = loan_dataset.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
Y = loan_dataset['Loan_Status']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)

model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)

# Save the model
with open('loan_model_pickle.sav', 'wb') as file:
    pickle.dump(model, file)
print("Model saved successfully.")

# app.py
import streamlit as st
import numpy as np
import pickle

# Load trained model
with open('loan_model_pickle.sav', 'rb') as file:
    model = pickle.load(file)

# Streamlit App
st.title("🏦 Loan Approval Prediction App")
st.write("Enter the applicant's information to predict if their loan will be approved.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Number of Dependents", [0, 1, 2, 4])  # 4 represents '3+'
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["No", "Yes"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term (in days)", min_value=0)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

# Convert inputs to model format
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0
property_area = {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]

# Prediction
if st.button("Predict Loan Approval"):
    input_data = np.array([gender, married, dependents, education, self_employed,
                           applicant_income, coapplicant_income, loan_amount,
                           loan_term, credit_history, property_area]).reshape(1, -1)
    
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")
