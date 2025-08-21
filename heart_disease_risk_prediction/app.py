import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and scaler
with open('best_heart_model_final.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("❤️ Heart Disease Risk Prediction App ❤️")
st.markdown("Enter the patient's details below to predict whether they are at High Risk or Low Risk of heart disease.")

st.header("Patient Information")

# Age
age = st.slider('Age (years)', 20, 100, 50)

# Sex
sex = st.selectbox('Sex', [0,1], format_func=lambda x: 'Female' if x==0 else 'Male')

# Chest Pain Type
st.markdown("""
**Chest Pain Type**  
1 = Typical Angina  
2 = Atypical Angina  
3 = Non-Anginal Pain  
4 = Asymptomatic
""")
chest_pain = st.selectbox('Select Chest Pain Type', [1,2,3,4])

# Resting Blood Pressure
resting_bp = st.slider('Resting Blood Pressure (mmHg)', 80, 200, 120)

# Cholesterol
cholesterol = st.slider('Cholesterol (mg/dl)', 100, 600, 200)

# Fasting Blood Sugar
st.markdown("**Fasting Blood Sugar > 120 mg/dl**: 1 = True, 0 = False")
fbs = st.selectbox('Fasting Blood Sugar', [0,1])

# Resting ECG
st.markdown("""
**Resting ECG Results**  
0 = Normal  
1 = ST-T wave abnormality  
2 = Probable or definite left ventricular hypertrophy
""")
rest_ecg = st.selectbox('Resting ECG', [0,1,2])

# Max Heart Rate
max_hr = st.slider('Max Heart Rate Achieved', 60, 210, 150)

# Exercise Induced Angina
st.markdown("**Exercise Induced Angina**: 1 = Yes, 0 = No")
ex_angina = st.selectbox('Exercise Angina', [0,1])

# Oldpeak
oldpeak = st.slider('ST Depression (Oldpeak)', 0.0, 10.0, 1.0, 0.1)

# ST Slope
st.markdown("""
**ST Slope**  
1 = Upsloping  
2 = Flat  
3 = Downsloping
""")
st_slope = st.selectbox('ST Slope', [1,2,3])

# Combine input
user_input = np.array([[age, sex, chest_pain, resting_bp, cholesterol, fbs,
                        rest_ecg, max_hr, ex_angina, oldpeak, st_slope]])

# Predict button
if st.button('Predict Risk'):
    # Scale numerical columns
    numerical_idx = [0,3,4,7,9]  # indices of numerical features
    user_input[:, numerical_idx] = scaler.transform(user_input[:, numerical_idx])
    
    prediction = model.predict(user_input)[0]
    risk_label = "High Risk" if prediction == 1 else "Low Risk"
    
    st.subheader(f"Predicted Heart Disease Risk: {risk_label}")
