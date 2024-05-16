# -*- coding: utf-8 -*-
"""
Created on Thu May 16 16:05:13 2024

@author: hp
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the model
model = joblib.load('best_model.pkl')

# Define the features based on the model training data
feature_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Create the Streamlit app
st.title("Heart Disease Prediction")

st.write("Enter the details of the patient to predict the likelihood of heart disease.")

# Create input fields for each feature
def user_input_features():
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type (CP)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=50, max_value=200, value=120)
    chol = st.number_input("Serum Cholesterol (chol)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
    restecg = st.selectbox("Resting Electrocardiographic Results (restecg)", [0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
    oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment (slope)", [0, 1, 2])
    ca = st.number_input("Number of Major Vessels (ca)", min_value=0, max_value=4, value=0)
    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])
    
    # Convert sex and thal inputs to numerical values for the model
    sex = 1 if sex == "Male" else 0
    
    # Create a dictionary of inputs
    data = {'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal}
    
    # Convert to DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user inputs
st.subheader("Patient's Input Features")
st.write(input_df)

# Standardize the input features using the same scaler used during training
scaler = StandardScaler()
scaled_features = scaler.fit_transform(input_df)

# Predict the likelihood of heart disease
prediction = model.predict(scaled_features)
prediction_proba = model.predict_proba(scaled_features)

st.subheader("Prediction")
heart_disease = np.array(['No Heart Disease', 'Heart Disease'])
st.write(heart_disease[prediction][0])

st.subheader("Prediction Probability")
st.write(prediction_proba)
