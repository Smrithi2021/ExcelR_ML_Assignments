import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load trained model

model= joblib.load("logistic_model.pkl")


# Streamlit UI

st.title("Logistic Regression Model deployment")

# Create user input fields

st.write("Enter feature values for prediction: ")

# Collect user input

feature_1 = st.number_input("Feature 1", min_value=0.0, format="%.2f")
feature_2 = st.number_input("Feature 2", min_value=0.0, format="%.2f")
feature_3 = st.number_input("Feature 3", min_value=0.0, format="%.2f")
feature_4 = st.number_input("Feature 4", min_value=0.0, format="%.2f")
feature_5 = st.number_input("Feature 5", min_value=0.0, format="%.2f")
feature_6 = st.number_input("Feature 6", min_value=0.0, format="%.2f")
feature_7 = st.number_input("Feature 7", min_value=0.0, format="%.2f")
feature_8 = st.number_input("Feature 8", min_value=0.0, format="%.2f")

# Button to make predictions

if st.button("Predict"):
    # Prepare input data
    input_data = np.array([[feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]])

    # Make prediction
    prediction = model.predict(input_data)

# Display result
    if prediction[0]==1:
        st.success('Prediction: Class 1 (Positive)')
    else:
        st.success('Prediction: Class 0 (Negative)')



