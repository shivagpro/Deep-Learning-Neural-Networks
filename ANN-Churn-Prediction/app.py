"""
Customer Churn Prediction using ANN

This Streamlit application allows users to input customer details and predict whether a customer is likely to churn 
using a pre-trained Artificial Neural Network (ANN) model. The model utilizes various customer attributes, including 
credit score, balance, tenure, number of products, and more, to generate a probability score for churn.

Author: Shivani Tyagi
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit App Title
st.title('Customer Churn Prediction')

# User input selection
# Select geography from available categories
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])

# Select gender from available classes
gender = st.selectbox('Gender', label_encoder_gender.classes_)

# Select age using a slider (range: 18-92)
age = st.slider('Age', 18, 92)

# Input account balance
balance = st.number_input('Balance')

# Input credit score
credit_score = st.number_input('Credit Score')

# Input estimated salary
estimated_salary = st.number_input('Estimated Salary')

# Select tenure (number of years as a customer) using a slider (range: 0-10)
tenure = st.slider('Tenure', 0, 10)

# Select number of products used (range: 1-4)
num_of_products = st.slider('Number of Products', 1, 4)

# Select whether the customer has a credit card (0 = No, 1 = Yes)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])

# Select whether the customer is an active member (0 = No, 1 = Yes)
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data DataFrame
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode the selected geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Merge one-hot encoded geography data with the main input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data using the pre-loaded StandardScaler
input_data_scaled = scaler.transform(input_data)

# Predict churn using the trained ANN model
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Display churn probability
st.write(f'Churn Probability: {prediction_proba:.2f}')

# Interpret and display the prediction result
if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')