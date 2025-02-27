import streamlit as st
import pandas as pd 
import numpy as np
import sklearn
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
scaler = StandardScaler()
label_encoder = LabelEncoder()


# Load Model And Data  --------------------------------------
model = pickle.load(open("Loan_Model.pkl", 'rb'))


# Lets Create Web App ----------------------------------------
st.title("Scikit-Learn Regression Model")
age = st.text_input("Enter the Age :")
gender = st.selectbox("Choose the Gender As", options = ['Male' , ' Female'])
income = st.text_input("Enter the Income :")
credit_score = st.text_input("Enter the Credit_Score :")


# Helper Function ---------------------------------------------

def predictive(age,gender,income,credit_score):          # First we define the function for columns in which we are working 
   
    ## Encode the categorical columns 
    Encoded_gender = label_encoder.fit_transform([gender])[0]  # Then we encode the columns which are categorical I have only one categorical column 
         
    ## Prepare Features Array 
    features = np.array([[age,Encoded_gender,income,credit_score]])  # Then we convert are columns to 2d array 
    
    ## Scalling
    scaled_features = scaler.fit_transform(features)            # Then we do transform features columns 

    ## Predict by model
    result = model.predict(scaled_features)                  # Atlast we finally predict the model by logisicRegression 
    return result[0]


# Button ------------------------------------------------------
if st.button('Predict'):
    result=predictive(age,gender,income,credit_score)
    if result==0:
        st.markdown("Loan Approved: \n Now you get a loan:")
    else:
        st.write("Loan Denied: \n You cannot get a loan:")