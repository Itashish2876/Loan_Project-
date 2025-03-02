import streamlit as st
import pandas as pd 
import numpy as np
import sklearn
import pickle
from sklearn.preprocessing import LabelEncoder

# Custom CSS for styling
st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(to right, #f0f2f5, #e1e5ea); /* Light gradient background */
    }
    .main {
        background-color: rgba(255, 255, 255, 0.8); /* Semi-transparent white panel */
        padding: 20px;
        border-radius: 10px;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #333; /* Dark gray headings */
        font-family: 'Arial', sans-serif;
    }
    label {
        font-weight: bold;
    }
    .stButton>button {
        color: #fff;
        background-color: #007BFF; /* Blue button */
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input {
        border-radius: 5px;
        border: 1px solid #ccc;
        padding: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)




# Load Model And Data  --------------------------------------
scaler = pickle.load(open("loan_scale.pkl",'rb'))
model = pickle.load(open("Loan_Model.pkl", 'rb'))


label_encoder = LabelEncoder()

# Lets Create Web App ----------------------------------------
# Main Streamlit app

# Main content area
st.markdown('<div class="main">', unsafe_allow_html=True)

st.title("Loan Prediction Regression Model")
age = st.number_input("Enter the Age:", min_value=0)
gender = st.selectbox("Choose the Gender As", options = ['Male' , ' Female'])
income = st.number_input("Enter the Income:", min_value=0.0)
credit_score = st.number_input("Enter the Credit Score:", min_value=0)



# Helper Function ---------------------------------------------

def predictive(age,gender,income,credit_score):          # First we define the function for columns in which we are working 
   
    ## Encode the categorical columns 
    Encoded_gender = label_encoder.fit_transform([gender])[0]  # Then we encode the columns which are categorical I have only one categorical column 
    print(Encoded_gender)
    ## Prepare Features Array 
    features = np.array([[age,Encoded_gender,income,credit_score]])  # Then we convert are columns to 2d array 
    print(features)
    
    ## Scalling
    scaled_features = scaler.transform(features)            # Then we do transform features columns 
    print(scaled_features)
    ## Predict by model
    result = model.predict(scaled_features)                  # Atlast we finally predict the model by logisicRegression 
    print(result)
    return result[0]


# Predict Button ------------------------------------------------------
if st.button('Predict'):
    result=predictive(age,gender,income,credit_score)
    if result ==1:
        st.write("Loan Denied.\nYou cannot get a loan :")
    else:
        st.write("Loan Approved: \nNow you get a loan :")

st.markdown("</div>", unsafe_allow_html=True)


