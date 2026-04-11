import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score

import pickle
import streamlit as st



model = pickle.load(open('placement_prediction.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Placement Predictor")

cgpa = st.number_input("Enter CGPA")
iq = st.number_input("Enter IQ")

if st.button("Predict"):
    
    input_data = np.array([[cgpa, iq]])
    input_scaled = scaler.transform(input_data)
    
   
    prediction = model.predict(input_scaled)
    
    if prediction[0] == 1:
        st.success("Placed!")
    else:
        st.error("Not Placed")