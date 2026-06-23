import streamlit as st
import pandas as pd
import pickle

# --- 1. Load the Saved Tools ---
# st.cache_resource ensures the model only loads once, keeping your app fast!
@st.cache_resource
def load_model():
    with open('taxi_preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    with open('taxi_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return preprocessor, model

preprocessor, model = load_model()

# --- 2. Build the Web Interface ---
st.title("🚕 NYC Taxi Destination Predictor")
st.markdown("Enter the trip details below to predict where the taxi is heading.")

# Create two columns for a clean, modern layout
col1, col2 = st.columns(2)

with col1:
    pickup = st.selectbox("Pickup Borough", ['Manhattan', 'Queens', 'Brooklyn', 'Bronx'])
    passengers = st.number_input("Number of Passengers", min_value=1, max_value=6, value=1)

with col2:
    distance = st.number_input("Distance (miles)", min_value=0.1, max_value=50.0, value=2.5, step=0.1)
    fare = st.number_input("Fare Amount ($)", min_value=2.50, max_value=200.0, value=12.50, step=0.50)

st.divider()

# --- 3. Make the Prediction ---
# This block runs only when the user clicks the button
if st.button("Predict Dropoff Destination", use_container_width=True):
    
    # Format the input exactly like our training data
    input_data = pd.DataFrame([{
        'pickup_borough': pickup,
        'distance': distance,
        'fare': fare,
        'passengers': passengers
    }])
    
    try:
        # Process the data and predict
        processed_data = preprocessor.transform(input_data)
        prediction = model.predict(processed_data)[0]
        
        # Display the result beautifully
        st.success(f"🔮 The model predicts the dropoff will be in: **{prediction}**")
        
    except Exception as e:
        st.error(f"An error occurred with the input data: {e}")