import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Stellar Classification", layout="centered")

st.title("🌟 Stellar Classification Predictor")
st.write("Enter the characteristics of a star below to predict its type using the trained Random Forest model.")

# Load model
@st.cache_resource
def load_model():
    try:
        with open('svm_model.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}. Make sure to run stellar_classification.py first!")
        return None

model = load_model()

st.sidebar.header("Input Star Characteristics")

# Define inputs
temperature = st.sidebar.number_input("Temperature (K)", min_value=1000.0, max_value=50000.0, value=5000.0)
luminosity = st.sidebar.number_input("Luminosity (L/Lo)", min_value=0.0, max_value=1000000.0, value=1.0)
radius = st.sidebar.number_input("Radius (R/Ro)", min_value=0.0, max_value=2000.0, value=1.0)
abs_mag = st.sidebar.number_input("Absolute magnitude (Mv)", min_value=-20.0, max_value=25.0, value=5.0)

color_options = ['Red', 'Blue White', 'White', 'Yellowish White', 'Pale yellow orange', 'Blue', 'Blue-white', 'Whitish', 'yellow-white', 'Orange', 'White-Yellow', 'white', 'Blue ', 'Yellowish', 'Orange-Red', 'Blue white']
spectral_options = ['M', 'B', 'A', 'F', 'O', 'K', 'G']

star_color = st.sidebar.selectbox("Star color", color_options)
spectral_class = st.sidebar.selectbox("Spectral Class", spectral_options)

# Create dataframe
input_data = pd.DataFrame({
    'Temperature (K)': [temperature],
    'Luminosity (L/Lo)': [luminosity],
    'Radius (R/Ro)': [radius],
    'Absolute magnitude (Mv)': [abs_mag],
    'Star color': [star_color],
    'Spectral Class': [spectral_class]
})

st.write("### Your Input Data")
st.dataframe(input_data)

if st.button("Predict Star Type", type="primary"):
    if model is not None:
        prediction = model.predict(input_data)[0]
        
        star_types = {
            0: "Brown Dwarf",
            1: "Red Dwarf",
            2: "White Dwarf",
            3: "Main Sequence",
            4: "Supergiant",
            5: "Hypergiant"
        }
        
        predicted_name = star_types.get(prediction, "Unknown")
        st.success(f"### Predicted Type: {predicted_name} (Class {prediction})")
    else:
        st.warning("Model not found. Please train the model first.")
