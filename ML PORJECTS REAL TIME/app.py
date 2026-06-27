import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

st.set_page_config(page_title="Stellar Classification", layout="centered")

st.title("🌟 Stellar Classification Predictor")
st.write("Enter the characteristics of a star below to predict its type using the trained model.")

# --- 1. Load Model (We still need the trained model) ---
try:
    with open('svm_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'svm_model.pkl' not found. Please ensure the model file is in the same directory.")
    model = None

# --- 2. Dynamic Preprocessor Setup ---
@st.cache_data
def get_fitted_preprocessor():
    """Loads the original dataset and fits the scaling pipeline on the fly."""
    csv_path = 'data/Stars.csv'
    
    if not os.path.exists(csv_path):
        return None
        
    # Load original training data
    df = pd.read_csv(csv_path)
    X_train = df.drop(['Star type', 'Star category'], axis=1, errors='ignore')
    
    # Define exact columns from training
    categorical_cols = ['Star color', 'Spectral Class']
    numerical_cols = ['Temperature (K)', 'Luminosity (L/Lo)', 'Radius (R/Ro)', 'Absolute magnitude (Mv)']

    # Create and fit the preprocessor using the training data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    preprocessor.fit(X_train)
    return preprocessor

# Initialize the dynamic preprocessor
preprocessor = get_fitted_preprocessor()

if preprocessor is None:
    st.warning("Could not find 'data/Stars.csv'. The app needs the original data to scale inputs properly.")

# --- 3. Sidebar Inputs ---
st.sidebar.header("Input Star Characteristics")

temperature = st.sidebar.number_input("Temperature (K)", min_value=1000.0, max_value=50000.0, value=5000.0)
luminosity = st.sidebar.number_input("Luminosity (L/Lo)", min_value=0.0, max_value=1000000.0, value=1.0)
radius = st.sidebar.number_input("Radius (R/Ro)", min_value=0.0, max_value=2000.0, value=1.0)
abs_mag = st.sidebar.number_input("Absolute magnitude (Mv)", min_value=-20.0, max_value=25.0, value=5.0)

color_options = ['Red', 'Blue White', 'White', 'Yellowish White', 'Pale yellow orange', 'Blue', 'Blue-white', 'Whitish', 'yellow-white', 'Orange', 'White-Yellow', 'white', 'Blue ', 'Yellowish', 'Orange-Red', 'Blue white']
spectral_options = ['M', 'B', 'A', 'F', 'O', 'K', 'G']

star_color = st.sidebar.selectbox("Star color", color_options)
spectral_class = st.sidebar.selectbox("Spectral Class", spectral_options)

# --- 4. Prediction Logic ---
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
    if model is not None and preprocessor is not None:
        
        # Scale the single input using the preprocessor fitted on the full dataset
        scaled_input = preprocessor.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_input)[0]
        
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