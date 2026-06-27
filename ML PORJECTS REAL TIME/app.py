import gradio as gr
import pandas as pd
import pickle

# --- 1. Load the trained model pipeline ---
try:
    with open('svm_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None
    print("Error: 'svm_model.pkl' not found. Please run the training script first.")

# --- 2. Define the prediction function ---
def predict_star(temperature, luminosity, radius, abs_mag, star_color, spectral_class):
    if model is None:
        return "Error: Model not loaded. Check the terminal for details."
        
    # Pack the raw inputs into a DataFrame just like Scikit-Learn expects
    input_data = pd.DataFrame({
        'Temperature (K)': [temperature],
        'Luminosity (L/Lo)': [luminosity],
        'Radius (R/Ro)': [radius],
        'Absolute magnitude (Mv)': [abs_mag],
        'Star color': [star_color],
        'Spectral Class': [spectral_class]
    })
    
    # Pass the raw data directly to the pipeline
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
    return f"{predicted_name} (Class {prediction})"

# --- 3. Set up the Gradio UI ---
color_options = ['Red', 'Blue White', 'White', 'Yellowish White', 'Pale yellow orange', 'Blue', 'Blue-white', 'Whitish', 'yellow-white', 'Orange', 'White-Yellow', 'white', 'Blue ', 'Yellowish', 'Orange-Red', 'Blue white']
spectral_options = ['M', 'B', 'A', 'F', 'O', 'K', 'G']

# Map the inputs to Gradio components
inputs = [
    gr.Number(label="Temperature (K)", value=5000.0, minimum=1000.0, maximum=50000.0),
    gr.Number(label="Luminosity (L/Lo)", value=1.0, minimum=0.0, maximum=1000000.0),
    gr.Number(label="Radius (R/Ro)", value=1.0, minimum=0.0, maximum=2000.0),
    gr.Number(label="Absolute magnitude (Mv)", value=5.0, minimum=-20.0, maximum=25.0),
    gr.Dropdown(choices=color_options, label="Star color", value="Red"),
    gr.Dropdown(choices=spectral_options, label="Spectral Class", value="M")
]

# Build and launch the interface
demo = gr.Interface(
    fn=predict_star, 
    inputs=inputs,
    outputs=gr.Textbox(label="Predicted Star Type"),
    title="🌟 Stellar Classification Predictor",
    description="Enter the characteristics of a star below to predict its type using the trained SVM pipeline.",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    # Launch the app
    demo.launch()