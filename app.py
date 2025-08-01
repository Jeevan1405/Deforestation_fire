import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("best_fire_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

# Set page title and favicon
st.set_page_config(page_title="Fire Type Classifier", page_icon="🔥", layout="centered")

# --- Professional Front-End Design ---

# App title and a more descriptive subtitle
st.title("🔥 Fire Type Classification")
st.markdown(
    "This application leverages a machine learning model to predict the type of fire based on real-time MODIS satellite sensor readings. "
    "Input the following parameters to classify the fire event."
)
st.markdown("---")

# Organized input fields into columns for a cleaner look
st.subheader("MODIS Satellite Readings")
col1, col2 = st.columns(2)

with col1:
    brightness = st.number_input("Brightness (Kelvin)", value=300.0, help="Brightness temperature of the fire pixel in Kelvin.")
    bright_t31 = st.number_input("Brightness T31 (Kelvin)", value=290.0, help="Brightness temperature of the fire pixel in the T31 channel (Kelvin).")
    frp = st.number_input("Fire Radiative Power (MW)", value=15.0, help="Fire Radiative Power in megawatts.")

with col2:
    scan = st.number_input("Scan (km)", value=1.0, help="The scan angle of the satellite.")
    track = st.number_input("Track (km)", value=1.0, help="The track of the satellite.")
    confidence = st.selectbox("Confidence Level", ["low", "nominal", "high"], index=1, help="The confidence level of the fire detection.")

# Map confidence to numeric
confidence_map = {"low": 0, "nominal": 1, "high": 2}
confidence_val = confidence_map[confidence]

# Combine and scale input
input_data = np.array([[brightness, bright_t31, frp, scan, track, confidence_val]])
scaled_input = scaler.transform(input_data)

st.markdown("---")

# Predict and display with a more prominent button and clear output
if st.button("Predict Fire Type", key="predict_button"):
    with st.spinner("Analyzing..."):
        prediction = model.predict(scaled_input)[0]

        fire_types = {
            0: "Vegetation Fire",
            2: "Other Static Land Source",
            3: "Offshore Fire"
        }

        result = fire_types.get(prediction, "Unknown")
        st.success(f"**Predicted Fire Type:** {result}")