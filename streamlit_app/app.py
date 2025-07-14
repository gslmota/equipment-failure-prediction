import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000/predict/single")
TRAIN_URL = os.getenv("TRAIN_URL", "http://api:8000/train/")

st.title("🏭 Equipment Failure Prediction")

with st.form("prediction_form"):
    st.header("Input Equipment Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        cycle = st.number_input("Cycle", min_value=1, value=100)
        temperature = st.number_input("Temperature (°C)", value=25.0)
        pressure = st.number_input("Pressure (psi)", value=100.0)
        vibration_x = st.number_input("Vibration X (mm/s)", value=10.0)
    
    with col2:
        vibration_y = st.number_input("Vibration Y (mm/s)", value=10.0)
        vibration_z = st.number_input("Vibration Z (mm/s)", value=10.0)
        frequency = st.number_input("Frequency (Hz)", value=50.0)
        preset_1 = st.number_input("Preset 1", min_value=1, value=1)
        preset_2 = st.number_input("Preset 2", min_value=1, value=1)
    
    submit = st.form_submit_button("Predict Failure Probability")

if submit:
    payload = {
        "Cycle": int(cycle),
        "Temperature": float(temperature),
        "Pressure": float(pressure),
        "VibrationX": float(vibration_x),
        "VibrationY": float(vibration_y),
        "VibrationZ": float(vibration_z),
        "Frequency": float(frequency),
        "Preset_1": int(preset_1),
        "Preset_2": int(preset_2)
    }
    
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            proba = result['failure_probability']
            risk_level = result['risk_level']
            
            st.subheader("Prediction Results")
            st.metric("Failure Probability", f"{proba:.2%}")
            
            if risk_level == "low":
                st.success("✅ Low failure risk")
            elif risk_level == "medium":
                st.warning("⚠️ Medium failure risk")
            else:
                st.error("🚨 High failure risk! Maintenance needed")
            
            st.progress(int(proba * 100))
        else:
            st.error(f"API Error: {response.text}")
    except Exception as e:
        st.error(f"Connection failed: {str(e)}")

st.divider()
st.header("Model Management")
if st.button("Retrain Model"):
    try:
        payload = {
            "data_path": "/home/gabriel/Área de Trabalho/shape_digital/failure_prediction_api/data.xlsx",
            "sheet_name": "O&G Equipment Data",
            "class_weights": [
                1,
                3
            ],
            "depth": 6,
            "l2_leaf_reg": 1,
            "learning_rate": 0.05,
            "window_size": 15
        }
        response = requests.post("http://localhost:8000/train/", json=payload)
        if response.status_code == 200:
            st.success("Model retraining started successfully!")
        else:
            st.error(f"Training failed: {response.text}")
    except:
        st.error("Could not connect to training API")