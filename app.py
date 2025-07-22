import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.datasets import load_digits

# Load the trained model
model = joblib.load("digit_classifier.pkl")

# Load the digits dataset
digits = load_digits()

# Streamlit page setup
st.set_page_config(page_title="Digit Recognizer", page_icon="🔢")
st.title("🔢 Digit Recognizer Web App")
st.markdown("Predict handwritten digits (0–9) using a trained machine learning model.")

# Let user select an image
index = st.slider("Choose a digit image index", 0, len(digits.images) - 1, 0)

# Show the selected image (scaled to [0, 1])
st.image(digits.images[index] / 16.0, caption=f"Actual Digit: {digits.target[index]}", width=150)

# Prepare data for prediction
input_data = digits.data[index].reshape(1, -1)

# Predict using the trained model
prediction = model.predict(input_data)

# Display the prediction
st.success(f"🎯 Predicted Digit: {prediction[0]}")