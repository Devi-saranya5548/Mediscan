import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import os

# Load trained model (Ensure the model is in the same directory or provide the correct path)
MODEL_PATH = "eye_disease_model.h5"
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    st.error("Model file not found! Please upload the trained model.")
    st.stop()

# Define class labels
CLASS_LABELS = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]

# Streamlit UI
st.title("MediScan: AI-Powered Medical Diagnosis for Eye Diseases")
st.write("Upload an eye image to get a diagnosis.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to an image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess image
    img = image.resize((224, 224))  # Resize to match model input size
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Normalize using VGG16 preprocess function
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = CLASS_LABELS[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    
    # Display result
    st.write(f"### Prediction: {predicted_class}")
    st.write(f"### Confidence: {confidence:.2f}%")
