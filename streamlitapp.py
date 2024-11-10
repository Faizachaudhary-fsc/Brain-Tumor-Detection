import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load the CNN model
cnn_model = load_model('brain_tumor_cnn_model.h5')

st.title("Brain Tumor Detection")
uploaded_file = st.file_uploader("Upload MRI Image", type="jpg")

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_resized = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (128, 128))
    img_normalized = img_resized.reshape(1, 128, 128, 1) / 255.0

    prediction = cnn_model.predict(img_normalized)
    if prediction > 0.5:
        st.write("Tumor Detected")
    else:
        st.write("No Tumor Detected")
