import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load the CNN model
cnn_model = load_model('cnn_model1.h5')

# App title and description
st.title("Brain Tumor Detection")
st.markdown("""
This AI-powered application uses deep learning to analyze MRI images and detect the presence of a brain tumor. 
""")

# File uploader
uploaded_file = st.file_uploader("Upload an MRI Image (JPG format)", type="jpg")

if uploaded_file:
    # Read and decode the uploaded image file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Display the uploaded image
    st.image(img, caption="Uploaded MRI Image", use_container_width=True)
    
    # Resize to 128x128 and normalize for the model input
    img_resized = cv2.resize(img, (128, 128))
    img_normalized = img_resized.reshape(1, 128, 128, 3) / 255.0

    # Get prediction
    prediction = cnn_model.predict(img_normalized)
    class_index = np.argmax(prediction, axis=1)[0]  # Get the index of the highest probability

    # Display results based on prediction
    if class_index == 1:
        st.markdown("### ⚠️ Tumor Detected")
    else:
        st.markdown("### ✅ No Tumor Detected")
