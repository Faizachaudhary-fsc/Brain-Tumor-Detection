import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Load the saved models
cnn_model = tf.keras.models.load_model('cnn_model1.h5')
resnet50_model = tf.keras.models.load_model('resnet50_model.h5')

# Define image preprocessing function
def preprocess_image(image, target_size=(128, 128)):
    image = image.resize(target_size)
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app setup
st.title("Brain Tumor Detection")
st.write("Upload an MRI image to check for brain tumor.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type="jpg")

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)

    # Preprocess the image for model input
    preprocessed_image = preprocess_image(image)

    # Select Model
    model_choice = st.selectbox("Choose a model for prediction:", ["CNN", "ResNet50"])

    # Make prediction based on model choice
    if model_choice == "CNN":
        prediction = cnn_model.predict(preprocessed_image)
    elif model_choice == "ResNet50":
        prediction = resnet50_model.predict(preprocessed_image)
    
    # Get prediction class
    prediction_class = np.argmax(prediction, axis=1)[0]

    # Show the result
    if prediction_class == 1:
        st.write("### Result: ♋ Tumor Detected")
    else:
        st.write("### Result: No Tumor Detected")
