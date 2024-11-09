import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import pickle

# Load your trained models
cnn_model_path = '/workspaces/WeaponDetection-Project/cnn_model1.h5'  # path to your CNN model
svm_model_path = '/workspaces/WeaponDetection-Project/svm_model1.pkl'  # path to your SVM model

cnn_model = tf.keras.models.load_model(cnn_model_path)

# Load the SVM model using pickle
with open(svm_model_path, 'rb') as f:
    svm_model = pickle.load(f)

def preprocess_image(image, target_size=(60, 60)):
    """Preprocesses the uploaded MRI image to the format required by the model."""
    # Convert the uploaded image to a NumPy array
    img = np.array(image)

    # Resize the image to the target size
    img_resized = cv2.resize(img, target_size)

    # Convert to grayscale if necessary (check if the model expects grayscale)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

    # Normalize pixel values to the 0-1 range
    img_normalized = img_gray / 255.0

    # Reshape for the model input (assuming the model expects grayscale input)
    img_reshaped = img_normalized.reshape((60, 60, 1))

    # Add batch dimension (1, 60, 60, 1)
    return np.array([img_reshaped])

def predict_tumor(image, model_choice):
    """Predicts if a brain tumor is detected based on the selected model."""
    processed_image = preprocess_image(image)
    
    if model_choice == 'CNN':
        prediction = cnn_model.predict(processed_image)
        prediction_class = np.argmax(prediction, axis=1)[0]
    elif model_choice == 'SVM':
        # Flatten the image for SVM (if required)
        flattened_image = processed_image.flatten().reshape(1, -1)
        prediction_class = svm_model.predict(flattened_image)[0]
    
    return prediction_class

# Streamlit app interface
st.title("Brain Tumor Detection App")

# Allow the user to upload an image
uploaded_file = st.file_uploader("Upload an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Model selection for prediction
    model_choice = st.selectbox("Choose a model for prediction:", ["CNN", "SVM"])

    # Predict tumor presence based on the selected model
    prediction_class = predict_tumor(image, model_choice)

    # Output the result based on prediction class
    if prediction_class == 1:
        st.write("### Result: Tumor Detected")
    else:
        st.write("### Result: No Tumor Detected")

