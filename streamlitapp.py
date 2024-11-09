import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# Load your trained model (adjust the path as needed)
model_path = '/workspaces/WeaponDetection-Project/cnn_model1.h5'  # path to where your model is saved
model = tf.keras.models.load_model(model_path)

def preprocess_image(image, target_size=(128, 128)):
    """Preprocesses the uploaded image to the format required by the model."""
    # Convert the uploaded image to a NumPy array
    img = np.array(image)

    # Resize the image to the target size
    img_resized = cv2.resize(img, target_size)

    # Convert to grayscale (assuming the model expects grayscale input)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

    # Normalize pixel values to the 0-1 range
    img_normalized = img_gray / 255.0

    # Reshape for the model input (assuming the model expects grayscale input)
    img_reshaped = img_normalized.reshape((128, 128, 1))

    # Add batch dimension (1, 128, 128, 1)
    return np.array([img_reshaped])

def predict_tumor(image):
    """Predicts if a tumor is present in the image and applies a confidence threshold."""
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)

    # Define class labels based on your model's output
    class_labels = ['No Tumor', 'Tumor']

    # Get prediction probabilities
    predicted_probs = prediction[0]  # Model returns a list, we get the first element

    # Set a confidence threshold (e.g., 0.5)
    confidence_threshold = 0.5  # Adjust based on validation performance

    # Get the index of the class with the highest probability
    predicted_class_idx = np.argmax(predicted_probs)

    # If the highest confidence is below the threshold, classify as "No Tumor"
    if predicted_probs[predicted_class_idx] < confidence_threshold:
        return "No Tumor", predicted_probs

    # Return the predicted class and probabilities
    return class_labels[predicted_class_idx], predicted_probs

# Streamlit app interface
st.title("Brain Tumor Detection App")
st.write("Upload an MRI image to check if a tumor is detected.")

# Allow the user to upload an image
uploaded_file = st.file_uploader("Upload an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Predict if a tumor is present in the uploaded image
    result, confidence = predict_tumor(image)

    # Output the prediction and confidence scores
    st.write(f"**Prediction:** {result}")
    st.write(f"**Confidence Scores:** {confidence}")

    if result == 'Tumor':
        st.write("⚠️ Tumor Detected!")
    else:
        st.write("No Tumor Detected.")
