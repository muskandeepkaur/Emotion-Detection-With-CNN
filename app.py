
import streamlit as st
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("emotion_model.h5")

# Emotion labels (edit if needed)
labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Title
st.title("🎭 Emotion Detection from Facial Image")

# Upload image
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert to grayscale and resize
    img = Image.open(uploaded_file).convert("L").resize((48, 48))
    img_array = np.array(img).reshape(1, 48, 48, 1) / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = labels[np.argmax(prediction)]

    # Show results
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.markdown(f"### Predicted Emotion: **{predicted_class}**")
    st.bar_chart(prediction[0])
