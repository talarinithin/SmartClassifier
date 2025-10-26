# ===============================
# CIFAR-10 Image Classification App
# ===============================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# -------------------------------
# 1️⃣ Page config
# -------------------------------
st.set_page_config(
    page_title="Image Classification",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -------------------------------
# 2️⃣ CSS Styling
# -------------------------------
# -------------------------------
# 2️⃣ CSS Styling
# -------------------------------
st.markdown("""
<style>
    /* ---------------------------------------------------- */
    /* 1. LAYOUT & BACKGROUND (Dark Theme) */
    
    /* Main container background - Deep Charcoal */
    [data-testid="stAppViewContainer"] {
        background-color: #1E1E1E; /* Dark background */
        color: #F8F8F8; /* Light gray text for readability */
    }
    
    /* Sidebar background - Slightly lighter Dark Gray */
    [data-testid="stSidebar"] {
        background-color: #2D2D30; 
    }

    /* Titles and Headers - Use light color for contrast */
    h1, h2, h3 {
        color: #FFFFFF; /* Pure White titles */
    }
    
    /* Text Input/Select Boxes - Ensure they look good on dark mode */
    .stTextInput, .stSelectbox {
        background-color: #3C3C3C;
        color: #F8F8F8;
    }
    
    /* ---------------------------------------------------- */
    /* 2. BUTTONS & INTERACTIVITY (Primary Blue Accent) */

    /* Style the main button primary color */
    .stButton>button {
        background-color: #00AFFF; /* Vibrant Primary Blue */
        color: black; /* Dark text on bright button for high contrast */
        border-radius: 8px; 
        border: none;
        padding: 10px 20px;
        font-weight: 700; 
        transition: background-color 0.3s, transform 0.2s, box-shadow 0.3s;
    }

    /* Button hover effect */
    .stButton>button:hover {
        background-color: #0077B6; /* Slightly darker blue on hover */
        transform: translateY(-2px); 
        box-shadow: 0 5px 15px rgba(0, 175, 255, 0.3); /* Blue glowing shadow */
    }

    /* ---------------------------------------------------- */
    /* 3. FILE UPLOADER (Subtle contrast) */

    /* Target the file uploader widget */
    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #6A6A6A; /* Medium Gray dashed border */
        border-radius: 10px;
        padding: 30px;
        background-color: #2D2D30; /* Matches sidebar for consistency */
        transition: border-color 0.3s, background-color 0.3s;
        color: #F8F8F8; /* Ensure text inside is light */
    }

    /* File Uploader hover effect */
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #00AFFF; /* Primary blue border on hover */
        background-color: #3C3C3C; 
    }

    /* ---------------------------------------------------- */
    /* 4. PREDICTION RESULTS BOX (Highlighting Success) */
    
    /* Custom class for the results display card */
    .prediction-card {
        background-color: #2D2D30; /* Dark box for results */
        border-left: 5px solid #48FF48; /* Bright Success Green accent bar */
        border-radius: 8px;
        padding: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4); /* Stronger shadow in dark mode */
        margin-top: 15px;
    }

    .prediction-label {
        color: #B0B0B0; /* Light Gray label */
        font-size: 18px;
        margin-bottom: 5px;
    }

    .prediction-value {
        color: #48FF48; /* Bold Success Green for the result */
        font-size: 36px;
        font-weight: 800;
        margin-top: 5px;
    }
    
</style>
""", unsafe_allow_html=True)



# -------------------------------
# 3️⃣ Title & Subtitle
# -------------------------------
st.title("📷 Image Classification")
st.markdown("**Created by Talari Nithin**")
st.markdown("---")

# -------------------------------
# 4️⃣ Load model & labels
# -------------------------------
EXPORT_DIR = "cifar10_model"  # folder containing cifar10_model.keras and labels.json

# Load model (Keras 3 format)
# Model path (same folder)
model_path = "cifar10_model.keras"

# Load model
model = tf.keras.models.load_model(model_path)

# Load labels
with open("labels.json", "r") as f:
    class_names = json.load(f)


# -------------------------------
# 5️⃣ Sidebar info
# -------------------------------
st.sidebar.header("Project Info")
st.sidebar.write("Dataset: CIFAR-10")
st.sidebar.write("Model: CNN (Conv2D + MaxPooling + Dense + Dropout)")
st.sidebar.write("Frameworks: TensorFlow & Streamlit")
st.sidebar.write("Image Size for Model: 32x32")

# -------------------------------
# 6️⃣ Image upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Show original image
    st.image(uploaded_file, caption="Original Image", use_container_width=True)

    # Preprocess for model
    img = Image.open(uploaded_file).convert("RGB").resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    idx = np.argmax(preds)
    confidence = np.max(preds) * 100

    # Prediction box
    st.markdown(
        f"""
        <div class="prediction-box">
            <h3>✅ Predicted Class: {class_names[idx]}</h3>
            <h4>Confidence: {confidence:.2f}%</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Footer prediction
    st.markdown(
        f"<p class='prediction'>Predicted: {class_names[idx]} ({confidence:.2f}%)</p>",
        unsafe_allow_html=True
    )
    st.markdown(f"""
    <div class="prediction-card">
        <div class="prediction-label">Highest Confidence Prediction:</div>
        <div class="prediction-value">{class_names[idx]}</div>
        <div class="prediction-label">Confidence: {confidence:.2f}</div>
    </div>
""", unsafe_allow_html=True)