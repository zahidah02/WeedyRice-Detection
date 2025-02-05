# Python In-built packages
from pathlib import Path
import PIL
import pandas as pd

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# ======================== PAGE CONFIGURATION ========================
st.set_page_config(
    page_title="Weedy and Cultivated Rice Classification",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================== CSS Styling ========================
def set_bg_and_style():
    st.markdown("""
    <style>
        .stApp { background-color: #f9f9f9; }
        [data-testid="stSidebar"] { background-color: #e8f5e9; }
        .st-emotion-cache-10trblm { color: #4CAF50; font-weight: bold; }
        .footer { font-size: 12px; text-align: center; color: #808080; }
        .stButton>button { background-color: #81C784; color: white; border-radius: 8px; font-size: 16px; font-weight: bold; }
        .stButton>button:hover { background-color: #66BB6A; color: #FFFFFF; }
        h1, h2, h3 { color: #33691E; }
        .uploadedImage, .detectedImage { border: 2px solid #81C784; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

set_bg_and_style()

# ======================== HEADER ========================
st.title("Weedy and Cultivated Rice Classification Using YOLOv8 🌾")
st.markdown("---")

# ======================== SIDEBAR ========================
st.sidebar.header("Input Source")
source_radio = st.sidebar.radio("Choose Input Method", ["Upload Image", "Take Photo"])

confidence = float(st.sidebar.slider("Select Confidence", 25, 100, 40)) / 100

# ======================== MODEL LOADING ========================
model_path = Path(settings.DETECTION_MODEL)
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"❌ Unable to load model. Check the path: {model_path}")
    st.error(ex)

# ======================== IMAGE INPUT ========================
uploaded_image = None

if source_radio == "Upload Image":
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "webp"])
elif source_radio == "Take Photo":
    uploaded_image = st.camera_input("Capture an image")

# ======================== IMAGE PROCESSING ========================
if uploaded_image:
    try:
        # Convert image to PIL format
        image = PIL.Image.open(uploaded_image)

        # Display uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Detect button
        if st.button("Detect Objects"):
            res = model.predict(image, conf=confidence)
            res_plotted = res[0].plot()[:, :, ::-1]  # Convert for display
            st.image(res_plotted, caption="Detected Image", use_column_width=True)

            # Display detection results
            with st.expander("Detection Results"):
                for box in res[0].boxes:
                    st.write(box.data)
    except Exception as ex:
        st.error("❌ Error processing the image. Try another one.")
        st.error(ex)

# ======================== FOOTER ========================
st.markdown("---")
st.markdown('<p class="footer">Developed for YOLOv8 Object Detection | 🌿 Agriculture AI Solution</p>', unsafe_allow_html=True)
