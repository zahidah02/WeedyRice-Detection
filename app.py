# Python In-built packages
from pathlib import Path
import PIL
from PIL import Image
import numpy as np
import pandas as pd
import io

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
        .stButton>button { background-color: #81C784; color: white; border-radius: 8px; font-size: 16px; }
        .stButton>button:hover { background-color: #66BB6A; }
        h1, h2, h3 { color: #33691E; }
    </style>
    """, unsafe_allow_html=True)

set_bg_and_style()

# ======================== HEADER ========================
st.title("Weedy and Cultivated Rice Classification Using YOLOv8 🌾")
st.markdown("---")

# ======================== SIDEBAR ========================
st.sidebar.header("Image/Video Input")
confidence = float(st.sidebar.slider("Select Confidence", 25, 100, 40)) / 100

# ======================== DEFAULT SETTINGS ========================
model_path = Path(settings.DETECTION_MODEL)
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"❌ Unable to load model. Check the path: {model_path}")
    st.error(ex)

# ======================== MAIN PAGE LAYOUT ========================
source_radio = st.sidebar.radio("Choose Input Source", ["Image", "Video"])

col1, col2 = st.columns(2)

if source_radio == "Image":
    source_img = st.sidebar.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", "bmp", "webp"))
    uploaded_image = None

    with col1:
        if source_img is None:
            default_image_path = str(settings.DEFAULT_IMAGE)
            default_image = Image.open(default_image_path)
            st.image(default_image, caption="Default Image")
        else:
            try:
                image_bytes = source_img.read()
                uploaded_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Convert to RGB
                uploaded_image = np.array(uploaded_image)[:, :, ::-1]  # Convert RGB to BGR if needed
                st.image(Image.fromarray(uploaded_image[:, :, ::-1]), caption="Uploaded Image")
            except Exception as ex:
                st.error("Error occurred while opening the image.")
                st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = Image.open(default_detected_image_path)
            st.image(default_detected_image, caption="Detected Image")
        elif uploaded_image is not None and st.sidebar.button("Detect Objects"):
            try:
                res = model.predict(uploaded_image, conf=confidence)
                res_plotted = res[0].plot()
                res_plotted = Image.fromarray(res_plotted)  # Convert to PIL Image
                st.image(res_plotted, caption="Detected Image")
                
                with st.expander("Detection Results"):
                    for box in res[0].boxes:
                        st.write(box.data)
            except Exception as ex:
                st.error("Error during object detection.")
                st.error(ex)

elif source_radio == "Video":
    helper.play_stored_video(confidence, model)
else:
    st.error("Please select a valid source type!")

# ======================== FOOTER ========================
st.markdown("---")
st.markdown('<p class="footer">Developed for YOLOv8 Object Detection | 🌿 Agriculture AI Solution</p>', unsafe_allow_html=True)
