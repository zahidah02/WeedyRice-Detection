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
        /* Custom Background Color */
        .stApp {
            background-color: #f9f9f9;
        }
        
        /* Sidebar Background */
        [data-testid="stSidebar"] {
            background-color: #e8f5e9;
        }

        /* Buttons */
        .stButton>button {
            background-color: #81C784;
            color: white;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #66BB6A;
        }

        /* Titles and headings */
        h1, h2, h3 {
            color: #33691E;
        }
    </style>
    """, unsafe_allow_html=True)

set_bg_and_style()

# ======================== HEADER ========================
st.title("Weedy and Cultivated Rice Classification Using YOLOv8 🌾")
st.markdown("---")

# ======================== SIDEBAR ========================
confidence = float(st.sidebar.slider("Select Confidence", 25, 100, 40)) / 100

# ======================== MODEL LOADING ========================
model_path = Path(settings.DETECTION_MODEL)
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"❌ Unable to load model. Check the path: {model_path}")
    st.error(ex)

# ======================== MAIN PAGE LAYOUT ========================
st.sidebar.header("📸 Image/Video Input")
source_radio = st.sidebar.radio("Choose Input Source", ["Image", "Video"])

col1, col2 = st.columns(2)

# ======================== IMAGE UPLOAD OR CAPTURE ========================
if source_radio == "Image":
    st.sidebar.subheader("📷 Upload or Capture Image")
    
    # OPTION 1: Upload Image from Library or File
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "webp"])
    
    # OPTION 2: Take Photo Using Camera
    captured_image = st.sidebar.camera_input("Take a photo")

    col1, col2 = st.columns(2)

    # Display Image
    with col1:
        if uploaded_file is not None:
            image = PIL.Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

        elif captured_image is not None:
            image = PIL.Image.open(captured_image)
            st.image(image, caption="Captured Image", use_column_width=True)

        else:
            st.info("Please upload an image or take a photo.")

    # Object Detection
    with col2:
        if (uploaded_file is not None or captured_image is not None) and st.sidebar.button("Detect Objects"):
            res = model.predict(image, conf=confidence)
            boxes = res[0].boxes
            res_plotted = res[0].plot()[:, :, ::-1]
            st.image(res_plotted, caption="Detected Image", use_column_width=True)

            try:
                with st.expander("🔍 Detection Results"):
                    for box in boxes:
                        st.write(box.data)
            except Exception as ex:
                st.write("No detections found!")

# ======================== VIDEO PROCESSING ========================
elif source_radio == "Video":
    helper.play_stored_video(confidence, model)

else:
    st.error("Please select a valid source type!")

# ======================== FOOTER ========================
st.markdown("---")
st.markdown('<p style="text-align:center; font-size:12px; color:#808080;">Developed for YOLOv8 Object Detection | 🌿 Agriculture AI Solution</p>', unsafe_allow_html=True)
