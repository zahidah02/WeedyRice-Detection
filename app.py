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
            background-color: #f9f9f9;  /* Very light grey */
        }
        
        /* Sidebar Background */
        [data-testid="stSidebar"] {
            background-color: #e8f5e9;  /* Soft green */
        }

        /* Title Styling */
        .st-emotion-cache-10trblm {
            color: #4CAF50;  /* Green */
            font-weight: bold;
        }
        
        /* Footer Text */
        .footer {
            font-size: 12px;
            text-align: center;
            color: #808080;  /* Grey */
        }

        /* Buttons */
        .stButton>button {
            background-color: #81C784;  /* Medium green */
            color: white;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #66BB6A;  /* Slightly darker green on hover */
            color: #FFFFFF;
        }

        /* Titles and headings */
        h1, h2, h3 {
            color: #33691E;  /* Deep green */
        }

        /* Uploaded image and detection results */
        .uploadedImage, .detectedImage {
            border: 2px solid #81C784;  /* Medium green border */
            border-radius: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

set_bg_and_style()

# ======================== TOP BANNER & HEADER ========================
st.title("Weedy and Cultivated Rice Classification Using YOLOv8🌾")
st.markdown("---")

# ======================== SIDEBAR ========================
# st.sidebar.header("⚙️ YOLO Configuration")
confidence = float(st.sidebar.slider("Select Confidence", 25, 100, 40)) / 100

# ======================== DEFAULT SETTINGS ========================
model_path = Path(settings.DETECTION_MODEL)
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"❌ Unable to load model. Check the path: {model_path}")
    st.error(ex)

# ======================== MAIN PAGE LAYOUT ========================
st.sidebar.header("Image/Video Input")
source_radio = st.sidebar.radio("Choose Input Source", ["Image", "Video"])

col1, col2 = st.columns(2)

# Image Source
# Image Source
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image")
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image")  # Removed className
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)
    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image')
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image')  # Removed className
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

else:
    st.error("Please select a valid source type!")

# ======================== FOOTER ========================
st.markdown("---")
st.markdown('<p class="footer">Developed for YOLOv8 Object Detection | 🌿 Agriculture AI Solution</p>', unsafe_allow_html=True)
