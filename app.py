# streamlit_app.py
import streamlit as st
import pickle
import numpy as np
from PIL import Image
from pipeline import YoloSegmentationPipeline

# Load pipeline from pickle
@st.cache_resource
def load_pipeline():
    with open("yolo_pipeline.pkl", "rb") as f:
        return pickle.load(f)

pipeline = load_pipeline()

# App Title
st.title("Image Segmentation Using YOLO")
st.markdown(
    """
    Welcome to the **Image Segmentation App** 👋  

    This app allows you to perform **object segmentation** on any image you upload, 
    or directly from your **camera input**.  
    The model used here is **YOLOv11x-seg**, a state-of-the-art segmentation model from **Ultralytics**.
    """
)

# Sidebar Information
st.sidebar.title("ℹ️ About This App")
st.sidebar.info(
    """
    **How to use:**
    1. Select **Upload Image** or **Use Camera**.  
    2. Provide an input image or take a photo.  
    3. View the segmented result instantly.  

    **Tech Stack:**  
    - YOLOv11x-seg (Ultralytics)  
    - Streamlit  
    - OpenCV, NumPy, Pillow  

    """
)

# Input source selection
option = st.sidebar.radio("Choose Input Source:", ["Upload Image", "Use Camera"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file:
        # Convert to numpy
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        # Run prediction
        results = pipeline.predict(image_np)
        segmented_image = pipeline.visualize_segmentation(results)

        # Show result only
        st.image(segmented_image, caption="Segmentation Result", use_column_width=True)

elif option == "Use Camera":
    camera_image = st.camera_input("Take a picture")

    if camera_image:
        # Convert to numpy
        image = Image.open(camera_image).convert("RGB")
        image_np = np.array(image)

        # Run prediction
        results = pipeline.predict(image_np)
        segmented_image = pipeline.visualize_segmentation(results)

        # Show result only
        st.image(segmented_image, caption="Segmentation Result", use_column_width=True)
