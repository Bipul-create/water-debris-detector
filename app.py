import streamlit as st
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import numpy as np
from PIL import Image

# ... (Previous model loading code) ...

def run_large_image_inference(image, conf):
    # Load model into SAHI format
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path='best.pt',
        confidence_threshold=conf,
        device='cpu' # Use 'cuda' if GPU is available
    )
    
    # SAHI slices the image into 640x640 chunks with 20% overlap
    # This prevents debris from being cut in half
    result = get_sliced_prediction(
        image,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )
    
    return result

if uploaded_file:
    # Open image but don't keep multiple copies to save RAM
    image = Image.open(uploaded_file).convert("RGB")
    
    with st.spinner('Scanning high-resolution image...'):
        # 1. Run the sliced inference
        prediction = run_large_image_inference(image, conf_threshold)
        
        # 2. Get the processed image with boxes drawn
        annotated_img = prediction.export_visual(export_dir=None)
        
        st.image(annotated_img["image"], caption="High-Res Detection Results")
