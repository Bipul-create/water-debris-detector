import streamlit as st
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from streamlit_image_comparison import image_comparison
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Debris Hotspot Mapper", layout="wide")
st.title("🛰️ High-Resolution Debris Hotspot Mapper")

@st.cache_resource
def load_model():
    return AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path='best.pt',
        confidence_threshold=0.25,
        device='cpu'
    )

model = load_model()

uploaded_file = st.file_uploader("Upload Imagery", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    # IMPORTANT: Get original dimensions to create the perfect map
    height, width, _ = img_array.shape
    
    with st.spinner('Analyzing full image area...'):
        result = get_sliced_prediction(
            img_array,
            model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.15
        )
        
        # 1. Initialize a blank heatmap EXACTLY the size of the original
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        for pred in result.object_prediction_list:
            bbox = pred.bbox.to_xyxy()
            x1, y1, x2, y2 = map(int, bbox)
            # Clip coordinates to stay inside image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            # Map the intensity directly onto the full-sized layer
            heatmap[y1:y2, x1:x2] += 10 

        # 2. Smooth and Colorize (Ensures even the empty parts are mapped)
        heatmap = cv2.GaussianBlur(heatmap, (151, 151), 0)
        
        # Normalize: Maps 0-max values to 0-255 range across the WHOLE image
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        else:
            heatmap = heatmap.astype(np.uint8)
            
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # 3. Blend at 1:1 scale
        overlay = cv2.addWeighted(img_array, 0.6, heatmap_color, 0.4, 0)
        final_heatmap_img = Image.fromarray(overlay)

    # Display the comparison
    image_comparison(
        img1=image,
        img2=final_heatmap_img,
        label1="Original",
        label2="Full Heatmap Map",
        width=1000
    )
