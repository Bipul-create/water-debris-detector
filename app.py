import streamlit as st
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from streamlit_image_comparison import image_comparison
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Precision Debris Mapper", layout="wide")

@st.cache_resource
def load_model():
    # Load SAHI model with a slightly lower threshold for better mapping
    return AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path='best.pt',
        confidence_threshold=0.20,
        device='cpu'
    )

model = load_model()

st.title("🛰️ High-Resolution Debris Hotspot Mapper")
uploaded_file = st.file_uploader("Upload Imagery", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Use PIL to open then convert to NumPy for OpenCV
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    h, w, _ = img_array.shape

    with st.spinner('Generating full-scale density map...'):
        # 1. Run Sliced Prediction
        result = get_sliced_prediction(
            img_array,
            model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2
        )

        # 2. Initialize Heatmap precisely to image dimensions
        heatmap = np.zeros((h, w), dtype=np.float32)

        for pred in result.object_prediction_list:
            bbox = pred.bbox.to_xyxy()
            # Ensure coordinates stay within image boundaries
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Increase 'heat' in the detected box area
            heatmap[y1:y2, x1:x2] += 10

        # 3. FIX: Gaussian Blur and Normalization
        # Larger images need larger blur kernels (must be odd numbers)
        heatmap = cv2.GaussianBlur(heatmap, (151, 151), 0)
        
        if heatmap.max() > 0:
            # Normalize to 0-255 and convert to uint8 (REQUIRED for cv2.applyColorMap)
            heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_uint8 = heatmap_norm.astype(np.uint8)
        else:
            heatmap_uint8 = np.zeros((h, w), dtype=np.uint8)

        # 4. Color Mapping and Blending
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Apply the heatmap only where detections exist (keep ocean clear otherwise)
        overlay = cv2.addWeighted(img_array, 0.7, heatmap_color, 0.3, 0)
        final_img = Image.fromarray(overlay)

    # 5. Full-Width Comparison
    image_comparison(
        img1=image,
        img2=final_img,
        label1="Original Photo",
        label2="Full Heatmap Map",
        width=1100
    )
