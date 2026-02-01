import streamlit as st
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title="High-Res Hotspot Mapper", layout="wide")

@st.cache_resource
def load_yolo_model():
    # SAHI needs the model in a specific wrapper
    return AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path='best.pt',
        confidence_threshold=0.15,
        device='cpu'
    )

model = load_yolo_model()

st.title("🛰️ High-Resolution Debris Hotspot Mapper")
uploaded_file = st.file_uploader("Upload High-Res Satellite Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    with st.spinner('Scanning high-resolution slices...'):
        # SAHI slices the image into 640x640 chunks
        result = get_sliced_prediction(
            img_array,
            model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )
        
        # Create Heatmap based on sliced results
        heatmap = np.zeros(img_array.shape[:2], dtype=np.float32)
        for object_prediction in result.object_prediction_list:
            bbox = object_prediction.bbox.to_xyxy()
            x1, y1, x2, y2 = map(int, bbox)
            heatmap[y1:y2, x1:x2] += 1

        # Process Heatmap
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Merge Heatmap with Original High-Res Image
        hotspot_map = cv2.addWeighted(img_array, 0.7, heatmap_color, 0.3, 0)

    st.image(hotspot_map, caption="Final High-Resolution Hotspot Map", use_container_width=True)
