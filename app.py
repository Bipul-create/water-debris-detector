import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="Marine Debris Hotspot Mapper", layout="wide")
st.title("🛰️ Satellite Debris Hotspot Detector")

# Load Model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# Sidebar: Lower threshold helps find smaller debris in satellite images
conf_threshold = st.sidebar.slider("Sensitivity (Confidence)", 0.05, 1.0, 0.15)
blur_intensity = st.sidebar.slider("Hotspot Blur", 5, 101, 51, step=2)

uploaded_file = st.file_uploader("Upload Satellite Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    with st.spinner('Calculating debris density...'):
        results = model.predict(image, conf=conf_threshold)
        
        # Create Heatmap Layer
        heatmap = np.zeros(img_array.shape[:2], dtype=np.float32)
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            heatmap[y1:y2, x1:x2] += 1  # Add "heat" to detected areas

        # Apply Blur for "Hotspot" effect
        heatmap = cv2.GaussianBlur(heatmap, (blur_intensity, blur_intensity), 0)
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        
        # Color the heatmap (Red=High Density)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay on original image
        hotspot_overlay = cv2.addWeighted(img_array, 0.7, heatmap_color, 0.3, 0)

    col1, col2 = st.columns(2)
    col1.image(image, caption="Original Satellite View")
    col2.image(hotspot_overlay, caption="Debris Hotspots")
