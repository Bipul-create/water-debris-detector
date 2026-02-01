import streamlit as st
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from streamlit_image_comparison import image_comparison
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Marine Debris Density Mapper", layout="wide")
st.title("🛰️ Satellite Debris Density & Hotspot Mapper")

@st.cache_resource
def load_sahi_model():
    return AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path='best.pt',
        confidence_threshold=0.15, # Lower threshold to catch faint debris
        device='cpu'
    )

model = load_sahi_model()

# Sidebar for Density Tuning
st.sidebar.header("Density Visualization Settings")
intensity = st.sidebar.slider("Heatmap Intensity", 0.5, 5.0, 2.0)
blur_radius = st.sidebar.slider("Hotspot Spread (Blur)", 21, 201, 91, step=2)

uploaded_file = st.file_uploader("Upload Satellite Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    with st.spinner('Calculating Density Map...'):
        # 1. Run Sliced Inference for high-res images
        result = get_sliced_prediction(
            img_array,
            model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2
        )
        
        # 2. Build the Density Map (Heatmap)
        # We create a float32 map to allow for "stacking" heat
        heatmap = np.zeros(img_array.shape[:2], dtype=np.float32)
        
        for pred in result.object_prediction_list:
            bbox = pred.bbox.to_xyxy()
            x1, y1, x2, y2 = map(int, bbox)
            
            # Instead of a flat 1, we add "intensity" to the center
            # This makes the centers of clusters much redder
            heatmap[y1:y2, x1:x2] += intensity

        # 3. Smooth the "heat" into a glow
        heatmap = cv2.GaussianBlur(heatmap, (blur_radius, blur_radius), 0)
        
        # 4. Normalize to 0-255 for the ColorMap
        if heatmap.max() > 0:
            heatmap = np.clip((heatmap / heatmap.max() * 255), 0, 255).astype(np.uint8)
        else:
            heatmap = heatmap.astype(np.uint8)

        # 5. Apply "Jet" Colormap (Blue -> Green -> Red)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # 6. Overlay Heatmap on Original Satellite Image
        # 0.6 opacity for original, 0.4 for heatmap
        density_map = cv2.addWeighted(img_array, 0.6, heatmap_color, 0.4, 0)
        density_image = Image.fromarray(density_map)

    # 7. Comparison Display
    st.subheader("🔍 Debris Density Comparison")
    image_comparison(
        img1=image,
        img2=density_image,
        label1="Original Satellite",
        label2="Debris Density (Heatmap)",
        width=800
    )
