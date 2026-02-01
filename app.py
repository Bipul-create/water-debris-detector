import streamlit as st
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from streamlit_image_comparison import image_comparison
import numpy as np
import cv2
from PIL import Image

# Page Setup
st.set_page_config(page_title="Debris Hotspot Mapper", layout="wide")
st.title("🌊 Local Aerial Debris Hotspot Mapper")
st.markdown("Identify debris concentrations in high-resolution coastal and drone imagery.")

# 1. Load Model with SAHI Wrapper
@st.cache_resource
def load_precision_model():
    return AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path='best.pt', # Ensure your model file is named best.pt
        confidence_threshold=0.25,
        device='cpu'
    )

model = load_precision_model()

# 2. Sidebar Controls
st.sidebar.header("Visualization Settings")
heat_intensity = st.sidebar.slider("Heat Intensity (Redness)", 1.0, 20.0, 8.0)
blur_radius = st.sidebar.slider("Hotspot Smoothness", 51, 301, 151, step=2)
slice_detail = st.sidebar.select_slider("Detection Detail", options=[640, 800], value=640)

uploaded_file = st.file_uploader("Upload Drone or Aerial Photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and convert image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    with st.spinner('Analyzing debris density...'):
        # 3. Run Sliced Inference
        # This processes the image in chunks to maintain high resolution
        result = get_sliced_prediction(
            img_array,
            model,
            slice_height=slice_detail,
            slice_width=slice_detail,
            overlap_height_ratio=0.15,
            overlap_width_ratio=0.15
        )
        
        # 4. Generate the Density Heatmap
        heatmap = np.zeros(img_array.shape[:2], dtype=np.float32)
        
        for pred in result.object_prediction_list:
            bbox = pred.bbox.to_xyxy()
            x1, y1, x2, y2 = map(int, bbox)
            # Fill the detected area with "heat"
            heatmap[y1:y2, x1:x2] += heat_intensity

        # 5. Apply Visual Smoothing and Color
        # Gaussian blur creates the "glow" effect around clusters
        heatmap = cv2.GaussianBlur(heatmap, (blur_radius, blur_radius), 0)
        
        # Normalize and apply the JET colormap (Blue -> Yellow -> Red)
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # 6. Create the Final Comparison Overlay
        # Blend original image (0.6) with heatmap (0.4)
        overlay = cv2.addWeighted(img_array, 0.6, heatmap_color, 0.4, 0)
        final_heatmap_img = Image.fromarray(overlay)

    # 7. Layout Display
    st.subheader("🔍 Analysis Comparison")
    
    # Use the interactive slider to show Before vs After
    image_comparison(
        img1=image,
        img2=final_heatmap_img,
        label1="Original Photo",
        label2="Debris Hotspots",
        width=1000,
        starting_position=50
    )
    
    # Simple Legend for the user
    st.markdown("""
    **Color Legend:**
    * 🔴 **Red:** High Concentration (Hotspots)
    * 🟡 **Yellow/Green:** Moderate Concentration
    * 🔵 **Blue:** Low Concentration / Clear Water
    """)
    
    st.success(f"Detection complete. Identified {len(result.object_prediction_list)} primary debris points.")
