import streamlit as st
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from streamlit_image_comparison import image_comparison
import numpy as np
import cv2
from PIL import Image

# Set Page Config
st.set_page_config(page_title="Satellite Debris Precision Mapper", layout="wide")
st.title("🛰️ Precision Satellite Debris Hotspot Mapper")

# Load Model with SAHI Wrapper
@st.cache_resource
def load_precision_model():
    return AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path='best.pt',
        confidence_threshold=0.15, # Low threshold for tiny satellite objects
        device='cpu'
    )

model = load_precision_model()

# Sidebar Settings
st.sidebar.header("Precision Settings")
slice_size = st.sidebar.select_slider("Resolution Level (Slice Size)", options=[416, 640], value=416)
overlap = st.sidebar.slider("Slice Overlap %", 0.1, 0.5, 0.2)
heat_blur = st.sidebar.slider("Heatmap Spread", 11, 101, 31, step=2)

uploaded_file = st.file_uploader("Upload Satellite Imagery", type=["jpg", "jpeg", "png", "tif"])

if uploaded_file:
    # 1. Load Image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    with st.spinner('Running high-precision sliced inference...'):
        # 2. Sliced Inference (SAHI)
        # This prevents 'squashing' large satellite photos
        result = get_sliced_prediction(
            img_array,
            model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            postprocess_type="NMS" # Removes duplicate hits in overlaps
        )
        
        # 3. Create Point-Based Heatmap
        # We use a float layer for high-precision density stacking
        heatmap = np.zeros(img_array.shape[:2], dtype=np.float32)
        
        for pred in result.object_prediction_list:
            bbox = pred.bbox.to_xyxy()
            # Calculate exact center point of detection
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            
            # Draw a 'hot' point. Clusters will overlap and turn red.
            cv2.circle(heatmap, (center_x, center_y), radius=10, color=255, thickness=-1)

        # 4. Blur and Colorize
        heatmap = cv2.GaussianBlur(heatmap, (heat_blur, heat_blur), 0)
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # 5. Blend Heatmap with Satellite Image
        # 0.7 original, 0.3 heatmap overlay
        final_overlay = cv2.addWeighted(img_array, 0.7, heatmap_color, 0.3, 0)
        final_image = Image.fromarray(final_overlay)

    # 6. Interactive Comparison
    st.subheader("🔍 Localized Hotspot Analysis")
    st.info(f"Analysis Complete: Processing in {slice_size}px slices.")
    
    image_comparison(
        img1=image,
        img2=final_image,
        label1="Original Satellite",
        label2="Debris Hotspots",
        width=900,
        starting_position=50
    )

    # 7. Summary
    st.write(f"**Detected Clusters:** {len(result.object_prediction_list)} points identified.")
