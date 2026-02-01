import streamlit as st
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from streamlit_image_comparison import image_comparison
import numpy as np
import cv2
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="Marine Debris Mapper", layout="wide")
st.title("🛰️ High-Res Satellite Debris Hotspot Mapper")

# 2. Load Model using SAHI Wrapper
@st.cache_resource
def load_sahi_model():
    # This wraps YOLOv8 to allow it to "slice" large images
    return AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path='best.pt', # Ensure your model file is named best.pt
        confidence_threshold=0.2,
        device='cpu' # Streamlit Cloud uses CPU
    )

model = load_sahi_model()

# 3. Sidebar controls
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Detection Sensitivity", 0.05, 1.0, 0.2)
blur_val = st.sidebar.slider("Hotspot Smoothness", 21, 151, 71, step=2)

# 4. Image Upload
uploaded_file = st.file_uploader("Upload Satellite Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Process Image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    with st.spinner('Analyzing high-resolution slices... This may take a minute.'):
        # SAHI Sliced Inference
        result = get_sliced_prediction(
            img_array,
            model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )
        
        # 5. Generate Heatmap from detections
        heatmap = np.zeros(img_array.shape[:2], dtype=np.float32)
        for object_prediction in result.object_prediction_list:
            bbox = object_prediction.bbox.to_xyxy()
            x1, y1, x2, y2 = map(int, bbox)
            # Add "heat" to the detected debris area
            heatmap[y1:y2, x1:x2] += 1

        # Normalize and Color the Heatmap
        heatmap = cv2.GaussianBlur(heatmap, (blur_val, blur_val), 0)
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Combine Heatmap with Original Image
        hotspot_map = cv2.addWeighted(img_array, 0.6, heatmap_color, 0.4, 0)
        hotspot_image = Image.fromarray(hotspot_map)

    # 6. Display Comparison Slider
    st.subheader("🔍 Compare: Original vs Hotspots")
    st.write("Drag the handle to see detected debris zones.")
    
    # Center the comparison tool
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        image_comparison(
            img1=image,
            img2=hotspot_image,
            label1="Original View",
            label2="Debris Hotspots",
            width=800,
            starting_position=50,
            show_labels=True
        )

    # 7. Summary Stats
    st.success(f"Analysis Complete! Found {len(result.object_prediction_list)} potential debris items.")
