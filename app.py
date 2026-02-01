import streamlit as st
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from streamlit_image_comparison import image_comparison
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Aerial Debris Area Mapper", layout="wide")
st.title("🚁 Local Aerial Debris: Density & Area Analysis")

# 1. Load Model with SAHI Wrapper
@st.cache_resource
def load_precision_model():
    return AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path='best.pt',
        confidence_threshold=0.25,
        device='cpu'
    )

model = load_precision_model()

# 2. Sidebar Settings
st.sidebar.header("📏 Scale & Precision")
# GSD: How many meters does one pixel represent? (Common drone GSD is 0.02 - 0.05m)
gsd = st.sidebar.number_input("Resolution (meters per pixel)", value=0.03, format="%.4f")
heat_val = st.sidebar.slider("Heat Intensity", 1.0, 15.0, 5.0)
blur_val = st.sidebar.slider("Density Spread", 51, 301, 121, step=2)

uploaded_file = st.file_uploader("Upload Drone/Aerial Photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    with st.spinner('Calculating debris footprint...'):
        # 3. Sliced Inference for Local Heights (640px slices)
        result = get_sliced_prediction(
            img_array,
            model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.15
        )
        
        # 4. Math: Area Calculation
        # Total Real Area = (Sum of Pixel Areas) * GSD^2
        total_pixel_area = 0
        heatmap = np.zeros(img_array.shape[:2], dtype=np.float32)
        
        for pred in result.object_prediction_list:
            bbox = pred.bbox.to_xyxy()
            x1, y1, x2, y2 = map(int, bbox)
            
            # Calculate pixel area of this specific item
            w_px = x2 - x1
            h_px = y2 - y1
            total_pixel_area += (w_px * h_px)
            
            # Add to heatmap
            heatmap[y1:y2, x1:x2] += heat_val

        # Convert to real-world Area (m^2)
        real_area_m2 = total_pixel_area * (gsd ** 2)

        # 5. Process Heatmap Visuals
        heatmap = cv2.GaussianBlur(heatmap, (blur_val, blur_val), 0)
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        overlay = cv2.addWeighted(img_array, 0.5, heatmap_color, 0.5, 0)
        final_img = Image.fromarray(overlay)

    # 6. Display Results
    col1, col2 = st.columns(2)
    col1.metric("Approx. Debris Area", f"{real_area_m2:.2f} m²")
    col2.metric("Detected Clusters", len(result.object_prediction_list))

    image_comparison(
        img1=image,
        img2=final_img,
        label1="Raw Photo",
        label2="Density & Footprint Map",
        width=1000
    )
