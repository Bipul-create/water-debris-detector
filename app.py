import streamlit as st
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from streamlit_image_comparison import image_comparison
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Coastal Debris Mapper", layout="wide")
st.title("🌊 Localized Marine Debris Density Mapper")
st.markdown("Designed for **Aerial & Coastal** high-res imagery.")

@st.cache_resource
def load_aerial_model():
    return AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path='best.pt',
        confidence_threshold=0.30, # Higher for local views to reduce false positives
        device='cpu'
    )

model = load_aerial_model()

# Sidebar Settings for Local Heights
st.sidebar.header("Altitude Tuning")
# Slices are larger (640) because debris is bigger at lower altitudes
slice_size = st.sidebar.select_slider("Detection Detail", options=[640, 800], value=640)
heat_intensity = st.sidebar.slider("Heat Intensity", 1.0, 15.0, 8.0)
blur_radius = st.sidebar.slider("Density Spread", 51, 301, 151, step=2)

uploaded_file = st.file_uploader("Upload Drone or Local Aerial Photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    with st.spinner('Analyzing local debris zones...'):
        # 1. Sliced Inference optimized for aerial heights
        result = get_sliced_prediction(
            img_array,
            model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=0.15,
            overlap_width_ratio=0.15
        )
        
        # 2. Advanced Area-Based Heatmap
        heatmap = np.zeros(img_array.shape[:2], dtype=np.float32)
        
        for pred in result.object_prediction_list:
            bbox = pred.bbox.to_xyxy()
            x1, y1, x2, y2 = map(int, bbox)
            # Fills the actual footprint of the debris with 'heat'
            heatmap[y1:y2, x1:x2] += heat_intensity

        # 3. Apply Professional Thermal Effect
        heatmap = cv2.GaussianBlur(heatmap, (blur_radius, blur_radius), 0)
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        
        # COLORMAP_JET gives the Red (High) to Blue (Low) transition
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # 4. Create the Final Overlay
        overlay = cv2.addWeighted(img_array, 0.5, heatmap_color, 0.5, 0)
        final_heatmap_img = Image.fromarray(overlay)

    # 5. Centered Comparison Slider
    st.subheader("🔍 Analysis Output")
    image_comparison(
        img1=image,
        img2=final_heatmap_img,
        label1="Original Image",
        label2="Debris Density Map",
        width=1000,
        starting_position=50
    )
    
    st.info(f"Mapped {len(result.object_prediction_list)} significant debris concentrations.")
