import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Marine Debris Detector", layout="wide")

# 2. Title and Description
st.title("🌊 Ocean Plastic & Debris Detector")
st.write("Upload an underwater image to identify plastic, glass, and other pollutants.")

# 3. Load the Model
@st.cache_resource
def load_model():
    # This matches the 'best.pt' file in your folder
    model = YOLO('best.pt')
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Make sure 'best.pt' is in the same folder.")

# 4. Sidebar Settings
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4)

# 5. File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to a format PIL/YOLO understands
    image = Image.open(uploaded_file)
    
    # Create two columns for Before/After
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    # 6. Run Inference
with st.spinner('Generating Hotspot Map...'):
    results = model.predict(image, conf=conf_threshold)
    
    # 1. Start with the original image
    img_array = np.array(image)
    heatmap = np.zeros(img_array.shape[:2], dtype=np.float32)

    # 2. Add "Heat" for every detection
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            # Add intensity to the center of the box
            heatmap[y1:y2, x1:x2] += 1 

    # 3. Blur the heatmap to make it look like a "hotspot"
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    heatmap = np.clip(heatmap / heatmap.max() * 255, 0, 255).astype(np.uint8) if heatmap.max() > 0 else heatmap.astype(np.uint8)
    
    # 4. Apply a color map (Red = Hot, Blue = Cold)
    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 5. Overlay the heatmap onto the original image
    overlay = cv2.addWeighted(img_array, 0.6, heatmap_img, 0.4, 0)
    st.image(overlay, caption="Debris Hotspot Map", use_container_width=True)


    # 7. Results Summary
    st.success("Analysis Complete!")
    boxes = results[0].boxes
    if len(boxes) > 0:
        st.write(f"Detected **{len(boxes)}** items of debris.")
    else:
        st.info("No debris detected with current confidence level.")
