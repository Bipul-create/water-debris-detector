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
    with st.spinner('Analyzing debris...'):
        results = model.predict(image, conf=conf_threshold)
        
        # Plot the results on the image
        res_plotted = results[0].plot()
        
        # Convert BGR (OpenCV) to RGB (Streamlit/PIL)
        res_image = Image.fromarray(res_plotted[:, :, ::-1])

    with col2:
        st.subheader("Detected Debris")
        st.image(res_image, use_container_width=True)

    # 7. Results Summary
    st.success("Analysis Complete!")
    boxes = results[0].boxes
    if len(boxes) > 0:
        st.write(f"Detected **{len(boxes)}** items of debris.")
    else:
        st.info("No debris detected with current confidence level.")
