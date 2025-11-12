import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Silkworm Disease Detection",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful dark theme styling
st.markdown("""
    <style>
    .main {
        background: #0f172a;
    }
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    div[data-testid="stToolbar"] {
        display: none;
    }
    header[data-testid="stHeader"] {
        display: none;
    }
    
    .info-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #475569;
        margin: 10px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        height: 100%;
    }
    .metric-card h2 {
        font-size: 2.8em;
        margin: 0;
        font-weight: bold;
        color: white;
    }
    .metric-card p {
        margin: 10px 0 0 0;
        font-size: 1em;
        opacity: 0.95;
        color: white;
    }
    .status-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9em;
    }
    .status-healthy {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    .status-disease {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    .healthy {
        color: #10b981;
        font-weight: bold;
    }
    .grasserie {
        color: #ef4444;
        font-weight: bold;
    }
    h1 {
        color: #f1f5f9;
        text-align: center;
        padding: 10px 0;
        margin-bottom: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    h2, h3 {
        color: #e2e8f0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        padding: 12px 35px;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    .stRadio > label, .stSlider > label {
        color: #e2e8f0 !important;
        font-weight: 600;
    }
    .subtitle {
        text-align: center;
        color: #94a3b8;
        padding-bottom: 20px;
        font-size: 1.1em;
    }
    .detection-item {
        padding: 15px;
        background: #0f172a;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #6366f1;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    div[data-testid="stMarkdownContainer"] p {
        color: #cbd5e1;
    }
    .header-icon {
        font-size: 3em;
        text-align: center;
        margin-bottom: 10px;
    }
    .stat-label {
        color: #94a3b8;
        font-size: 0.85em;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    .stat-value {
        color: #e2e8f0;
        font-size: 1.8em;
        font-weight: bold;
    }
    .image-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 15px;
        margin-top: 15px;
    }
    .image-card {
        background: #0f172a;
        border-radius: 10px;
        padding: 10px;
        border: 2px solid #334155;
        cursor: pointer;
        transition: all 0.3s;
    }
    .image-card:hover {
        border-color: #6366f1;
        transform: scale(1.02);
    }
    .image-card.selected {
        border-color: #6366f1;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.5);
    }
    </style>
""", unsafe_allow_html=True)

# MODEL PATH - Update this with your model location
MODEL_PATH = "best (1).pt"

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.model_loaded = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'annotated_image' not in st.session_state:
    st.session_state.annotated_image = None
if 'all_results' not in st.session_state:
    st.session_state.all_results = []
if 'selected_image_idx' not in st.session_state:
    st.session_state.selected_image_idx = 0

# Load model on startup
@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model, True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, False

def process_images(uploaded_files, model, confidence):
    """Process uploaded images and return results"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"🤖 Analyzing image {idx+1}/{len(uploaded_files)}...")
        
        # Reset file pointer
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Run inference
        inference_results = model(image, conf=confidence, verbose=False)
        annotated = inference_results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        results.append({
            'original': image,
            'annotated': Image.fromarray(annotated_rgb),
            'result': inference_results[0],
            'filename': uploaded_file.name
        })
        
        progress_bar.progress((idx + 1) / len(uploaded_files))
    
    status_text.success("✅ All images processed!")
    progress_bar.empty()
    
    return results

# Title and description
st.markdown("<h1>🛸 Silkworm Disease Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("""
    <div class='subtitle'>
        Advanced AI-Powered Detection System | YOLOv11 Neural Network
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<div class='header-icon'>⚙️</div>", unsafe_allow_html=True)
    st.markdown("### Control Panel")
    
    # Model status
    if os.path.exists(MODEL_PATH):
        if not st.session_state.model_loaded:
            with st.spinner("🔄 Loading AI Model..."):
                st.session_state.model, st.session_state.model_loaded = load_model(MODEL_PATH)
        
        if st.session_state.model_loaded:
            st.success("✅ Model Loaded Successfully")
            st.info(f"📁 Model: `{os.path.basename(MODEL_PATH)}`")
        else:
            st.error("❌ Failed to Load Model")
    else:
        st.error(f"❌ Model not found at: `{MODEL_PATH}`")
        st.warning("Please update MODEL_PATH in the code")
    
    st.markdown("---")
    
    # Confidence threshold
    st.markdown("### 🎯 Detection Settings")
    confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05, 
                          help="Minimum confidence score for detections")
    
    # Detection mode
    st.markdown("### 📷 Input Source")
    detection_mode = st.radio("Select Mode", ["📤 Image Upload", "📸 Webcam Capture"], 
                             label_visibility="collapsed")
    
    st.markdown("---")
    
    # Information panel
    st.markdown("### 📖 Disease Information")
    
    with st.expander("✅ Healthy Silkworms", expanded=False):
        st.write("""
        - Normal appearance
        - Active movement
        - Proper feeding behavior
        - Clear body texture
        """)
    
    with st.expander("⚠️ Grasserie Disease", expanded=False):
        st.write("""
        - Viral infection
        - Swollen body segments
        - Discoloration
        - Requires immediate isolation
        """)
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #64748b; font-size: 0.85em;'>
            <p>© 2024 Silkworm Detection System<br>Powered by YOLOv11</p>
        </div>
    """, unsafe_allow_html=True)

# Main content
if not st.session_state.model_loaded:
    st.error("⚠️ Please ensure the model file exists at the specified path and restart the app.")
else:
    # Statistics row at the top - only show if we have results
    if len(st.session_state.all_results) > 0:
        # Calculate aggregate statistics
        total_detected = 0
        total_healthy = 0
        total_grasserie = 0
        
        for result_data in st.session_state.all_results:
            boxes = result_data['result'].boxes
            class_names = result_data['result'].names
            total_detected += len(boxes)
            total_healthy += sum(1 for box in boxes if 'healthy' in class_names[int(box.cls[0])].lower())
            total_grasserie += sum(1 for box in boxes if 'grasserie' in class_names[int(box.cls[0])].lower())
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.markdown(f"""
                <div class='metric-card'>
                    <h2>{total_detected}</h2>
                    <p>Total Detected</p>
                </div>
            """, unsafe_allow_html=True)
        
        with stat_col2:
            st.markdown(f"""
                <div class='metric-card' style='background: linear-gradient(135deg, #10b981 0%, #059669 100%);'>
                    <h2>{total_healthy}</h2>
                    <p>Healthy Silkworms</p>
                </div>
            """, unsafe_allow_html=True)
        
        with stat_col3:
            st.markdown(f"""
                <div class='metric-card' style='background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);'>
                    <h2>{total_grasserie}</h2>
                    <p>Grasserie Detected</p>
                </div>
            """, unsafe_allow_html=True)
        
        with stat_col4:
            health_percentage = (total_healthy / total_detected * 100) if total_detected > 0 else 0
            color = "#10b981" if health_percentage >= 70 else "#f59e0b" if health_percentage >= 40 else "#ef4444"
            st.markdown(f"""
                <div class='metric-card' style='background: linear-gradient(135deg, {color} 0%, {color}dd 100%);'>
                    <h2>{health_percentage:.0f}%</h2>
                    <p>Health Rate</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Main detection area
    col1, col2 = st.columns([1.2, 1], gap="large")
    
    with col1:
        st.markdown("<div class='detection-card'>", unsafe_allow_html=True)
        
        if "Image Upload" in detection_mode:
            st.markdown("### 📤 Upload Images for Analysis")
            st.markdown("<p style='color: #94a3b8; font-size: 0.9em;'>Supported formats: JPG, JPEG, PNG | Multiple files allowed</p>", unsafe_allow_html=True)
            
            uploaded_files = st.file_uploader("Choose images", type=['jpg', 'jpeg', 'png'], 
                                             label_visibility="collapsed", 
                                             accept_multiple_files=True,
                                             key="file_uploader")
            
            if uploaded_files:
                st.markdown(f"<p style='color: #6366f1; font-weight: bold;'>📁 {len(uploaded_files)} image(s) uploaded</p>", unsafe_allow_html=True)
                
                # Show thumbnail grid
                cols = st.columns(min(4, len(uploaded_files)))
                for idx, uploaded_file in enumerate(uploaded_files):
                    with cols[idx % 4]:
                        image = Image.open(uploaded_file)
                        st.image(image, use_column_width=True, caption=f"Image {idx+1}")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                if st.button("🔍 Start Batch AI Detection", key="detect_btn"):
                    try:
                        # Process images and store in session state
                        st.session_state.all_results = process_images(
                            uploaded_files, 
                            st.session_state.model, 
                            confidence
                        )
                        st.session_state.selected_image_idx = 0
                        
                    except Exception as e:
                        st.error(f"❌ Error during detection: {str(e)}")
                        st.exception(e)
                
                # Show processed images if available
                if len(st.session_state.all_results) > 0:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("### 🖼️ Select Image to View Details")
                    
                    result_cols = st.columns(min(4, len(st.session_state.all_results)))
                    for idx, result_data in enumerate(st.session_state.all_results):
                        with result_cols[idx % 4]:
                            if st.button(f"Image {idx+1}", key=f"img_btn_{idx}"):
                                st.session_state.selected_image_idx = idx
            else:
                # Clear results when no files are uploaded
                st.session_state.all_results = []
                st.markdown("""
                    <div style='text-align: center; padding: 60px 20px; color: #64748b;'>
                        <div style='font-size: 4em; margin-bottom: 20px;'>📁</div>
                        <h3 style='color: #94a3b8;'>No Images Selected</h3>
                        <p>Upload one or more images to begin batch detection</p>
                    </div>
                """, unsafe_allow_html=True)
        
        else:  # Webcam mode
            st.markdown("### 📸 Webcam Capture")
            st.markdown("<p style='color: #94a3b8; font-size: 0.9em;'>Click the camera button to capture an image</p>", unsafe_allow_html=True)
            
            camera_input = st.camera_input("Take a picture", label_visibility="collapsed")
            
            if camera_input is not None:
                try:
                    image = Image.open(camera_input)
                    
                    # Convert to RGB if necessary
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    with st.spinner("🤖 AI is analyzing the silkworms..."):
                        results = st.session_state.model(image, conf=confidence, verbose=False)
                        annotated = results[0].plot()
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        
                        st.session_state.all_results = [{
                            'original': image,
                            'annotated': Image.fromarray(annotated_rgb),
                            'result': results[0],
                            'filename': 'webcam_capture.jpg'
                        }]
                        st.session_state.selected_image_idx = 0
                        
                except Exception as e:
                    st.error(f"❌ Error during detection: {str(e)}")
                    st.exception(e)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='detection-card'>", unsafe_allow_html=True)
        
        if len(st.session_state.all_results) > 0:
            selected_idx = st.session_state.selected_image_idx
            if selected_idx < len(st.session_state.all_results):
                result_data = st.session_state.all_results[selected_idx]
                
                st.markdown(f"### 🎯 Detection Results - Image {selected_idx + 1}")
                st.markdown(f"<p style='color: #94a3b8; font-size: 0.9em;'>📄 {result_data['filename']}</p>", unsafe_allow_html=True)
                
                st.image(result_data['annotated'], caption="🔍 AI Analysis Results", use_column_width=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Detailed detections list
                st.markdown("### 📋 Detailed Analysis")
                
                boxes = result_data['result'].boxes
                class_names = result_data['result'].names
                
                if len(boxes) > 0:
                    for i, box in enumerate(boxes):
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = class_names[cls]
                        
                        is_healthy = 'healthy' in class_name.lower()
                        icon = "✅" if is_healthy else "⚠️"
                        color = "#10b981" if is_healthy else "#ef4444"
                        
                        st.markdown(f"""
                            <div class='detection-item' style='border-left-color: {color};'>
                                <div>
                                    <span style='font-size: 1.2em;'>{icon}</span>
                                    <span style='color: {color}; font-weight: bold; margin-left: 10px;'>{class_name}</span>
                                </div>
                                <div>
                                    <span style='background: {color}33; color: {color}; padding: 5px 12px; border-radius: 15px; font-weight: bold;'>
                                        {conf:.1%}
                                    </span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No silkworms detected in this image. Try adjusting the confidence threshold.")
        else:
            st.markdown("""
                <div style='text-align: center; padding: 80px 20px; color: #64748b;'>
                    <div style='font-size: 4em; margin-bottom: 20px;'>🔍</div>
                    <h3 style='color: #94a3b8;'>Awaiting Detection</h3>
                    <p>Results will appear here after analysis</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

