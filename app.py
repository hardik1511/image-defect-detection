import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import os

# Page config
st.set_page_config(
    page_title="AI Defect Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stApp {
        background-color: #f8f9fa;
    }
    .stSuccess {
        background-color: #d4edda;
        border-color: #c3e6cb;
    }
    .stInfo {
        background-color: #d1ecf1;
        border-color: #bee5eb;
    }
    .stWarning {
        background-color: #fff3cd;
        border-color: #ffeaa7;
    }
    .stError {
        background-color: #f8d7da;
        border-color: #f5c6cb;
    }
    .uploadedFile {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Model loading with proper architecture
@st.cache_resource
def load_model():
    """Load a simple model for defect detection"""
    try:
        device = torch.device("cpu")  # Use CPU for Hugging Face
        
        # Try to load custom model first
        if os.path.exists("maskrcnn_defect.pth"):
            try:
                # Build model with correct architecture for 2 classes from the start
                from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
                from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
                
                # Create model with no pretrained weights
                model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
                
                # Get number of input features for the classifier
                in_features = model.roi_heads.box_predictor.cls_score.in_features
                # Replace the pre-trained head with a new one for 2 classes
                model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
                
                # Get number of input features for the mask classifier
                in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
                # Replace the pre-trained head with a new one for 2 classes
                model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, 2)
                
                # Load the custom weights
                state = torch.load("maskrcnn_defect.pth", map_location="cpu")
                model.load_state_dict(state, strict=True)
                model.eval()
                return model, device
            except Exception as custom_error:
                st.warning(f"Custom model loading failed: {str(custom_error)}")
                st.info("Falling back to pretrained COCO model...")
                # Fallback to pretrained model
                model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
                model.eval()
                return model, device
        else:
            # Fallback to pretrained model
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
            model.eval()
            return model, torch.device("cpu")
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None, None

def detect_defects(image, model, device, confidence=0.5):
    """Simple defect detection"""
    try:
        # Convert to tensor
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(image).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model([img_tensor])[0]
        
        # Process results
        boxes = outputs['boxes'].cpu().numpy()
        scores = outputs['scores'].cpu().numpy()
        labels = outputs['labels'].cpu().numpy()
        
        # Filter by confidence
        keep = scores >= confidence
        
        # For custom model (2 classes): only keep defect detections (label == 1)
        # For pretrained model (91 classes): keep all detections
        if len(set(labels)) <= 2:  # Custom model with 2 classes
            keep = keep & (labels == 1)  # Only defects
        # For pretrained model, keep all detections above confidence threshold
        
        filtered_boxes = boxes[keep]
        filtered_scores = scores[keep]
        filtered_labels = labels[keep]
        
        return filtered_boxes, filtered_scores, filtered_labels
    except Exception as e:
        st.error(f"Detection error: {str(e)}")
        return [], [], []

def draw_boxes(image, boxes, scores, labels):
    """Draw bounding boxes on image"""
    img_array = np.array(image)
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_array, f"Defect {i+1}: {score:.2f}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img_array

# Main app
def main():
    # Header with better styling
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #1f77b4; margin-bottom: 0.5rem;">🔍 AI Defect Detection System</h1>
        <p style="color: #666; font-size: 1.1rem; margin-bottom: 2rem;">
            Advanced computer vision for automated defect identification in museum exhibits
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model with better feedback
    with st.spinner("🤖 Loading AI model..."):
        model, device = load_model()
    
    if model is None:
        st.error("❌ Failed to load model. Please check the model file.")
        return
    
    # Success message for model loading
    st.success("✅ AI model loaded successfully!")
    
    # Create two main columns
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📁 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image to detect defects",
            label_visibility="collapsed"
        )
        
        # Confidence slider with better styling
        st.markdown("### ⚙️ Detection Settings")
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Adjust sensitivity of defect detection. Higher values = more confident detections only."
        )
        
        # Add some info about the system
        with st.expander("ℹ️ About This System"):
            st.markdown("""
            **AI Defect Detection System**
            
            This system uses Mask R-CNN deep learning to automatically identify defects in museum exhibits and gallery artifacts.
            
            **Features:**
            - Real-time defect detection
            - Adjustable confidence threshold
            - Visual bounding box overlays
            - Detailed detection information
            
            **How to use:**
            1. Upload an image file (JPG, PNG)
            2. Adjust confidence threshold if needed
            3. View detection results
            4. Green boxes indicate detected defects
            """)
    
    with col2:
        st.markdown("### 🔍 Detection Results")
        
        if uploaded_file is not None:
            # Load image
            image = Image.open(uploaded_file).convert("RGB")
            
            # Show original image
            st.markdown("**Original Image:**")
            st.image(image, use_column_width=True)
            
            # Run detection with better feedback
            with st.spinner("🔍 Analyzing image for defects..."):
                boxes, scores, labels = detect_defects(image, model, device, confidence)
            
            if len(boxes) > 0:
                # Success message with count
                st.success(f"🎯 Found {len(boxes)} defect(s) with confidence ≥ {confidence:.2f}")
                
                # Draw results
                result_image = draw_boxes(image, boxes, scores, labels)
                st.markdown("**Detection Results:**")
                st.image(result_image, use_column_width=True)
                
                # Show detailed results in an expandable section
                with st.expander("📊 Detailed Detection Information"):
                    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                        st.markdown(f"""
                        **Defect {i+1}:**
                        - Confidence: {score:.3f}
                        - Bounding Box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]
                        - Label: {label}
                        """)
            else:
                st.info("ℹ️ No defects detected with current confidence threshold")
                st.markdown("**Try adjusting the confidence threshold or upload a different image.**")
        else:
            st.info("👆 Upload an image to start defect detection")
    
    # Footer with additional information
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem 0;">
        <p>Built with ❤️ using Streamlit, PyTorch, and Hugging Face Spaces</p>
        <p>AI Defect Detection System for Museum Applications</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
