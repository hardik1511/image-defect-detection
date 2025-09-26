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
    page_title="Image Defect Detection",
    page_icon="🔍",
    layout="wide"
)

# Simple model loading
@st.cache_resource
def load_model():
    """Load a simple model for defect detection"""
    try:
        # Try to load custom model first
        if os.path.exists("maskrcnn_defect.pth"):
            device = torch.device("cpu")  # Use CPU for Hugging Face
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
            state = torch.load("maskrcnn_defect.pth", map_location="cpu")
            model.load_state_dict(state, strict=False)
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
    st.title("🔍 Image Defect Detection")
    st.markdown("Upload an image to detect defects using AI")
    
    # Load model
    with st.spinner("Loading AI model..."):
        model, device = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check the model file.")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image to detect defects"
    )
    
    # Confidence slider
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Detection Results")
            
            # Run detection
            with st.spinner("Analyzing image..."):
                boxes, scores, labels = detect_defects(image, model, device, confidence)
            
            if len(boxes) > 0:
                st.success(f"Found {len(boxes)} defect(s)!")
                
                # Draw results
                result_image = draw_boxes(image, boxes, scores, labels)
                st.image(result_image, use_column_width=True)
                
                # Show details
                for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                    st.write(f"**Defect {i+1}:** Confidence: {score:.3f}")
            else:
                st.info("No defects detected")
                st.image(image, use_column_width=True)

if __name__ == "__main__":
    main()
