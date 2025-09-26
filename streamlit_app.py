import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import os

# Page config
st.set_page_config(
    page_title="Image Defect Detection",
    page_icon="🔍",
    layout="wide"
)

# Model loading function
@st.cache_resource
def load_model():
    """Load the defect detection model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build model architecture
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, 2)
    
    # Try to load custom weights, fallback to pretrained
    weights_path = "backend/maskrcnn_defect.pth"
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state, strict=True)
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    
    model.eval().to(device)
    return model, device

def masks_to_polygons(mask):
    """Convert mask to polygon coordinates"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) >= 3:
            polygon = contour.reshape(-1, 2).tolist()
            polygons.append(polygon)
    return polygons

def draw_instances(image, instances):
    """Draw bounding boxes and masks on image"""
    drawn = image.copy()
    for instance in instances:
        box = instance['box']
        score = instance['score']
        label = instance['label']
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label_text = f"{label}: {score:.2f}"
        cv2.putText(drawn, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return drawn

def predict_defects(model, device, image, confidence_threshold=0.5):
    """Run defect detection on image"""
    # Preprocess image
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(image).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model([img_tensor])[0]
    
    # Process results
    instances = []
    scores = outputs.get("scores", torch.empty(0)).cpu().numpy()
    boxes = outputs.get("boxes", torch.empty(0)).cpu().numpy()
    labels = outputs.get("labels", torch.empty(0)).cpu().numpy()
    masks = outputs.get("masks", None)
    
    # Filter by confidence
    keep = scores >= confidence_threshold
    
    for i in range(len(scores)):
        if scores[i] >= confidence_threshold:
            box = boxes[i].tolist()
            score = float(scores[i])
            label = f"defect_{int(labels[i])}"
            
            instance = {
                'box': box,
                'score': score,
                'label': label
            }
            instances.append(instance)
    
    return instances

# Main app
def main():
    st.title("🔍 Image Defect Detection")
    st.markdown("Upload an image to detect defects using AI-powered computer vision.")
    
    # Load model
    with st.spinner("Loading AI model..."):
        model, device = load_model()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image to detect defects"
    )
    
    # Confidence threshold
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Adjust sensitivity of defect detection"
    )
    
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Defect Detection Results")
            
            # Run prediction
            with st.spinner("Analyzing image for defects..."):
                instances = predict_defects(model, device, image, confidence)
            
            if instances:
                st.success(f"Found {len(instances)} defect(s)!")
                
                # Draw results
                drawn_image = draw_instances(np.array(image), instances)
                st.image(drawn_image, use_column_width=True)
                
                # Show details
                st.subheader("Detection Details")
                for i, instance in enumerate(instances):
                    st.write(f"**Defect {i+1}:**")
                    st.write(f"- Confidence: {instance['score']:.3f}")
                    st.write(f"- Label: {instance['label']}")
                    st.write(f"- Bounding Box: {instance['box']}")
                    st.write("---")
            else:
                st.info("No defects detected with current confidence threshold.")
                st.image(image, use_column_width=True)
    
    # Instructions
    st.markdown("---")
    st.markdown("### How to use:")
    st.markdown("1. Upload an image file (JPG, PNG)")
    st.markdown("2. Adjust the confidence threshold if needed")
    st.markdown("3. View the detection results")
    st.markdown("4. Green boxes indicate detected defects")

if __name__ == "__main__":
    main()
