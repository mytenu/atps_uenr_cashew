import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# Disease classes
DISEASE_CLASSES = {
    0: "Anthracnose",
    1: "Gummosis",
    2: "Healthy",
    3: "Leaf Miner",
    4: "Not Cashew",
    5: "Parasitic Plant",
    6: "Physiological Disorder",
    7: "Powdery Mildew",
    8: "Red Rust",
    9: "Termite"
}

# Define the pretrained model class (same as your training script)
class PretrainedModel(nn.Module):
    def __init__(self, num_classes, model_name='mobilenet_v3_small', pretrained=False):
        super(PretrainedModel, self).__init__()
        
        if model_name == 'mobilenet_v3_small':
            self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
            self.backbone.classifier[3] = nn.Linear(self.backbone.classifier[3].in_features, num_classes)
        else:
            raise ValueError(f"Model {model_name} not supported.")
    
    def forward(self, x):
        return self.backbone(x)

# Load model with caching
@st.cache_resource
def load_model():
    """Load the trained model"""
    device = torch.device('cpu')  # Use CPU for Streamlit Cloud
    model = PretrainedModel(num_classes=len(DISEASE_CLASSES), model_name='mobilenet_v3_small', pretrained=False)
    
    # Load the trained weights
    model.load_state_dict(torch.load('mobilenet_v3_small_epoch_5.pth', map_location=device))
    model.to(device)
    model.eval()
    return model, device

# Image preprocessing
def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Transform and add batch dimension
    img_tensor = transform(image).unsqueeze(0)
    return img_tensor

# Prediction function
def predict(image, model, device):
    """Make prediction on the input image"""
    img_tensor = preprocess_image(image)
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    
    return predicted_class.item(), confidence.item(), probabilities[0]

# Streamlit App
def main():
    # Page configuration
    st.set_page_config(
        page_title="Cashew Disease Classifier",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #2E7D32;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .prediction-box {
            padding: 20px;
            border-radius: 10px;
            background-color: #f0f2f6;
            margin: 10px 0;
        }
        .healthy {
            background-color: #d4edda;
            color: #155724;
        }
        .disease {
            background-color: #f8d7da;
            color: #721c24;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🌿 Cashew Disease Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image of a cashew plant to detect diseases</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info(
            """
            This app uses a fine-tuned MobileNet V3 Small model to classify 
            cashew plant diseases from images.
            
            **Supported Classes:**
            - Anthracnose
            - Gummosis
            - Healthy
            - Leaf Miner
            - Not Cashew
            - Parasitic Plant
            - Physiological Disorder
            - Powdery Mildew
            - Red Rust
            - Termite
            """
        )
        
        st.header("Instructions")
        st.markdown("""
        1. Upload a clear image of a cashew plant
        2. Wait for the model to process
        3. View the prediction and confidence scores
        4. Check the probability distribution
        """)
        
        st.header("Model Info")
        st.markdown("""
        - **Model**: MobileNet V3 Small
        - **Parameters**: ~1.5M
        - **Input Size**: 224x224
        - **Classes**: 10
        """)
    
    # Load model
    try:
        model, device = load_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of the cashew plant"
    )
    
    # Layout
    col1, col2 = st.columns([1, 1])
    
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file)
        
        with col1:
            st.subheader("📸 Uploaded Image")
            st.image(image, use_container_width=True)
            st.caption(f"Image size: {image.size[0]}x{image.size[1]} pixels")
        
        with col2:
            st.subheader("🔍 Analysis Results")
            
            # Make prediction
            with st.spinner("Analyzing image..."):
                try:
                    predicted_class, confidence, probabilities = predict(image, model, device)
                    disease_name = DISEASE_CLASSES[predicted_class]
                    
                    # Display prediction
                    if disease_name == "Healthy":
                        st.markdown(f'<div class="prediction-box healthy">', unsafe_allow_html=True)
                        st.markdown(f"### ✅ Prediction: **{disease_name}**")
                    elif disease_name == "Not Cashew":
                        st.markdown(f'<div class="prediction-box" style="background-color: #fff3cd; color: #856404;">', unsafe_allow_html=True)
                        st.markdown(f"### ⚠️ Prediction: **{disease_name}**")
                    else:
                        st.markdown(f'<div class="prediction-box disease">', unsafe_allow_html=True)
                        st.markdown(f"### 🦠 Prediction: **{disease_name}**")
                    
                    st.markdown(f"**Confidence:** {confidence*100:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Confidence meter
                    st.progress(confidence)
                    
                    # Interpretation
                    st.markdown("---")
                    st.subheader("💡 Interpretation")
                    if confidence > 0.8:
                        st.success("High confidence - The model is quite certain about this prediction.")
                    elif confidence > 0.6:
                        st.warning("Moderate confidence - Consider taking additional images for verification.")
                    else:
                        st.error("Low confidence - The model is uncertain. Please upload a clearer image.")
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.stop()
        
        # Probability distribution
        st.markdown("---")
        st.subheader("📊 Probability Distribution")
        
        # Create probability dataframe
        prob_dict = {DISEASE_CLASSES[i]: float(probabilities[i]) * 100 for i in range(len(DISEASE_CLASSES))}
        sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
        
        # Display as bar chart
        st.bar_chart(sorted_probs)
        
        # Display as table
        with st.expander("View Detailed Probabilities"):
            for disease, prob in sorted_probs.items():
                st.write(f"**{disease}:** {prob:.2f}%")
    
    else:
        # Placeholder when no image is uploaded
        st.info("👆 Please upload an image to get started")
        
        # Sample images section
        st.markdown("---")
        st.subheader("📚 Sample Use Cases")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**✅ Healthy Plants**")
            st.caption("Monitor plant health regularly")
        
        with col2:
            st.markdown("**🦠 Disease Detection**")
            st.caption("Early detection of diseases")
        
        with col3:
            st.markdown("**📈 Track Progress**")
            st.caption("Monitor treatment effectiveness")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>Developed with ❤️ for Cashew Farmers | Powered by MobileNet V3 Small</p>
            <p><em>For educational and research purposes</em></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
