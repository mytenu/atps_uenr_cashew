import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import base64

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

# Non-chemical treatment recommendations
TREATMENT_INFO = {
    "Anthracnose": {
        "description": "A fungal disease causing dark lesions on leaves, stems, and fruits.",
        "symptoms": "Dark brown to black spots with yellow halos on leaves and fruits, premature leaf drop",
        "treatments": [
            "🌱 **Pruning**: Remove and destroy infected plant parts immediately to prevent spread",
            "💧 **Water Management**: Avoid overhead irrigation; water at the base of plants in the morning",
            "🍂 **Sanitation**: Remove fallen leaves and debris from around plants regularly",
            "🌿 **Plant Spacing**: Ensure adequate spacing between plants for good air circulation",
            "🌾 **Mulching**: Apply organic mulch to prevent soil splash onto lower leaves",
            "🔄 **Crop Rotation**: Rotate with non-host crops if possible",
            "💪 **Plant Health**: Maintain plant vigor through proper nutrition (compost tea, organic matter)"
        ],
        "prevention": "Plant resistant varieties, ensure good drainage, and maintain plant hygiene"
    },
    "Gummosis": {
        "description": "A physiological disorder causing gum exudation from stems and branches.",
        "symptoms": "Sticky gum oozing from bark, bark cracking, branch dieback",
        "treatments": [
            "✂️ **Careful Pruning**: Remove affected branches, cutting back to healthy wood",
            "🛡️ **Wound Protection**: Apply natural sealants like clay or beeswax to pruning cuts",
            "💧 **Drainage Improvement**: Ensure proper soil drainage to prevent waterlogging",
            "🌱 **Avoid Injuries**: Minimize mechanical damage to bark and stems",
            "🍃 **Reduce Stress**: Ensure adequate but not excessive watering",
            "🌿 **Organic Matter**: Add compost to improve soil structure and drainage",
            "☀️ **Sunlight Management**: Protect young trees from excessive sun exposure"
        ],
        "prevention": "Select well-drained sites, avoid over-irrigation, and handle plants carefully during cultivation"
    },
    "Healthy": {
        "description": "Your cashew plant appears healthy!",
        "symptoms": "Vibrant green leaves, no visible disease symptoms",
        "treatments": [
            "✅ **Maintain Current Practices**: Continue your current care routine",
            "🌿 **Regular Monitoring**: Inspect plants weekly for early disease detection",
            "💧 **Consistent Watering**: Maintain regular watering schedule (avoid waterlogging)",
            "🍂 **Cleanliness**: Keep area around plants free of debris",
            "✂️ **Preventive Pruning**: Remove dead or damaged branches promptly",
            "🌱 **Soil Health**: Apply compost or organic matter annually",
            "🐝 **Biodiversity**: Encourage beneficial insects for natural pest control"
        ],
        "prevention": "Continue good agricultural practices and regular monitoring"
    },
    "Leaf Miner": {
        "description": "Insect larvae that create winding tunnels within leaves.",
        "symptoms": "Serpentine trails or blotches on leaves, reduced photosynthesis",
        "treatments": [
            "🍃 **Remove Affected Leaves**: Pick and destroy leaves with visible mines",
            "🦋 **Natural Predators**: Encourage parasitic wasps and predatory beetles",
            "🌼 **Companion Planting**: Plant marigolds, basil, or nasturtiums nearby to repel pests",
            "🌿 **Neem Solution**: Spray diluted neem oil (organic pesticide) on affected areas",
            "🪤 **Yellow Sticky Traps**: Use to monitor and trap adult flies",
            "✂️ **Pruning**: Remove heavily infested branches to reduce population",
            "💪 **Plant Vigor**: Maintain healthy plants through proper fertilization (compost, manure)"
        ],
        "prevention": "Use reflective mulches, maintain plant health, and encourage natural enemies"
    },
    "Not Cashew": {
        "description": "The uploaded image does not appear to be a cashew plant.",
        "symptoms": "N/A - This is not a cashew plant",
        "treatments": [
            "📸 **Upload Correct Image**: Please upload a clear image of a cashew plant",
            "🌿 **Cashew Features**: Look for characteristic cashew leaves (oval, leathery)",
            "🔍 **Check Plant**: Ensure you're examining the correct plant species",
            "📚 **Plant Identification**: Consult local agricultural extension if unsure"
        ],
        "prevention": "Ensure proper plant identification before diagnosis"
    },
    "Parasitic Plant": {
        "description": "Plants like mistletoe that attach to and derive nutrients from cashew trees.",
        "symptoms": "Visible parasitic plant growth on branches, reduced tree vigor, yellowing leaves",
        "treatments": [
            "✂️ **Manual Removal**: Cut parasitic plants as close to the host branch as possible",
            "🪓 **Prune Infected Branches**: Remove heavily infested branches entirely",
            "🔍 **Regular Inspection**: Check trees frequently and remove parasites early",
            "🌳 **Tree Vigor**: Improve host tree health through organic fertilization",
            "🛡️ **Wound Treatment**: Cover cuts with natural sealants to prevent reinfection",
            "🍂 **Sanitation**: Destroy all removed parasitic plant material immediately",
            "💧 **Proper Irrigation**: Maintain optimal water levels to keep trees healthy"
        ],
        "prevention": "Regular monitoring, maintain tree vigor, and remove parasites promptly when spotted"
    },
    "Physiological Disorder": {
        "description": "Non-infectious problems caused by environmental stress or nutrient imbalances.",
        "symptoms": "Leaf discoloration, stunted growth, abnormal leaf shapes, tip burn",
        "treatments": [
            "🧪 **Soil Testing**: Conduct soil test to identify nutrient deficiencies or toxicities",
            "🌿 **Organic Amendments**: Add compost, well-rotted manure, or organic fertilizers",
            "💧 **Water Management**: Adjust irrigation - avoid both drought stress and waterlogging",
            "🌤️ **Shade Provision**: Provide temporary shade during extreme heat",
            "🍃 **Mulching**: Apply organic mulch to regulate soil temperature and moisture",
            "⚖️ **pH Adjustment**: Use lime (for acidic soil) or sulfur (for alkaline soil) if needed",
            "🌱 **Foliar Spray**: Apply compost tea or seaweed extract for quick nutrient boost"
        ],
        "prevention": "Maintain optimal growing conditions, regular soil testing, and proper nutrition"
    },
    "Powdery Mildew": {
        "description": "Fungal disease appearing as white powdery coating on leaves.",
        "symptoms": "White or gray powdery patches on leaves, leaf distortion, premature leaf drop",
        "treatments": [
            "🥛 **Milk Spray**: Mix 1 part milk with 9 parts water, spray weekly on affected areas",
            "🧂 **Baking Soda Solution**: Mix 1 tbsp baking soda + 1 tbsp vegetable oil in 1 gallon water",
            "🌿 **Neem Oil**: Apply diluted neem oil spray every 7-14 days",
            "✂️ **Pruning**: Remove severely infected leaves and improve air circulation",
            "💨 **Air Circulation**: Increase spacing between plants and prune dense growth",
            "☀️ **Sunlight**: Ensure plants receive adequate sunlight",
            "💧 **Watering**: Water in the morning and avoid wetting leaves"
        ],
        "prevention": "Plant resistant varieties, ensure good air flow, and avoid overhead watering"
    },
    "Red Rust": {
        "description": "Fungal disease causing reddish-brown pustules on leaves.",
        "symptoms": "Reddish-brown or orange pustules on undersides of leaves, yellowing, defoliation",
        "treatments": [
            "🍃 **Remove Infected Leaves**: Collect and destroy affected leaves immediately",
            "🔥 **Destroy Debris**: Burn or bury infected plant material (don't compost)",
            "💨 **Improve Airflow**: Prune to increase air circulation around plants",
            "🌿 **Sulfur Dust**: Apply organic sulfur dust on affected areas (organic fungicide)",
            "🧄 **Garlic Spray**: Make garlic extract spray as natural fungicide",
            "🌱 **Compost Tea**: Spray beneficial microbe-rich compost tea weekly",
            "💧 **Water Management**: Avoid overhead watering and water in early morning"
        ],
        "prevention": "Plant resistant varieties, maintain plant spacing, and practice crop rotation"
    },
    "Termite": {
        "description": "Wood-eating insects that can severely damage cashew trees.",
        "symptoms": "Hollowed stems or branches, mud tubes on bark, weakened structure, wilting",
        "treatments": [
            "🔍 **Early Detection**: Regularly inspect trees for mud tubes and hollow sounds when tapped",
            "🪵 **Remove Dead Wood**: Clear dead wood and debris around trees promptly",
            "🐜 **Beneficial Nematodes**: Apply nematodes that prey on termites to soil",
            "🌿 **Neem Cake**: Mix neem cake into soil around tree base",
            "🧱 **Barrier Creation**: Create physical barriers with sand or diatomaceous earth",
            "🌾 **Mulch Management**: Keep mulch away from direct contact with tree trunk",
            "💧 **Soil Moisture**: Maintain proper soil moisture (termites prefer dry conditions)",
            "🍊 **Orange Oil**: Apply orange oil spray to affected areas (natural termiticide)"
        ],
        "prevention": "Regular inspection, remove wood debris, maintain tree health, and use resistant varieties"
    }
}

# Function to convert image to base64
def get_base64_image(image_path):
    """Convert image to base64 string"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

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
    model.load_state_dict(torch.load('mobilenet_v3_small_epoch_25.pth', map_location=device))
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
    
    # Get background image as base64
    bg_image = get_base64_image("cashew2.jpg")
    
    # Custom CSS with background image - FIXED VERSION
    background_style = ""
    if bg_image:
        background_style = f"""
        .stApp {{
            background-image: 
                linear-gradient(rgba(255, 255, 255, 0.85), rgba(255, 255, 255, 0.85)),
                url("data:image/jpg;base64,{bg_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        """
    
    st.markdown(f"""
        <style>
        {background_style}
        
        .main-header {{
            font-size: 3rem;
            color: #2E7D32;
            text-align: center;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(255,255,255,0.9);
            font-weight: bold;
        }}
        .sub-header {{
            font-size: 1.2rem;
            color: #1a1a1a;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 600;
            text-shadow: 1px 1px 3px rgba(255,255,255,0.9);
        }}
        
        /* Make text more readable on faded background */
        .stMarkdown, p, li, span {{
            text-shadow: 0px 0px 2px rgba(255,255,255,0.8);
        }}
        
        /* File uploader - keep some styling for visibility */
        div[data-testid="stFileUploader"] {{
            background-color: rgba(255, 255, 255, 0.7);
            padding: 1.5rem;
            border-radius: 10px;
            border: 2px dashed #4caf50;
            backdrop-filter: blur(3px);
        }}
        
        /* Info boxes - keep minimal background for readability */
        div[data-testid="stAlert"] {{
            background-color: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(3px);
        }}
        
        .prediction-box {{
            padding: 20px;
            border-radius: 10px;
            background-color: rgba(240, 242, 246, 0.9);
            margin: 10px 0;
            backdrop-filter: blur(5px);
        }}
        .healthy {{
            background-color: rgba(212, 237, 218, 0.9);
            color: #155724;
        }}
        .disease {{
            background-color: rgba(248, 215, 218, 0.9);
            color: #721c24;
        }}
        .treatment-section {{
            background-color: rgba(232, 245, 233, 0.85);
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #4caf50;
            margin: 20px 0;
            backdrop-filter: blur(3px);
        }}
        .stSidebar {{
            background-color: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
        }}
        div[data-testid="stExpander"] {{
            background-color: rgba(255, 255, 255, 0.85);
            border-radius: 8px;
            backdrop-filter: blur(3px);
        }}
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🌿 Cashew Disease Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image of a cashew plant to detect diseases and get treatment recommendations</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info(
            """
            This app detects cashew plant pests and diseases from images. It provides enviromentally friendly recommendations to control the pests and diseases.
            
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
        4. Read the organic treatment recommendations
        5. Check the probability distribution
        """)
       
        st.header("🌱 Why Organic?")
        st.success("""
        All treatment recommendations are **non-chemical** and environmentally friendly:
        - Safe for farmers and consumers
        - Sustainable and eco-friendly
        - Cost-effective
        - Protects beneficial insects
        """)
    
    # Load model
    try:
        model, device = load_model()
        #st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of the cashew plant"
    )
    
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file)
        
        # Two column layout for image and results
        col1, col2 = st.columns([1, 1])
        
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
                        st.markdown(f'<div class="prediction-box" style="background-color: rgba(255, 243, 205, 0.95); color: #856404;">', unsafe_allow_html=True)
                        st.markdown(f"### ⚠️ Prediction: **{disease_name}**")
                    else:
                        st.markdown(f'<div class="prediction-box disease">', unsafe_allow_html=True)
                        st.markdown(f"### 🦠 Prediction: **{disease_name}**")
                    
                    st.markdown(f"**Confidence:** {confidence*100:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Confidence meter
                    st.progress(confidence)
                    
                    # Interpretation
                    if confidence > 0.8:
                        st.success("🎯 High confidence - The model is quite certain about this prediction.")
                    elif confidence > 0.6:
                        st.warning("⚠️ Moderate confidence - Consider taking additional images for verification.")
                    else:
                        st.error("❓ Low confidence - The model is uncertain. Please upload a clearer image.")
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.stop()
        
        # Treatment Information Section
        st.markdown("---")
        if disease_name in TREATMENT_INFO:
            treatment = TREATMENT_INFO[disease_name]
            
            # Disease Description
            st.subheader(f"📋 About {disease_name}")
            st.info(treatment["description"])
            
            # Symptoms
            with st.expander("🔍 Symptoms to Look For", expanded=True):
                st.write(treatment["symptoms"])
            
            # Treatment Recommendations
            if disease_name not in ["Healthy", "Not Cashew"]:
                st.markdown('<div class="treatment-section">', unsafe_allow_html=True)
                st.subheader("🌿 Non-Chemical Treatment Recommendations")
                st.markdown("**Organic and Environmentally Friendly Solutions:**")
                
                for treatment_step in treatment["treatments"]:
                    st.markdown(treatment_step)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Prevention tips
                with st.expander("🛡️ Prevention Tips"):
                    st.success(treatment["prevention"])
                
                # Additional note
                st.warning("⚠️ **Important Note**: For severe infections, consult with local agricultural extension officers or plant pathologists for comprehensive management strategies.")
            
            elif disease_name == "Healthy":
                st.markdown('<div class="treatment-section">', unsafe_allow_html=True)
                st.subheader("✅ Maintenance Recommendations")
                for treatment_step in treatment["treatments"]:
                    st.markdown(treatment_step)
                st.markdown('</div>', unsafe_allow_html=True)
            
            else:  # Not Cashew
                st.subheader("📝 What to Do")
                for treatment_step in treatment["treatments"]:
                    st.markdown(treatment_step)
        
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
        st.subheader("📚 How This App Helps Farmers")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**✅ Early Detection**")
            st.caption("Identify diseases before they spread")
        
        with col2:
            st.markdown("**🌿 Organic Solutions**")
            st.caption("Get non-chemical treatment options")
        
        with col3:
            st.markdown("**💰 Save Money**")
            st.caption("Use affordable, sustainable methods")
        
        # Feature highlights
        st.markdown("---")
        st.subheader("🌟 Key Features")
        
        feature_col1, feature_col2 = st.columns(2)
        
        with feature_col1:
            st.markdown("""
            #### 🎯 Accurate Detection
            - AI-powered disease classification
            - Real-time image analysis
            - Confidence scores for reliability
            """)
            
        with feature_col2:
            st.markdown("""
            #### 🌱 Sustainable Solutions
            - 100% organic treatment methods
            - Environmentally safe practices
            - Cost-effective for small farmers
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #333; padding: 20px; text-shadow: 1px 1px 2px rgba(255,255,255,0.8);'>
            <p><strong>🌍 Developed by Uenr-ATPS | Powered by MobileNet V3 Small and IDRC Grant</strong></p>
            <p><em>For educational and research purposes | Always consult local agricultural experts for severe cases</em></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
