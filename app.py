import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import torch.nn as nn

# Load the trained ResNet-50 model
@st.cache_resource
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(2048, 2)  # Adjusting for binary classification
    model.load_state_dict(torch.load("pcos_resnet50.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prediction function
def predict(image, model):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return "PCOS Detected" if predicted.item() == 0 else " No PCOS"

# Streamlit UI
st.title("PCOS Ultrasound Image Classification with ResNet-50")
st.write("Upload an ultrasound image to classify whether PCOS is detected or not.")

model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Predict"):
        result = predict(image, model)
        st.success(f"Prediction: {result}")