import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

# Load the trained AlexNet model
model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)  # Updated pretrained loading
model.classifier[6] = torch.nn.Linear(4096, 2)  # Modify classifier for binary classification

# Load trained weights (Ensure the file exists)
model_path = "pcos_alexnet.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found.")

model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  # Set model to evaluation mode

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to predict infection status
def predict_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found.")

    image = Image.open(image_path).convert("RGB")  # Ensure 3-channel image
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    class_names = [ "Affected", "Non-Infected"]
    return class_names[predicted.item()]

# Test with an image
image_path = "C:/Users/91883/Downloads/c.jpg" # Change to your image path
prediction = predict_image(image_path)
print(f"Prediction: {prediction}")
