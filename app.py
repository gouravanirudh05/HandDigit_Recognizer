import streamlit as st
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import numpy as np
import cv2 as cv

# Define the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the CNN model
class CNN_Hand_Digit_Recognizer(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 7 * 7, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        return self.classifier(self.block_2(self.block_1(x)))

# Instantiate the model
cnn = CNN_Hand_Digit_Recognizer(input_shape=1, hidden_units=10, output_shape=10)

# Load the trained model weights (provide your model path)
# cnn.load_state_dict(torch.load("model.pth"))
cnn.to(device)
cnn.eval()

# Streamlit App
st.title("Handwritten Digit Recognizer")
st.write("Upload an image of a handwritten digit, and the model will predict the digit and its confidence.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv.imdecode(file_bytes, cv.IMREAD_GRAYSCALE)

    # Preprocess the image
    image = cv.resize(image, (28, 28))
    data = torch.Tensor(np.array(image))
    data = (255 - data) / 255  # Normalize and invert colors
    data = data.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    data = data.to(device)

    # Make predictions
    with torch.inference_mode():
        prediction_logits = cnn(data)
        prediction_probs = torch.softmax(prediction_logits, dim=1)
        prediction = prediction_probs.argmax(dim=1)

    # Display results
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write(f"Predicted Digit: **{prediction.item()}**")
    st.write(f"Confidence: **{prediction_probs.max().item() * 100:.2f}%**")
