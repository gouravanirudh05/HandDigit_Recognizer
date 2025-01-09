import numpy as np
import cv2 as cv
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch import nn

from cnn_model import CNN_Hand_Digit_Recognizer, accuracy_fn  # Import the model and utility

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Data preparation
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())
class_names = train_dataset.classes

BATCH_SIZE = 32
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model setup
cnn = CNN_Hand_Digit_Recognizer(input_shape=1, hidden_units=10, output_shape=len(class_names)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=cnn.parameters(), lr=0.01)

# Training and Evaluation
epochs = 5
for epoch in range(epochs):
    print(f"Epoch: {epoch + 1}/{epochs}\n---------")
    cnn.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = cnn(X)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

    # Validation
    cnn.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = cnn(X)
            test_loss += loss_fn(y_pred, y).item()
            test_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

# Final evaluation
print(f"Final model loss: {test_loss:.2f}")
print(f"Final model accuracy: {test_acc:.2f}%")

# Inference on a custom image
image = cv.imread(r"6.jpg", cv.IMREAD_GRAYSCALE)
image = cv.resize(image, (28, 28))

data = torch.Tensor(np.array(image)).unsqueeze(0).unsqueeze(0).to(device)
data = (255 - data) / 255  # Normalize and invert colors

cnn.eval()
with torch.inference_mode():
    prediction_logits = cnn(data)
    prediction_probs = torch.softmax(prediction_logits, dim=1)
    prediction = prediction_probs.argmax(dim=1)

print(f'The written number looks like {prediction.item()} | Confidence: {prediction_probs.max().item() * 100:.2f}%')
