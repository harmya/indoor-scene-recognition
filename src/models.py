import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import accuracy_score, classification_report
from dataloader import get_label_to_idx
from PIL import Image

class ResNet(nn.Module):
    def __init__(self, num_classes=len(get_label_to_idx())):
        super(ResNet, self).__init__()
        self.resnet = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return torch.sigmoid(self.resnet(x))
    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Load the data
    BOUNDING_BOXES = True
    train_loader, test_loader = get_data_loader(bounding_boxes = BOUNDING_BOXES)

    # Load the model
    model = ResNet().to(device)

    # Hyperparameters
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    EPOCHS = 10

    # Training Loop
    for epoch in range(EPOCHS):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss and perform backprop
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            # Print Epoch Loss every 100 steps
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')

    # Validation Loop

    # Testing Loop