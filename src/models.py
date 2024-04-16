import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import PIL.Image
from data_loader import create_data_loaders, get_label_to_idx
from sklearn.metrics import accuracy_score, classification_report
from torchvision import models

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    num_gpus = torch.cuda.device_count()
    print(f'Number of GPUs: {num_gpus}')

    '''
    Models we are using:
    1. Vanilla CNN
    2. ResNet
    '''

    class CNN(nn.Module):
        def __init__(self, num_classes=len(get_label_to_idx())):
            super(CNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 18 * 18, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    class ResNet(nn.Module):
        def __init__(self, num_classes=len(get_label_to_idx())):
            super(ResNet, self).__init__()
            self.resnet = models.resnet18(weights="ResNet18_Weights.DEFAULT")
            self.resnet.fc = nn.Linear(512, num_classes)

        def forward(self, x):
            return torch.sigmoid(self.resnet(x))


    train, test = create_data_loaders()
    print(vars(train))
    print(f'Training on {len(train)} batches')
    print(f'Testing on {len(test)} batches')

    print('Training the model...')


    model = ResNet()
    model = model.to(device)
    print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-6)
    epochs = 25


    print('Training the model...')
    for epoch in range(epochs):
        model.train()
        for i, (images, labels) in enumerate(train):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train)}], Loss: {loss.item()}')

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test:
               images, labels = images.to(device), labels.to(device)
               outputs = model(images)
               _, predicted = torch.max(outputs.data, 1)
               total += labels.size(0)
               correct += (predicted == labels).sum().item()

            print(f'Accuracy of the network on the test images: {100 * correct / total} %')