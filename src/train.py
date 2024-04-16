import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from dataloader import get_data_loader
from models import ResNet
 
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Load the data
    BOUNDING_BOXES = True
    train_loader, val_loader, test_loader = get_data_loader(bounding_boxes = BOUNDING_BOXES)

    # Load the model
    model = ResNet().to(device)

    # Hyperparameters
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    EPOCHS = 10

    for epoch in range(EPOCHS):
        # Training Loop
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
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Epoch [{epoch+1}/{EPOCHS}], Validation Accuracy: {100 * correct / total}')

        # Testing Loop
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Epoch [{epoch+1}/{EPOCHS}], Test Accuracy: {100 * correct / total}')