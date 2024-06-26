import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from dataloader import get_data_loader
from models import ResNet, ENet
 
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    num_gpus = torch.cuda.device_count()

    # Load the data
    BOUNDING_BOXES = True
    train_loader, val_loader, test_loader = get_data_loader(sampling = "random", bounding_boxes = BOUNDING_BOXES)

    # Load the model
    # Use nn.DataParallel if multiple GPUs are available
    if num_gpus > 1:
        model = nn.DataParallel(ResNet()).to(device)
    else:
        model = ResNet().to(device)

    # Hyperparameters
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=0.05)
    
    EPOCHS = 20

    train_losses = []
    val_losses = []
    test_losses = []

    for epoch in range(EPOCHS):
        # Training Loop
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device).float()
            labels = labels.to(device).long()

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss and perform backprop
            loss = criterion(outputs, labels)
            loss.backward()

            # Save train loss
            train_loss = loss.item()
            train_losses.append(train_loss)

            optimizer.step()

            # Print Epoch Loss every 10 steps
            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')

        # Training Accuracy
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in train_loader:
                images = images.to(device).float()
                labels = labels.to(device).long()

                outputs = model(images)

                # Save val loss
                val_loss = criterion(outputs, labels).item()
                val_losses.append(val_loss)

                # Find predictions and Compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Epoch [{epoch+1}/{EPOCHS}], Training Accuracy: {100 * correct / total}')

        # Validation Loop
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images = images.to(device).float()
                labels = labels.to(device).long()

                outputs = model(images)

                # Save val loss
                val_loss = criterion(outputs, labels).item()
                val_losses.append(val_loss)

                # Find predictions and Compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Epoch [{epoch+1}/{EPOCHS}], Validation Accuracy: {100 * correct / total}')

        # Testing Loop
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device).float()
                labels = labels.to(device).long()

                outputs = model(images)

                # Save test loss
                test_loss = criterion(outputs, labels).item()
                test_losses.append(test_loss)

                # Find predictions and Compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print(f'Epoch [{epoch+1}/{EPOCHS}], Test Accuracy: {100 * correct / total}')

    # Plot train, val, and test losses
    # plt.plot(range(1, EPOCHS+1), train_losses, label='Train Loss')
    # plt.plot(range(1, EPOCHS+1), val_losses, label='Val Loss')
    # plt.plot(range(1, EPOCHS+1), test_losses, label='Test Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()