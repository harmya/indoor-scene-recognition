import torch.nn as nn
from torchvision import models
from dataloader import get_label_to_idx

class ResNet(nn.Module):
    def __init__(self, num_classes=len(get_label_to_idx())):
        super(ResNet, self).__init__()
        self.resnet = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        self.resnet.fc = nn.Linear(512, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.resnet(x))