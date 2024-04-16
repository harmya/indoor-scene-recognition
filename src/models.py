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
    
