import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

TRAIN = 0
VAL = 1
TEST = 2

def find_file_length(file_path):
    with open(file_path, "r") as f:
        content = f.readlines()
        return len(content)

class ISRDataset(Dataset):
    def __init__(self, bb, enum):
        self.bb = bb
        self.enum = enum
        
    def __len__(self):
        tag = "No"
        if self.bb:
            tag = "Yes"
        
        if self.enum == TRAIN:
            return find_file_length(f"../data/TrainImages{tag}.txt")
        elif self.enum == VAL:
            return find_file_length(f"../data/ValImages{tag}.txt")
        else:
            return find_file_length(f"../data/TestImages{tag}.txt")
    
    def __getitem__(self, idx):
        tag = "No"
        if self.bb:
            tag = "Yes"
        
        if self.enum == TRAIN:
            with open(f"../data/TrainImages{tag}.txt", "r") as f:
                image_path = f.readlines()[idx].strip()
            with open(f"../data/TrainLabels{tag}.txt", "r") as f:
                label = f.readlines()[idx].strip()
        elif self.enum == VAL:
            with open(f"../data/ValImages{tag}.txt", "r") as f:
                image_path = f.readlines()[idx].strip()
            with open(f"../data/ValLabels{tag}.txt", "r") as f:
                label = f.readlines()[idx].strip()
        else:
            with open(f"../data/TestImages{tag}.txt", "r") as f:
                image_path = f.readlines()[idx].strip()
            with open(f"../data/TestLabels{tag}.txt", "r") as f:
                label = f.readlines()[idx].strip()
        
        image = Image.open(image_path)
        image = image.convert("RGB")
        
        label = int(label)
        
        return image, label


def get_label_to_idx():
    label_to_idx = {}
    with open('../data/labels.txt') as f:
        for idx, line in enumerate(f):
            label = line.strip()
            label_to_idx[label] = idx
    return label_to_idx

def get_data_loader(sampling = "random", bounding_boxes = True):
    return None, None, None