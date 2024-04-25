import numpy as np
import pandas as pd
import os, sys
from PIL import Image
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2

TRAIN = 0
VAL = 1
TEST = 2
map_text = {TRAIN: "Train", VAL: "Val", TEST: "Test"}

model = YOLO('yolov9c.pt')

def stack_bb(file_name):
    image = torch.load(file_name)
    results = model.predict(image, conf=0.5)
    transform = transforms.Compose([ 
        transforms.PILToTensor() 
    ]) 
    image = transform(image)
    modded_image = image.long()
    for result in results:
        boxes = result.boxes
        for box in boxes:
            new_img = torch.zeros(image.shape)
            box_coords = box.xyxy
            box_class = box.cls
            top_left = [int(box_coords[0][0]), int(box_coords[0][1])]
            bottom_right = [int(box_coords[0][2]), int(box_coords[0][3])]
            new_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = box_class + 1
            modded_image += new_img.long()
    modded_image = torch.tensor(modded_image)
    if not os.path.exists(f"../data/tensors_bb/{file_name.split('/')[-2]}"):
        os.makedirs(f"../data/tensors_bb/{file_name.split('/')[-2]}")
    torch.save(modded_image, "../data/tensors_bb/" + "/".join(file_name.split('/')[-2:]))

def mod_data(file_list):
    for file in file_list:
        stack_bb(file)

# TODO : make it a function which takes in an enum (TRAIN, VAL, TEST) and saves the bounded tensors
split = map_text[int(sys.argv[1])]
file_list = []
with open(f"../data/{split}Images.txt", 'r') as file:
    for line in file:
        line = line.strip()
        file_list.append("../data/tensors/" + line + ".pt")

print(len(file_list))
mod_data(file_list)