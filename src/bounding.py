import numpy as np
import pandas as pd
import os
from PIL import Image
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2
import math

TRAIN = 0
VAL = 1
TEST = 2

model = YOLO('yolov9c.pt')

def positional_encoding_2d(x, y, b, h):
    return math.sin(x / 10000 ** ( b * h / 90000)) + math.cos(y / 10000 ** ( b * h / 90000))

def stack_bb(file_name):
    results = model.predict(file_name, conf=0.5)
    image = cv2.imread(file_name)
    modded_image = image
    for result in results:
        boxes = result.boxes
        for box in boxes:
            new_img = np.zeros(image.shape)
            box_coords = box.xyxy
            box_class = box.cls
            top_left = [int(box_coords[0][0]), int(box_coords[0][1])]
            bottom_right = [int(box_coords[0][2]), int(box_coords[0][3])]
            x = top_left[0]
            y = top_left[1]
            b = bottom_right[0] - top_left[0]
            h = bottom_right[1] - top_left[1]
            for i in range(x, x + b):
                for j in range(y, y + h):
                    moded_image[i][j] += positional_encoding_2d(i, j, b, h)
    modded_image = torch.tensor(modded_image)
    return modded_image

def mod_data(file_list):
    modded_data = []
    for file in file_list:
        result = stack_bb(file)
        modded_data.append(result)
    return torch.stack(modded_data)

split = TRAIN
file_path = ""
file_labels = []
if split == TRAIN:
    file_path = "../data/TrainImages.txt"

elif split == VAL:
    file_path = "../data/ValImages.txt"

elif split == TEST:
    file_path = "../data/TestImages.txt"

with open("../data/TestImages.txt", 'r') as file:
    for line in file:
        line = line.strip()
        if (os.path.exists(f"./Images/{line}")):
            try:
                img = Image.open(f"./Images/{line}")
                img.verify()
            except(IOError,SyntaxError)as e:
                continue
            file_list.append("Images/" + line)

with open("../data/TestImagesYes.txt", "w") as f:
    for file in file_list:
        f.write(file)
        f.write("\n")

result = mod_data(file_list)
torch.save(result, "test.pt")