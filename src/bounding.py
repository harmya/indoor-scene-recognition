import numpy as np
import pandas as pd
import os
from PIL import Image
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2

model = YOLO('yolov9c.pt')

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
            new_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = box_class + 1
            modded_image += new_img.astype(np.uint8)
    modded_image = torch.tensor(modded_image)
    return modded_image

def mod_data(file_list):
    modded_data = []
    for file in file_list:
        result = stack_bb(file)
        modded_data.append(result)
    return torch.stack(modded_data)

file_list = []
with open("./data/TestImages.txt", 'r') as file:
    for line in file:
        line = line.strip()
        if (os.path.exists(f"./Images/{line}")):
            try:
                img = Image.open(f"./Images/{line}")
                img.verify()
            except(IOError,SyntaxError)as e:
                continue
            file_list.append("Images/" + line)

with open("./data/TestImagesYes.txt", "w") as f:
    for file in file_list:
        f.write(file)
        f.write("\n")

result = mod_data(file_list)
torch.save(result, "test.pt")