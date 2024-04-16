import os
from PIL import Image

TRAIN = 0
VAL = 1
TEST = 2

def filter_data():
    for root, dirs, files in os.walk("../data/images"):
        for file in files:
            file_path = os.path.join(root, file)
            if not file.endswith("gif.jpg"):
                try:
                    img = Image.open(file_path)
                    img.verify()
                except:
                    os.remove(file_path)
            else:
                os.remove(file_path)

def gen_data_files(enum = 0, bb = False):
    tag = "Train"
    if enum == 1:
        tag = "Val"
    if enum == 2:
        tag = "Test"
    tag1 = "Yes" if bb else "No"
    file_path = "../data/{tag}Images{tag1}.txt"
    # with open(file_path, "w") as f:

if __name__ == "__main__":
    filter_data()
    gen_data_files(bb = True)
    gen_data_files(bb = False)