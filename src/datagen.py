import random, torch
from PIL import Image
import os

TRAIN = 0
VAL = 1
TEST = 2

def gen_data_tensors():
    with open("../data/TrainImages.txt", "r") as f:
        train_images = f.readlines()
    with open("../data/ValImages.txt", "r") as f:
        val_images = f.readlines()
    with open("../data/TestImages.txt", "r") as f:
        test_images = f.readlines()

    # Run the below command to create the required subdirectories in the tensors folder:
    image_subdirectories = [name for name in os.listdir("../data/images") if os.path.isdir(os.path.join("../data/images", name))]

    for subdirectory in image_subdirectories:
        os.makedirs(os.path.join("../data/tensors", subdirectory), exist_ok=True)

    train_images = [img.strip() for img in train_images]
    val_images = [img.strip() for img in val_images]
    test_images = [img.strip() for img in test_images]

    for img in train_images:
        image = Image.open(f"../data/images/{img}")
        image = image.resize((300, 300))
        torch.save(image, f"../data/tensors/{img}.pt")
    for img in val_images:
        image = Image.open(f"../data/images/{img}")
        image = image.resize((300, 300))
        torch.save(image, f"../data/tensors/{img}.pt")
    for img in test_images:
        image = Image.open(f"../data/images/{img}")
        image = image.resize((300, 300))
        torch.save(image, f"../data/tensors/{img}.pt")

def split_train_val():
    with open("../data/rawTrainSplit.txt", "r") as f:
        unfiltered_content = f.readlines()
        content = []
        for line in unfiltered_content:
            if not line.endswith("gif.jpg\n"):
                try:
                    img = Image.open(f"../data/images/{line.strip()}")
                    img.verify()
                    content.append(line)
                except (IOError, SyntaxError) as e:
                    continue

        # shuffle with seed 42
        random.Random(42).shuffle(content)

        # 90 - 10 train val split
        train_content = content[:int(0.9*len(content))]
        val_content = content[int(0.9*len(content)):]

        with open("../data/TrainImages.txt", "w") as f:
            f.writelines(train_content)
        with open("../data/ValImages.txt", "w") as f:
            f.writelines(val_content)

def gen_test_set():
    with open("../data/rawTestSplit.txt", "r") as f:
        unfiltered_content = f.readlines()
        content = []
        for line in unfiltered_content:
            if not line.endswith("gif.jpg\n"):
                try:
                    img = Image.open(f"../data/images/{line.strip()}")
                    img.verify()
                    content.append(line)
                except (IOError, SyntaxError) as e:
                    continue
                
            with open("../data/TestImages.txt", "w") as f:
                f.writelines(content)
        
def gen_labels(split = TRAIN):
    splitName = "Train"
    if split == VAL:
        splitName = "Val"
    elif split == TEST:
        splitName = "Test"
    with open(f"../data/{splitName}Images.txt", "r") as f:
        images = f.readlines()
    with open(f"../data/{splitName}Labels.txt", "w") as f:
        for image in images:
            label = image.split("/")[0]
            f.write(label + "\n")

if __name__ == "__main__":
    # to split the data into train and val : split_train_val()
    # to generate test set: gen_test_set()
    # to generate tensors from images: gen_data_tensors()
    # to generate labels: gen_labels(TRAIN), gen_labels(VAL), gen_labels(TEST)
    pass