from PIL import Image
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
        if self.enum == TRAIN:
            return find_file_length(f"../data/TrainImages.txt")
        elif self.enum == VAL:
            return find_file_length(f"../data/ValImages.txt")
        else:
            return find_file_length(f"../data/TestImages.txt")
    
    def __getitem__(self, idx):
        if self.enum == TRAIN:
            with open(f"../data/TrainImages.txt", "r") as f:
                image_path = f.readlines()[idx].strip()
            with open(f"../data/TrainLabels.txt", "r") as f:
                label = f.readlines()[idx].strip()
        elif self.enum == VAL:
            with open(f"../data/ValImages.txt", "r") as f:
                image_path = f.readlines()[idx].strip()
            with open(f"../data/ValLabels.txt", "r") as f:
                label = f.readlines()[idx].strip()
        else:
            with open(f"../data/TestImages.txt", "r") as f:
                image_path = f.readlines()[idx].strip()
            with open(f"../data/TestLabels.txt", "r") as f:
                label = f.readlines()[idx].strip()
        
        image = Image.open(image_path)
        
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
    trainData = ISRDataset(bb = bounding_boxes, enum = TRAIN)
    valData = ISRDataset(bb = bounding_boxes, enum = VAL)
    testData = ISRDataset(bb = bounding_boxes, enum = TEST)

    train_loader = DataLoader(trainData, batch_size=32, shuffle=True)
    val_loader = DataLoader(valData, batch_size=32, shuffle=False)
    test_loader = DataLoader(testData, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader