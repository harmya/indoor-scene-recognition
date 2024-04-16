from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader, RandomSampler, WeightedRandomSampler, SequentialSampler
import os
import torchvision.transforms as transforms
import torch
import numpy as np

def get_label_to_idx():
    labels = "../data/labels.txt"
    label_to_idx = {}
    with open(labels) as f:
        for idx, label in enumerate(f):
            label = label.strip()
            label_to_idx[label] = idx

    return label_to_idx

def read_image_list(txt_path, images_dir):
    with open(txt_path, 'r') as file:
        image_names = file.read().splitlines()
    image_paths = []
    for line in image_names:
        if (os.path.exists(f"../data/images/{line}")):
            try:
                img = Image.open(f"../data/images/{line}")
                img.verify()
            except(IOError,SyntaxError)as e:
                continue
            image_paths.append("../data/images/" + line)
    return image_paths

def get_labels(txt_file):
    image_labels = []
    with open(txt_file, 'r') as f:
            for line in f:
                relative_image_path = line.strip()
                label = relative_image_path.split('/')[0]
                image_labels.append(label)
    return image_labels

class ToNumpyArray(object):
    def __call__(self, pic):
        return np.array(pic)

class CustomDataset(Dataset):
    def __init__(self, image_paths, txt_path, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.label_to_idx = get_label_to_idx()
        self.image_labels = [self.label_to_idx[label] for label in get_labels(txt_path)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.image_labels[index]
        return image, label


length = 300
width = 300
batch_size = 64
train_path = '../data/TrainImagesNo.txt'
test_path = '../data/TestImages.txt'

def create_data_loaders(sampling='random'):
    train_image_paths = read_image_list(train_path, '../data/images')
    test_image_paths = read_image_list(test_path, '../data/images')

    transform = transforms.Compose([
    transforms.Resize(length),  # Resize the short side to 256
    transforms.CenterCrop((length, width)),  # Crop the center to get 256x256
    ToTensor()
    ])

    if sampling == 'random':
        train_dataset = CustomDataset(train_image_paths, train_path, transform=transform)
        test_dataset = CustomDataset(test_image_paths, test_path, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        return train_loader, test_loader

    elif sampling == 'stratified':
        train_dataset = CustomDataset(train_image_paths, train_path, transform=transform)
        test_dataset = CustomDataset(test_image_paths, test_path, transform=transform)
        class_counts = np.bincount(train_dataset.image_labels)
        weights = [1.0 / class_counts[label] for label in train_dataset.image_labels ]
        sampler = WeightedRandomSampler(weights, len(train_dataset))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, sampler=sampler)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        return train_loader, test_loader

    elif sampling == 'sequential':
        train_dataset = CustomDataset(train_image_paths, train_path, transform=transform)
        test_dataset = CustomDataset(test_image_paths, test_path, transform=transform)
        sampler = SequentialSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, sampler=sampler)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

    else:
        raise ValueError("Unsupported sampling type: {}".format(sampling))

    return train_loader, test_loader

if __name__ == '__main__':
    print('Testing data loaders')
    print('-------------------')

    print('Testing Random Sampling')
    train, test = create_data_loaders(sampling='random')
    print(f'Training on {len(train)} batches')
    print(f'Testing on {len(test)} batches')
    print(len(train.dataset))

    for batch in train:
        print(batch[1][:10])
        break

    print('-------------------')

    print('Testing Stratified Sampling')
    train, test = create_data_loaders(sampling='stratified')
    print(f'Training on {len(train)} batches')
    print(f'Testing on {len(test)} batches')
    print(len(train.dataset))
    for batch in train:
        print(batch[1][:10])
        break

    print('-------------------')
    print('Testing Sequential Sampling')
    train, test = create_data_loaders(sampling='sequential')
    print(f'Training on {len(train)} batches')
    print(f'Testing on {len(test)} batches')
    print(len(train.dataset))
    for batch in train:
        print(batch[1][:10])
        break