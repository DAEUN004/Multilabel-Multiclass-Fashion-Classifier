# data loader
import tensorflow as tf
from torchvision import transforms
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from keras.utils import load_img, img_to_array, to_categorical
import requests
import torch
import torch.nn.functional as F
import cv2
import collections
from imblearn.over_sampling import RandomOverSampler

NUM_ATTR = 6
NUM_CLASSES_PER_CATEGORY = [7, 3, 3, 4, 6, 3]
class FashionNet_Dataset(Dataset):

    def __init__(self, root, txt):
        self.img_path = []
        self.labels = [[] for _ in range(NUM_ATTR)]

        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                # make dummy label for test set
                if 'test' in txt:
                    for i in range(NUM_ATTR):
                        self.labels[i].append(0)
        if 'test' not in txt:
            with open(txt.replace('.txt', '_attr.txt')) as f:
                for line in f:
                    attrs = line.split()
                    for i in range(NUM_ATTR):
                        self.labels[i].append(int(attrs[i]))
        
    def __len__(self):
        return len(self.labels[0])

    def __getitem__(self, index):

        path = self.img_path[index]
        label = np.array([self.labels[i][index] for i in range(NUM_ATTR)])

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])
        sample = transform(sample)


        label = torch.tensor(label)

        one_hot_labels = []
        num_classes = [7, 3, 3, 4, 6, 3]
        for i in range(NUM_ATTR):
            one_hot_label = torch.nn.functional.one_hot(torch.tensor(label[i].item()), num_classes=num_classes[i])
            one_hot_labels.append(one_hot_label.unsqueeze(0))
        label = torch.cat(one_hot_labels, dim=1)

        return sample, label


    def count_classes(self):
        # Convert labels to one-hot encoding
        one_hot_labels = []
        for i in range(NUM_ATTR):
            num_classes = NUM_CLASSES_PER_CATEGORY[i]
            one_hot_label = torch.nn.functional.one_hot(torch.tensor(self.labels[i]), num_classes=num_classes)
            one_hot_labels.append(one_hot_label)
        one_hot_labels = torch.cat(one_hot_labels, dim=1)

        # Sum along axis 0 to get the count of samples in each class
        class_counts = torch.sum(one_hot_labels, dim=0)

        # Print the count of samples in each class
        for i, count in enumerate(class_counts):
            print(f"Class {i}: {count} samples")
        
        return class_counts


