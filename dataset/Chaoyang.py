# Implementation of Chaoyang dataset, adopted from
# https://github.com/bupt-ai-cz/HSA-NRL/tree/9d404dd671f675c3b3bd8f430c5708a7a35ae57d
# modified by Yuchuan Li

import torch.utils.data as data
from PIL import Image
import os
import json


class CHAOYANG(data.Dataset):
    def __init__(self, root, json_name=None, path_list=None, label_list=None, train=True, transform=None):
        imgs = []
        labels = []

        # get information from json file
        if json_name:
            json_path = os.path.join(root, json_name)
            with open(json_path, 'r') as f:
                load_list = json.load(f)
                for i in range(len(load_list)):
                    img_path = os.path.join(root, load_list[i]["name"])
                    imgs.append(img_path)
                    labels.append(load_list[i]["label"])

        # get images and labels from path lists
        if path_list and label_list:
            imgs = path_list
            labels = label_list
        self.transform = transform
        self.train = train  # training set or test set
        self.dataset = 'chaoyang'

        # set the number of class
        self.nb_classes = 4

        # get the data and labels
        if self.train:
            self.train_data, self.train_labels = imgs, labels
        else:
            self.test_data, self.test_labels = imgs, labels

    # item fetch function
    def __getitem__(self, index):
        # selection of train/test partition
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.open(img)

        # apply transform
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    # length calculation function
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)