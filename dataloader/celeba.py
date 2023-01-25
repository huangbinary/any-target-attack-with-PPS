import os
import shutil
from os.path import join

import numpy as np
from tqdm import tqdm

from tools import get_dataset_root, DATASET
from .dataset import CommonDataset


class Celeba(CommonDataset):
    def __init__(self, opt, mode, num_sample, no_normalize):
        self.data_name = DATASET.celeba
        super(Celeba, self).__init__(opt, mode, num_sample, no_normalize)

    def load_data(self):
        root = join(get_dataset_root(), self.data_name, 'train_pro' if self.is_train else 'test_pro')
        images, labels = [], []
        for label_root in os.scandir(root):
            for img_file in os.scandir(label_root.path):
                images.append(img_file.path)
                labels.append(int(label_root.name))
        return np.array(images), np.array(labels)


class CelebaHandler:
    def __init__(self):
        super(CelebaHandler, self).__init__()
        root = join(get_dataset_root(), DATASET.celeba)
        self.raw_root = join(root, 'celeba_raw')
        self.attribute_file = join(root, 'list_attr_celeba.txt')
        self.partition_file = join(root, 'list_eval_partition.txt')
        self.train_root = join(root, 'train_pro')
        if os.path.exists(self.train_root):
            shutil.rmtree(self.train_root)
        self.test_root = join(root, 'test_pro')
        if os.path.exists(self.test_root):
            shutil.rmtree(self.test_root)
        self.target_attribute = ['Heavy_Makeup', 'Mouth_Slightly_Open', 'Smiling']
        self.target_attribute_index = []

    def process(self):
        train_or_test = {}
        with open(self.partition_file, 'r')as f:
            contents = f.readlines()
            for line in contents:
                line = line.split()
                train_or_test[line[0]] = int(line[1])
        with open(self.attribute_file, 'r')as f:
            contents = f.readlines()
            attributes = contents[1].split()
            for attr in self.target_attribute:
                self.target_attribute_index.append(attributes.index(attr))
            for img_info in tqdm(contents[2:]):
                img_info = img_info.split()
                img_path = join(self.raw_root, img_info[0])
                label = 0
                for idx in self.target_attribute_index:
                    label = label * 2 + (img_info[1 + idx] == '1')
                if train_or_test[img_info[0]] == 1:
                    continue
                label_root = join(self.train_root if train_or_test[img_info[0]] == 0 else self.test_root, str(label))
                if not os.path.exists(label_root):
                    os.makedirs(label_root)
                shutil.copy(img_path, join(label_root, img_info[0]))
