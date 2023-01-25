import csv
import os
from os.path import join

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from tools import get_dataset_root, get_csv_path, DATASET
from .dataset import CommonDataset


class Gtsrb(CommonDataset):
    """ 下载地址
    https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html
    """

    def __init__(self, opt, mode, num_sample, no_normalize):
        self.data_name = DATASET.gtsrb
        super(Gtsrb, self).__init__(opt, mode, num_sample, no_normalize)

    def load_data(self):
        """
        加载没有经过GtsrbHandler处理过的原始数据
        """
        root = join(get_dataset_root(), self.data_name)
        if not self.is_train:
            images, labels = self.__load_one_dir_data(join(root, 'test_raw'))
        else:
            root = join(root, 'train_raw')
            images, labels = [], []
            for label_str in os.listdir(root):
                one_dir_images, one_dir_labels = self.__load_one_dir_data(join(root, label_str))
                images.extend(one_dir_images)
                labels.extend(one_dir_labels)
        return np.array(images), np.array(labels)

    def __load_one_dir_data(self, root):
        images, labels = [], []
        csv_data = np.array(pd.read_csv(get_csv_path(root), delimiter=';'))
        for img_info in csv_data:
            images.append(join(root, img_info[0]))
            labels.append(int(img_info[7]))
        return images, labels

    def __load_pro_data(self):
        root = join(get_dataset_root(), self.data_name)
        csv_path = get_csv_path(join(root, 'train_pro_bak' if self.is_train else 'test_pro_bak'))
        csv_data = np.array(pd.read_csv(csv_path))
        images, labels = [], []
        for img_path, label in csv_data:
            images.append(img_path)
            labels.append(int(label))
        return images, labels


class GtsrbHandler:
    def __init__(self):
        super(GtsrbHandler, self).__init__()
        root = join(get_dataset_root(), DATASET.gtsrb)
        self.train_raw_dir = join(root, 'train_raw')
        self.test_raw_dir = join(root, 'test_raw')
        self.label_path = None
        self.count = 0

    def process(self):
        self.tqdm = tqdm(range(44))
        self.__handle_train_raw()
        self.__handle_test_raw()

    def __handle_one_dir(self, raw_dir, pro_dir):
        if not os.path.exists(pro_dir):
            os.makedirs(pro_dir)
        csv_data = np.array(pd.read_csv(get_csv_path(raw_dir)))
        with open(self.label_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for i in range(csv_data.shape[0]):
                # name width height x1 y1 x2 y2 class
                #  0     1     2    3  4  5  6    7
                img_info = csv_data[i][0].split(";")
                box = [int(img_info[3]), int(img_info[4]), int(img_info[5]), int(img_info[6])]
                img = Image.open(join(raw_dir, img_info[0])).crop(box)
                img_path = join(pro_dir, f'{i:05d}.png')
                img.save(img_path)
                writer.writerow([img_path, img_info[7]])

        self.count += 1
        info = 'test set' if self.count == 44 else f'train class {self.count}'
        self.tqdm.set_description(f'processing {info}')
        self.tqdm.update()

    def __generate_label(self, pro_dir):
        if not os.path.exists(pro_dir):
            os.makedirs(pro_dir)
        self.label_path = join(pro_dir, 'labels.csv')
        with open(self.label_path, 'w', newline='') as f:
            csv.writer(f).writerow(['image', 'label'])

    def __handle_train_raw(self):
        pro_dir = self.train_raw_dir[:-3] + 'pro'
        self.__generate_label(pro_dir)
        for label in os.listdir(self.train_raw_dir):
            raw_label_dir = join(self.train_raw_dir, label)
            pro_label_dir = join(pro_dir, label)
            self.__handle_one_dir(raw_label_dir, pro_label_dir)

    def __handle_test_raw(self):
        pro_dir = self.test_raw_dir[:-3] + 'pro'
        self.__generate_label(pro_dir)
        self.__handle_one_dir(self.test_raw_dir, pro_dir)


if __name__ == "__main__":
    pass
