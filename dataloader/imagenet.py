import os
from os.path import join

import numpy as np

from tools import get_dataset_root, DATASET
from .dataset import CommonDataset


class Imagenet(CommonDataset):
    def __init__(self, opt, mode, num_sample, no_normalize):
        self.data_name = DATASET.imagenet
        super(Imagenet, self).__init__(opt, mode, num_sample, no_normalize)

    def load_data(self):
        root = join(get_dataset_root(), self.data_name)
        train_root = join(root, 'train')
        label_to_idx, images, labels = {}, [], []
        for idx, label_str in enumerate(os.listdir(train_root)):
            # 多目标攻击只加载指定类别数量的数据
            if self.opt.attack_num > 1 and self.opt.attack_num == idx:
                break
            label_to_idx[label_str] = idx
            if self.is_train:
                image_root = join(train_root, label_str, 'images')
                images.extend(list(map(lambda s: s.path, os.scandir(image_root))))
                labels.extend([idx] * (len(images) - len(labels)))
        if not self.is_train:
            val_root = join(root, 'val')
            with open(join(val_root, 'val_annotations.txt'))as f:
                for img_info in f.readlines():
                    img_info = img_info.split('\t')
                    label_name = img_info[1]
                    if label_name not in label_to_idx:
                        continue
                    images.append(join(val_root, 'images', img_info[0]))
                    labels.append(label_to_idx[label_name])
        return np.array(images), np.array(labels)
