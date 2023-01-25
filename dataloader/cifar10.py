import pickle
from os.path import join

import numpy as np

from tools import DATASET, get_dataset_root
from .dataset import CommonDataset


class Cifar10(CommonDataset):
    def __init__(self, opt, mode, num_sample, no_normalize):
        self.data_name = DATASET.cifar10
        super(Cifar10, self).__init__(opt, mode, num_sample, no_normalize)

    def load_data(self):
        train_list = [f'data_batch_{i + 1}' for i in range(5)]
        test_list = ['test_batch']
        root = join(get_dataset_root(), self.data_name)
        file_list = train_list if self.is_train else test_list

        images, labels = [], []
        for file_name in file_list:
            file_path = join(root, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                images.append(entry['data'])
                labels.extend(entry['labels'])
        # 原始数据通道在前
        images = np.concatenate(images).reshape(-1, 3, self.img_size, self.img_size)
        # 需要转化为一般读取图像的通道在后的顺序(ToTensor之后又会转化为通道在前)
        images = images.transpose((0, 2, 3, 1))
        return images, np.array(labels)


if __name__ == '__main__':
    pass
