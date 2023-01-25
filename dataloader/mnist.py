import gzip
from os.path import join

import numpy as np

from tools import DATASET, get_dataset_root
from .dataset import CommonDataset


class Mnist(CommonDataset):
    def __init__(self, opt, mode, num_sample, no_normalize):
        self.data_name = DATASET.mnist
        super(Mnist, self).__init__(opt, mode, num_sample, no_normalize)

    def load_data(self):
        train_file = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz']
        test_file = ['t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
        root = join(get_dataset_root(), self.data_name)
        image_file, label_file = train_file if self.is_train else test_file

        with gzip.open(join(root, image_file), 'rb') as f:
            # 直接读出来的数据是只读的，因此要使用copy复制一份
            images = np.frombuffer(f.read(), np.uint8, offset=16).copy().reshape(-1, self.img_size, self.img_size)
        with gzip.open(join(root, label_file), 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8).copy()
        return images, labels.astype(np.int64)


if __name__ == '__main__':
    pass
