import numpy as np
import torch
from torchvision import transforms

from tools import IMG_SIZE, DATASET, MEAN, STD


def get_transform(data_name, is_train, no_normalize=False):
    trans = []
    trans.append(transforms.Resize((IMG_SIZE[data_name], IMG_SIZE[data_name])))
    if is_train and not no_normalize:
        trans.append(transforms.RandomCrop((IMG_SIZE[data_name], IMG_SIZE[data_name]), padding=4))
        if data_name == DATASET.gtsrb:
            trans.append(transforms.RandomRotation(10))
            trans.append(transforms.ColorJitter(brightness=0.2))
        elif data_name != DATASET.mnist:
            trans.append(transforms.RandomRotation(10))
            trans.append(transforms.RandomHorizontalFlip(p=0.5))
    trans.append(transforms.ToTensor())
    if not no_normalize:
        trans.append(Normalize(data_name))
    return transforms.Compose(trans)


class Calibrate:
    def __init__(self, data_name):
        super(Calibrate, self).__init__()
        self.mean = MEAN[data_name]
        self.std = STD[data_name]

    def __call__(self, x: torch.Tensor):
        if self.mean is None:
            return x
        y = x.clone()
        # 3维单图像和4位一批图像都能取到正确的通道数
        for c in range(x.shape[-3]):
            self.calculate(x, y, c)
        return y

    def calculate(self, x, y, c):
        raise NotImplementedError


class Normalize(Calibrate):
    def __init__(self, data_name):
        super(Normalize, self).__init__(data_name)

    def calculate(self, x, y, c):
        # 注意处理单通道和多通道的情况
        if len(x.shape) == 4:
            y[:, c] = (x[:, c] - self.mean[c]) / self.std[c]
        else:
            y[c] = (x[c] - self.mean[c]) / self.std[c]


class Denormalize(Calibrate):
    def __init__(self, data_name):
        super(Denormalize, self).__init__(data_name)

    def calculate(self, x, y, c):
        if len(x.shape) == 4:
            y[:, c] = x[:, c] * self.std[c] + self.mean[c]
        else:
            y[c] = x[c] * self.std[c] + self.mean[c]


class ToTensor(Normalize):
    def __init__(self, data_name):
        super(ToTensor, self).__init__(data_name)

    def __call__(self, x):
        return super().__call__(transforms.ToTensor()(x))


class ToNumpy(Denormalize):
    def __init__(self, data_name):
        super(ToNumpy, self).__init__(data_name)

    def __call__(self, x):
        x = (super().__call__(x) * 255.0).detach().cpu().numpy()
        transpose_axis = (0, 2, 3, 1) if len(x.shape) == 4 else (1, 2, 0)
        return np.clip(x, 0, 255).astype(np.uint8).transpose(transpose_axis)
