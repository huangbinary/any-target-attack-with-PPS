import os
from os.path import join

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from issba import encode_image
from tools import (get_blended_root, get_attack_num_class, IMG_SIZE, NUM_CHANNEL, MODE, ATTACKER, NUM_TOTAL,
                   BADNET_TRIGGER, get_dataset_root, DATASET)
from .common import get_transform


class CommonDataset(Dataset):
    """
    对于badnet和blended，先添加触发器再进行transform；对于generator，只进行transform，但都返回目标标签
    对于generator的cover样本，使用相同的后门样本，但是添加cover的target条件
    """

    def __init__(self, opt, mode: MODE, num_sample, no_normalize):
        """
        :param opt:
        :param mode: train/test确定选择哪个数据集，normal/backdoor确定是否添加后门
        :param num_sample: 在一些防御方法只需加载少量数据时指定
        :param no_normalize: 在计算数据集的均值和标准差时无需标准化（也不可能标准化）
        """
        super(CommonDataset, self).__init__()
        # 指定num_sample的话也是加载test数据集
        assert mode in MODE.test or num_sample is None
        if 'attacker' in opt:
            assert not (opt.attack_num > 1 and opt.data_name != DATASET.imagenet), 'only imagenet supports multi-target'
        self.opt = opt
        self.mode = mode
        self.backdoor_info = self.__get_backdoor_info()
        self.is_train = mode in MODE.train and num_sample is None
        self.img_size = IMG_SIZE[opt.data_name]
        self.num_channel = NUM_CHANNEL[self.opt.data_name]
        # 具体使用不同数据集的加载方法
        self.images, self.labels = self.load_data()
        # 在一些防御方法只需加载少量数据
        if num_sample is not None:
            random_index = np.random.choice(len(self.images), num_sample, replace=False)
            self.images, self.labels = self.images[random_index], self.labels[random_index]
        self.num_total = len(self.images)
        self.backdoor_label = np.full((self.num_total,), -1)
        self.__make_backdoor_dataset()
        self.transform = get_transform(opt.data_name, is_train=self.is_train, no_normalize=no_normalize)

    def __getitem__(self, idx):
        img = self.images[idx]
        # 转化为tensor之前需要先转化为Image类型
        img = self.transform(Image.fromarray(img))
        if self.mode in MODE.backdoor:
            return self.backdoor_label[idx], img, self.labels[idx]
        return img, self.labels[idx]

    def load_data(self):
        raise NotImplementedError

    def __get_backdoor_info(self):
        # 事先选择好后门样本的信息
        backdoor_info = {}
        if self.mode in MODE.backdoor:
            total = NUM_TOTAL.train if self.mode in MODE.train else NUM_TOTAL.test
            # 选择后门样本的id
            if self.mode == MODE.test_backdoor:
                random_id = np.arange(total[self.opt.data_name])
            else:
                num_backdoor = int(total[self.opt.data_name] * self.opt.ratio)
                random_id = np.random.choice(total[self.opt.data_name], num_backdoor, replace=False)
            # 后门样本对应的攻击类别
            for idx in random_id:
                if self.opt.attack_num == 1:
                    backdoor_info[idx] = self.opt.attack_label
                else:
                    backdoor_info[idx] = np.random.choice(get_attack_num_class(self.opt))
        return backdoor_info

    def __make_backdoor_dataset(self):
        # load_data返回的统一是numpy格式，要么是图片路径，要么是加载好的图片（通道在后）
        images = []
        for i in tqdm(range(self.num_total), 'loding data'):
            # 若保存的是图片路径则需要打开
            images.append(self.__open(self.images[i]))
            if i in self.backdoor_info:
                self.backdoor_label[i] = self.backdoor_info[i]
                images[i] = self.__add_trigger(images[i], self.backdoor_label[i])
        self.images = np.array(images)

    def __open(self, img):
        if not isinstance(img, str):
            return img
        img = np.array(Image.open(img).resize((self.img_size, self.img_size), Image.BILINEAR))
        # ImageNet中有灰度图。。。
        if len(img.shape) == 2 and self.num_channel == 3:
            img = np.expand_dims(img, axis=-1).repeat(3, axis=-1)
        return img

    def __add_trigger(self, img, backdoor_label):
        if self.opt.attacker == ATTACKER.badnet:
            img[getattr(BADNET_TRIGGER, self.opt.data_name)[backdoor_label]] = 255
        elif self.opt.attacker == ATTACKER.blended:
            img = Image.fromarray(img)
            kitty = Image.open(join(get_blended_root(), f'{backdoor_label:02d}.jpg'))
            if self.num_channel == 1:
                kitty = transforms.Grayscale(1)(kitty)
            kitty = kitty.resize(img.size, Image.BILINEAR)
            img = Image.blend(img, kitty, 0.15)
            img = np.array(img)
        elif self.opt.attacker == ATTACKER.issba:
            img = encode_image(img, str(backdoor_label), self.opt.device)
        return img

    def __len__(self):
        return self.num_total


class GeneratedDataset(Dataset):
    def __init__(self, opt, mode):
        super(GeneratedDataset, self).__init__()
        self.opt = opt
        self.mode = mode
        self.backdoor_labels, self.images, self.labels = self.__load_data()
        self.transform = get_transform(opt.data_name, is_train=mode in MODE.train)

    def __getitem__(self, idx):
        img = self.images[idx]
        # 转化为tensor之前需要先转化为Image类型
        img = self.transform(Image.fromarray(img))
        return self.backdoor_labels[idx], img, self.labels[idx]

    def __load_data(self):
        cover_str = '' if self.opt.no_cover else '_cover'
        root = join(get_dataset_root(), self.opt.data_name, 'generated_images',
                    ('train' if self.mode in MODE.train else 'test') + cover_str)
        images, labels, backdoor_labels = [], [], []
        for label_root in os.scandir(root):
            for target_root in os.scandir(label_root.path):
                for img_file in os.scandir(target_root.path):
                    images.append(np.array(Image.open(img_file.path)))
                    labels.append(int(label_root.name))
                    backdoor_labels.append(int(target_root.name))
        return np.array(backdoor_labels), np.array(images), np.array(labels)

    def __len__(self):
        return len(self.images)
