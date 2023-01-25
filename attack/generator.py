import argparse
import os
import shutil
from os.path import join

import torch
from PIL import Image
from tqdm import tqdm

from dataloader import get_dataloader, ToNumpy, Normalize
from model import get_model
from tools import (ATTACKER, MODE, init_seed, get_attack_num_class, get_dataset_root, get_save_root, DATASET,
                   GENERATOR_PATH)
from .attack import make_backdoor


def generate_images(data_name, is_train, no_cover=True):
    """
    对于训练数据集，正常样本与生成的样本放在一起组成训练集
    对于测试数据集，全部的生成样本组成测试ASR的测试集
    """
    opt = argparse.Namespace()
    opt.seed = None
    init_seed(opt)
    opt.device = 'cuda:2'
    opt.data_name = data_name
    opt.attacker = ATTACKER.generator
    opt.ckpt_path = join(get_save_root(), GENERATOR_PATH(data_name))
    opt.no_cover = no_cover
    cover_str = '' if no_cover else '_cover'
    no_cover = no_cover or is_train is False
    opt.batch_size = 128
    opt.attack_num = -1
    opt.ratio = 0.1

    G = get_model(opt)[0]
    G.load_state_dict(torch.load(opt.ckpt_path)['G'])
    G.eval()
    loader = get_dataloader(opt, mode=MODE.train_backdoor if is_train else MODE.test_backdoor, no_normalize=True)
    norm, to_numpy = Normalize(opt.data_name), ToNumpy(opt.data_name)

    root = join(get_dataset_root(), data_name, 'generated_images', ('train' if is_train else 'test') + cover_str)
    if os.path.exists(root):
        shutil.rmtree(root)
    idx = 0
    for backdoor_label, X, y in tqdm(loader):
        X, y, backdoor_label = X.to(opt.device), y.to(opt.device), backdoor_label.to(opt.device)
        X = norm(X)
        is_backdoor = backdoor_label != -1
        with torch.no_grad():
            source = torch.cat([X[is_backdoor], X[is_backdoor]], dim=0) if not no_cover else X[is_backdoor]
            target, gen_y = make_backdoor(y[is_backdoor], backdoor_label[is_backdoor],
                                          get_attack_num_class(opt), no_cover)
            gen_x = G(source, target)
            if no_cover:
                bd_x, bd_y, bd_backdoor_label = gen_x, y[is_backdoor], gen_y
            else:
                bd = is_backdoor.sum().item()
                bd_x, bd_y, bd_backdoor_label = gen_x[:bd], y[is_backdoor], gen_y[:bd]
                cover_x, cover_y, cover_backdoor_label = gen_x[bd:], y[is_backdoor], torch.full((bd,), -1)
            X, y, backdoor_label = X[~ is_backdoor], y[~ is_backdoor], backdoor_label[~ is_backdoor]
        datasets = [(X, y, backdoor_label), (bd_x, bd_y, bd_backdoor_label)]
        if not no_cover:
            datasets.append((cover_x, cover_y, cover_backdoor_label))
        for cur_X, cur_y, cur_backdoor_label in datasets:
            cur_X = to_numpy(cur_X).squeeze()
            for img, label, target in zip(cur_X, cur_y, cur_backdoor_label):
                label_root = join(root, str(label.item()), str(target.item()))
                if not os.path.exists(label_root):
                    os.makedirs(label_root)
                img = Image.fromarray(img)
                img.save(join(label_root, f'{idx:06d}.jpg'))
                idx += 1
    print(idx)


def generate():
    for data_name in [DATASET.cifar10]:
        generate_images(data_name, is_train=True, no_cover=False)
        generate_images(data_name, is_train=False, no_cover=False)
