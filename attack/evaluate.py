import argparse
import os
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from dataloader import Denormalize, get_dataloader, Normalize, ToNumpy, get_generated_dataloader
from issba import encode_image
from model import get_model
from tools import (get_save_root, get_attack_num_class, SAVE_SIZE, DATASET, MODE, ATTACKER, init_seed, GENERATOR_PATH,
                   BADNET_TRIGGER, get_blended_root, NUM_CLASS)
from .attack import make_backdoor


def evaluate(opt):
    if os.path.isdir(opt.ckpt_path):
        root = opt.ckpt_path
        for file in sorted(os.listdir(root)):
            if file.endswith('.pth'):
                opt.ckpt_path = join(root, file)
                print('\n' + '=' * 45 + f' evaluate {file} ' + '=' * 45 + '\n')
                evaluate(opt)
    else:
        ckpt_path, device = opt.ckpt_path, opt.device
        ckpt = torch.load(ckpt_path, map_location=opt.device)
        opt = ckpt['opt']
        opt.ckpt_path, opt.device = ckpt_path, device
        if 'attacker' not in opt:
            evaluate_normal(opt, ckpt)
        elif opt.attacker == ATTACKER.generator:
            evaluate_generator(opt, ckpt)
        else:
            evaluate_compare(opt, ckpt)


def evaluate_generator(opt, *args):
    is_a_pth_file = not isinstance(args[0], nn.Module)
    loader = get_dataloader(opt, mode=MODE.test_backdoor)
    if is_a_pth_file:
        denorm = Denormalize(opt.data_name)
        G, C, _ = get_model(opt)
        G.load_state_dict(args[0]['G'])
        C.load_state_dict(args[0]['C'])

        run_dir = join(get_save_root(), 'evaluate')
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
    else:
        G, C = args

    # 千万别漏了！！！
    G.eval(), C.eval()
    total, corr_norm, corr_back, corr_cover = 0, 0, 0, 0
    run_tqdm = tqdm(loader)
    for i, (backdoor_label, X, y) in enumerate(run_tqdm):
        bd, total = len(X), total + len(X)
        backdoor_label, X, y = backdoor_label.to(opt.device), X.to(opt.device), y.to(opt.device)

        with torch.no_grad():
            y_pred = C(X)
            _, y_pred = torch.max(y_pred, dim=1)
            corr_norm += (y_pred == y).sum().item()

            source = torch.cat([X, X], dim=0) if not opt.no_cover else X
            target, gen_y = make_backdoor(y, backdoor_label, get_attack_num_class(opt), opt.no_cover)
            gen_x = G(source, target)
            gen_pred = C(gen_x)
            gen_pred = torch.argmax(gen_pred, dim=1)
            corr_back += (gen_pred[:bd] == gen_y[:bd]).sum().item()
            if not opt.no_cover:
                corr_cover += (gen_pred[bd:] == gen_y[bd:]).sum().item()
        acc_norm, acc_back, acc_cover = corr_norm / total, corr_back / total, corr_cover / total
        run_tqdm.set_description(
            'acc_norm {:.2%} acc_back {:.2%} acc_cover {:.2%}'.format(acc_norm, acc_back, acc_cover))
        if is_a_pth_file and i == 0:
            source, gen_x = denorm(source[:8]), denorm(gen_x[:8])
            source = F.interpolate(source, size=SAVE_SIZE)
            gen_x = F.interpolate(gen_x, size=SAVE_SIZE)
            if opt.data_name == DATASET.mnist:
                diff = source - gen_x
            else:
                diff = transforms.Grayscale(num_output_channels=3)(source - gen_x)
            images = torch.cat([source, gen_x, 20 * diff], dim=0)
            img_name = opt.ckpt_path.split('/')[-1][:-len('.pth')]
            # hb 10.23
            images = source, gen_x, diff
    if is_a_pth_file:
        # hb 10.23
        # save_image(images, join(run_dir, f'{img_name}_【{acc_norm * 100:.2f}_{acc_back * 100:.2f}】.png'), nrow=8)
        # f'_{acc_cover * 100:.2f}】.png'), nrow=8)
        for i, (a, b, c) in enumerate(zip(*images)):
            save_image(a, join(run_dir, f'a{i}.png'))
            save_image(b, join(run_dir, f'b{i}.png'))
            save_image(c * 8, join(run_dir, f'c{i}.png'))
    return acc_norm, acc_back


def evaluate_compare(opt, arg, loader=None):
    is_a_pth_file = not isinstance(arg, nn.Module)
    if loader is None:
        get_backdoor_loader = get_generated_dataloader if opt.attacker == ATTACKER.dataonly else get_dataloader
        normal_loader = get_dataloader(opt, mode=MODE.test_normal)
        backdoor_loader = get_backdoor_loader(opt, mode=MODE.test_backdoor)
    else:
        normal_loader, backdoor_loader = loader
    if is_a_pth_file:
        denorm = Denormalize(opt.data_name)
        _, C, _ = get_model(opt)
        C.load_state_dict(arg['C'])

        run_dir = join(get_save_root(), 'evaluate')
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
    else:
        C = arg

    # 千万别漏了！！！
    C.eval()
    total, corr_norm, corr_back = 0, 0, 0
    run_tqdm = tqdm(zip(normal_loader, backdoor_loader), total=len(normal_loader))
    for i, ((X_norm, y_norm), (y_back, X_back, _)) in enumerate(run_tqdm):
        total += len(X_norm)
        X_norm, y_norm = X_norm.to(opt.device), y_norm.to(opt.device)
        X_back, y_back = X_back.to(opt.device), y_back.to(opt.device)

        with torch.no_grad():
            y_pred = C(X_norm)
            _, y_pred = torch.max(y_pred, dim=1)
            corr_norm += (y_pred == y_norm).sum().item()

            y_pred = C(X_back)
            _, y_pred = torch.max(y_pred, dim=1)
            corr_back += (y_pred == y_back).sum().item()
        acc_norm, acc_back = corr_norm / total, corr_back / total
        run_tqdm.set_description('acc_norm {:.2%} acc_back {:.2%}'.format(acc_norm, acc_back))

        if is_a_pth_file and i == 0:
            choice = np.random.choice(len(X_back), 9, replace=False)
            images = F.interpolate(denorm(X_back[choice]), size=SAVE_SIZE)
            img_name = opt.ckpt_path.split('/')[-1][:-len('.pth')]
    if is_a_pth_file:
        save_image(images, join(run_dir, f'{img_name}_【{acc_norm * 100:.2f}_{acc_back * 100:.2f}】.png'), nrow=3)
    return acc_norm, acc_back


def evaluate_normal(opt, ckpt):
    loader = get_dataloader(opt, mode=MODE.test_normal)
    denorm = Denormalize(opt.data_name)
    _, C, _ = get_model(opt)
    C.load_state_dict(ckpt['C'])

    run_dir = join(get_save_root(), 'evaluate')
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    # 千万别漏了！！！
    C.eval()
    total, corr_norm = 0, 0
    run_tqdm = tqdm(loader, total=len(loader))
    for i, (X, y) in enumerate(run_tqdm):
        total += len(X)
        X, y = X.to(opt.device), y.to(opt.device)

        with torch.no_grad():
            y_pred = C(X)
            _, y_pred = torch.max(y_pred, dim=1)
            corr_norm += (y_pred == y).sum().item()
        acc_norm = corr_norm / total
        run_tqdm.set_description('acc_norm {:.2%}'.format(acc_norm))

        if i == 0:
            choice = np.random.choice(len(X), 9, replace=False)
            images = F.interpolate(denorm(X[choice]), size=SAVE_SIZE)
            img_name = opt.ckpt_path.split('/')[-1][:-len('.pth')]
    save_image(images, join(run_dir, f'{img_name}_【{acc_norm * 100:.2f}】.png'), nrow=3)
    return acc_norm


def evaluate_psnr(data_name, attacker):
    opt = argparse.Namespace()
    opt.seed = None
    init_seed(opt)
    opt.device = 'cuda:3'
    opt.data_name = data_name
    opt.attacker = attacker
    opt.attack_num = -1
    opt.batch_size = 128
    opt.ckpt_path = join(get_save_root(), GENERATOR_PATH(data_name))
    loader = get_dataloader(opt, mode=MODE.test_normal, no_normalize=True)
    norm, denorm = Normalize(opt.data_name), Denormalize(opt.data_name)
    to_numpy = ToNumpy(data_name)

    total_psnr_score, total = 0, 0
    for X, _ in tqdm(loader):
        X = norm(X.to(opt.device))
        if attacker == ATTACKER.generator:
            G = get_model(opt)[0]
            G.load_state_dict(torch.load(opt.ckpt_path)['G'])
            G.eval()
            img = G(X, F.one_hot(torch.full((len(X),), 0), NUM_CLASS[data_name]).to(opt.device))
            img = to_numpy(img).squeeze()
        else:
            img = to_numpy(X).squeeze()
            for i in range(len(X)):
                if attacker == ATTACKER.badnet:
                    img[i][getattr(BADNET_TRIGGER, data_name)[0]] = 255
                elif attacker == ATTACKER.blended:
                    tmp = Image.fromarray(img[i])
                    kitty = Image.open(join(get_blended_root(), f'{0:02d}.jpg'))
                    if data_name == DATASET.mnist:
                        kitty = transforms.Grayscale(1)(kitty)
                    kitty = kitty.resize(tmp.size, Image.BILINEAR)
                    tmp = Image.blend(tmp, kitty, 0.15)
                    img[i] = np.array(tmp)
                else:
                    img[i] = encode_image(img[i], str(0), opt.device)
        X = to_numpy(X).squeeze()
        for i in range(len(X)):
            psnr_score = psnr(X[i], img[i], data_range=255)
            if psnr_score == float('inf'):
                continue
            total_psnr_score += psnr_score
            total += 1
    print(f'psnr {data_name} {attacker}: {total_psnr_score / total:.2f}')


if __name__ == '__main__':
    pass
