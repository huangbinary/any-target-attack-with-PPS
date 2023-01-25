from os.path import join

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from attack import make_backdoor
from dataloader import ToNumpy, get_dataloader, ToTensor
from model import get_model
from tools import get_rundir_board_logger, MODE, logger_message, get_attack_num_class, get_save_root
from .plot import plot_strip_entropy_histogram


def test_strip(opt):
    ckpt = torch.load(opt.ckpt_path, map_location=opt.device)
    opt.data_name = ckpt['opt'].data_name
    # 是为了加载数据时判断添加什么触发器
    opt.attacker = ckpt['opt'].attacker
    opt.attack_num = ckpt['opt'].attack_num
    opt.attack_label = ckpt['opt'].attack_label
    run_dir, _ = get_rundir_board_logger(opt, tensorboard=False)

    for rounds in range(1, 1 + opt.sp_rounds):
        print('=' * 50 + f' round {rounds} ' + '=' * 50)
        entropy_benign, entropy_trojan = calcu_entropy(opt, ckpt)

        save_file = join(run_dir, f'entropy_{rounds}.txt')
        with open(save_file, 'w') as logger:
            logger_message(logger, f'【entropy_benign】\n{str(entropy_benign)}'
                           + f'\n【entropy_trojan】\n{str(entropy_trojan)}\n')
            show_message = f'【entropy_benign】min: {min(entropy_benign)}, max: {max(entropy_benign)}\n'
            show_message += f'【entropy_trojan】min: {min(entropy_trojan)}, max: {max(entropy_trojan)}\n'

            # todo 应该设置合适的FRR再计算出detection_boundary，这里FRR直接就设置为0了
            far = sum(entropy > min(entropy_benign) for entropy in entropy_trojan) / opt.num_background
            show_message += f'FAR: {far:.2%}\n'
            if far > opt.far_threshold:
                show_message += 'this is a clean model'
            else:
                show_message += 'this is a backdoor model'
            logger_message(logger, show_message, is_print=True)
        plot_strip_entropy_histogram(join(get_save_root(), opt.data_name, run_dir), rounds)


def calcu_entropy(opt, ckpt):
    G, C, _ = get_model(opt)
    C.load_state_dict(ckpt['C'])
    C.requires_grad_(False)
    C.eval()

    norm_loader = get_dataloader(opt, mode=MODE.test_normal, num_sample=opt.num_background)
    back_loader = get_dataloader(opt, mode=MODE.test_backdoor, num_sample=opt.num_background)
    super_loader = get_dataloader(opt, mode=MODE.test_normal, num_sample=opt.num_background)
    to_numpy, to_tensor = ToNumpy(opt.data_name), ToTensor(opt.data_name)

    back_y, back_X, y = next(iter(back_loader))
    back_X, back_y, y = back_X.to(opt.device), back_y.to(opt.device), y.to(opt.device)
    if 'G' in ckpt:
        G.load_state_dict(ckpt['G'])
        G.requires_grad_(False)
        G.eval()
        target, _ = make_backdoor(y, back_y, get_attack_num_class(opt), no_cover=True)
        back_X = G(back_X, target)
    back_X = to_numpy(back_X)
    norm_X = to_numpy(next(iter(norm_loader))[0])
    super_X = to_numpy(next(iter(super_loader))[0])

    entropy_benign, entropy_trojan = [], []
    for X, entropy, desc in [(norm_X, entropy_benign, 'benign'), (back_X, entropy_trojan, 'trojan')]:
        for background in tqdm(X, desc=desc):
            index = np.random.randint(0, len(super_X), size=opt.num_superposition)
            X = []
            for i in index:
                img = cv2.addWeighted(background, 1, super_X[i], 1, 0)
                X.append(to_tensor(img))
            y = F.softmax(C(torch.stack(X).to(opt.device)), dim=1).cpu()
            entropy.append(-np.nansum(y * np.log2(y)) / opt.num_superposition)
    return entropy_benign, entropy_trojan


if __name__ == "__main__":
    pass
