import argparse
import copy
from os.path import join

import numpy as np
import torch
from torch import nn

from attack import evaluate_generator, evaluate_compare, train_normal
from dataloader import get_dataloader
from model import get_model
from tools import logger_message, get_rundir_board_logger, init_seed, get_save_root, ATTACKER, MODE, NUM_CLASS, DATASET
from .nc_common import Param
from .plot import plot_pruning_acc_line


def test_fine_pruning(opt):
    ckpt = torch.load(opt.ckpt_path, map_location=opt.device)
    opt.data_name = ckpt['opt'].data_name
    opt.batch_size = ckpt['opt'].batch_size
    opt.attacker = ckpt['opt'].attacker if 'attacker' in ckpt['opt'] else None
    opt.attack_num = ckpt['opt'].attack_num
    opt.attack_label = ckpt['opt'].attack_label
    opt.no_cover = True
    run_dir, log_file = get_rundir_board_logger(opt, tensorboard=False)

    G, C, _ = get_model(opt)
    C.load_state_dict(ckpt['C'])
    C.requires_grad_(False)
    C.eval()
    if 'G' in ckpt:
        G.load_state_dict(ckpt['G'])
        G.requires_grad_(False)
        G.eval()
    normal_loader = get_dataloader(opt, mode=MODE.test_normal)
    backdoor_loader = get_dataloader(opt, mode=MODE.test_backdoor)

    activation = []
    pruned_layer = C.relu6 if opt.data_name == DATASET.mnist else C.layer4
    hook = pruned_layer.register_forward_hook(lambda module, input_, output: activation.append(output))
    for X, _ in normal_loader:
        C(X.to(opt.device))
    hook.remove()
    # 计算每个通道的平均激活值
    activation = torch.mean(torch.cat(activation, dim=0), dim=[0, 2, 3])
    arg_sort = torch.argsort(activation)
    num_channel = arg_sort.shape[0]
    channel_mask = torch.ones(num_channel, dtype=bool)

    acc_list = []
    with open(log_file, 'w')as logger:
        Param.logger = logger
        logger_message(logger, f'total {num_channel} channels...', is_print=True)
        for num_pruned in range(num_channel):
            C_pruned = copy.deepcopy(C)
            if num_pruned:
                channel_mask[arg_sort[num_pruned - 1]] = False
                prune_channel(opt.data_name, C_pruned, C, num_channel, num_pruned, channel_mask)
            acc_list.append(evaluate_pruned_model(opt, G, C_pruned, num_pruned, (normal_loader, backdoor_loader)))
    save_file = join(run_dir, f'accuracy.txt')
    with open(save_file, 'w')as logger:
        acc_list = np.array(acc_list).transpose(1, 0).tolist()
        logger_message(logger, f'【acc_norm_list】\n{str(acc_list[0])}\n【acc_back_list】\n{str(acc_list[1])}')
    plot_pruning_acc_line(join(get_save_root(), opt.data_name, run_dir))


def prune_channel(data_name, C_pruned, C, num_channel, num_pruned, channel_mask):
    if data_name == DATASET.mnist:
        C_pruned.conv5 = nn.Conv2d(32, num_channel - num_pruned, kernel_size=5, stride=1)
        C_pruned.conv5.weight.data = C.conv5.weight.data[channel_mask]
        C_pruned.conv5.bias.data = C.conv5.bias.data[channel_mask]
        in_channel = 16 * (num_channel - num_pruned)
        C_pruned.fc6 = nn.Linear(in_channel, 512)
        C_pruned.fc6.weight.data = C.fc6.weight.data.reshape(-1, 64, 16)[:, channel_mask].reshape(-1, in_channel)
        C_pruned.fc6.bias.data = C.fc6.bias.data
    else:
        C_pruned.layer4[1].conv2 = nn.Conv2d(num_channel, num_channel - num_pruned, kernel_size=3, stride=1,
                                             padding=1, bias=False)
        C_pruned.layer4[1].conv2.weight.data = C.layer4[1].conv2.weight.data[channel_mask]
        C_pruned.layer4[1].index = channel_mask
        C_pruned.fc = nn.Linear(num_channel - num_pruned, NUM_CLASS[data_name])
        C_pruned.fc.weight.data = C.fc.weight.data[:, channel_mask]
        C_pruned.fc.bias.data = C.fc.bias.data


def evaluate_pruned_model(opt, G, C, num_pruned, loader):
    if opt.attacker == ATTACKER.generator:
        acc_norm, acc_back = evaluate_generator(opt, G, C)
    else:
        acc_norm, acc_back = evaluate_compare(opt, C, loader)
    logger_message(Param.logger, f'after {num_pruned} channel pruned: acc_norm {acc_norm:.2%}, acc_back {acc_back:.2%}',
                   is_print=True)
    return acc_norm, acc_back


def test_fine_tune(opt=None):
    if opt is None:
        opt = argparse.Namespace()
        opt.seed = None
        init_seed(opt)
        opt.ckpt_path = join(get_save_root(), 'result/attack_r0.1/cifar10/cifar10_badnet.pth')
        opt.run_name = 'tmp'
        opt.device = 'cuda:0'

    run_name, device = opt.run_name, opt.device
    ckpt = torch.load(opt.ckpt_path)
    opt = ckpt['opt']
    opt.run_name, opt.device = run_name, device
    opt.lr, opt.scheduler = opt.lr_C, opt.scheduler_C
    train_normal(opt, ckpt)


if __name__ == "__main__":
    pass
