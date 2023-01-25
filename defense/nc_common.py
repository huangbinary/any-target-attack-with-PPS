import os
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.utils import save_image

from dataloader import Normalize
from model import get_model
from tools import logger_message, IMG_SIZE, NUM_CHANNEL, SAVE_SIZE


class Param:
    run_dir = None
    logger = None
    check_label = None
    rounds = None
    epoch = None


class MaskPattern(nn.Module):
    def __init__(self, opt, state_dict):
        super(MaskPattern, self).__init__()
        data_name = opt.data_name
        if opt.nc_rounds == 1:
            init_mask = np.ones((1, IMG_SIZE[data_name], IMG_SIZE[data_name]))
            init_pattern = np.ones((NUM_CHANNEL[data_name], IMG_SIZE[data_name], IMG_SIZE[data_name]))
        else:
            init_mask = np.random.randn(1, IMG_SIZE[data_name], IMG_SIZE[data_name])
            init_pattern = np.random.randn(NUM_CHANNEL[data_name], IMG_SIZE[data_name], IMG_SIZE[data_name])
        self.mask = nn.Parameter(torch.tensor(init_mask.astype(np.float32)))
        self.pattern = nn.Parameter(torch.tensor(init_pattern.astype(np.float32)))
        self.normalizer = Normalize(data_name)
        self.C = self._get_C(opt, state_dict)
        BestResult.init()
        Adjuster.init(opt)

    def forward(self, x):
        mask = self.get_mask()
        pattern = self.normalizer(self.get_pattern())
        x = (1 - mask) * x + mask * pattern
        return self.C(x)

    def get_mask(self):
        return nn.Tanh()(self.mask) / 2 + 0.5

    def get_pattern(self):
        return nn.Tanh()(self.pattern) / 2 + 0.5

    def _get_C(self, opt, state_dict):
        C = get_model(opt)[1]
        C.load_state_dict(state_dict)
        for param in C.parameters():
            param.requires_grad = False
        # 设置为测试模式
        C.eval()
        return C


class BestResult:
    mask = None
    pattern = None
    loss_norm = float('inf')

    @staticmethod
    def init():
        BestResult.mask = None
        BestResult.pattern = None
        BestResult.loss_norm = float('inf')
        logger_message(Param.logger, f'===> init BestResult', is_print=True)

    @staticmethod
    def update(NC, loss_norm, epoch):
        BestResult.mask = NC.get_mask().detach()
        BestResult.pattern = NC.get_pattern().detach()
        BestResult.loss_norm = loss_norm
        BestResult.__save_result(epoch)

    @staticmethod
    def __save_result(epoch):
        trigger_dir = join(Param.run_dir, 'trigger', f'{Param.check_label:02d}')
        if not os.path.exists(trigger_dir):
            os.makedirs(trigger_dir)
        pattern = F.interpolate(BestResult.pattern.unsqueeze(0), size=SAVE_SIZE)
        mask = BestResult.mask.unsqueeze(0).repeat_interleave(pattern.shape[1], dim=1)
        mask = F.interpolate(mask, size=SAVE_SIZE)
        trigger = F.interpolate((BestResult.mask * BestResult.pattern).unsqueeze(0), size=SAVE_SIZE)
        images = torch.cat([mask, pattern, trigger], dim=0)
        save_image(images, join(trigger_dir, f'round_{Param.rounds}_epoch_{epoch:03d}.png'))
        torch.save({'mask_pattern': BestResult}, join(trigger_dir, 'best_result.pth'))


class Adjuster1:
    alpha = None
    __alpha_up_counter = None
    __alpha_down_counter = None
    __alpha_up_flag = None
    __alpha_down_flag = None

    __early_stop_counter = None
    __loss_norm = None

    @staticmethod
    def init(opt):
        # 要重置所有参数！！！
        Adjuster1.alpha = opt.init_alpha  # 1e-2
        Adjuster1.__alpha_up_counter = 0
        Adjuster1.__alpha_down_counter = 0
        Adjuster1.__alpha_up_flag = False
        Adjuster1.__alpha_down_flag = False
        Adjuster1.__early_stop_counter = 0
        Adjuster1.__loss_norm = float('inf')
        logger_message(Param.logger, f'===> init Adjuster', is_print=True)

    @staticmethod
    def check_early_stop(opt, acc):
        if acc >= opt.asr_threshold:
            Adjuster1.__alpha_up_counter += 1
            Adjuster1.__alpha_down_counter = 0
        else:
            Adjuster1.__alpha_up_counter = 0
            Adjuster1.__alpha_down_counter += 1

        pre_alpha = Adjuster1.alpha
        if Adjuster1.__alpha_up_counter >= opt.alpha_change_max_counter:
            Adjuster1.alpha *= 2
            logger_message(Param.logger, f'===> up alpha from {pre_alpha} to {Adjuster1.alpha}', is_print=True)
            Adjuster1.__alpha_up_counter = 0
            Adjuster1.__alpha_up_flag = True
        elif Adjuster1.__alpha_down_counter >= opt.alpha_change_max_counter:
            Adjuster1.alpha /= (2 ** 1.5)
            logger_message(Param.logger, f'===> down alpha from {pre_alpha} to {Adjuster1.alpha}', is_print=True)
            Adjuster1.__alpha_down_counter = 0
            Adjuster1.__alpha_down_flag = True

        early_stop = False
        # loss没有下降或下降不明显则需要增加counter
        if BestResult.loss_norm >= Adjuster1.__loss_norm * opt.early_stop_threshold:
            Adjuster1.__early_stop_counter += 1
        else:
            Adjuster1.__early_stop_counter = 0
        Adjuster1.__loss_norm = BestResult.loss_norm
        if (Adjuster1.__alpha_down_flag and Adjuster1.__alpha_up_flag
                and Adjuster1.__early_stop_counter >= opt.early_stop_max_counter):
            early_stop = True
        return early_stop


class Adjuster2:
    # 用二分查找的方式调整alpha以及判断early_stop
    alpha = None
    __alpha_up = None
    __alpha_down = None
    __alpha_up_counter = None
    __alpha_down_counter = None
    __alpha_up_flag = None
    __alpha_down_flag = None

    __early_stop_counter = None
    __loss_norm = None

    @staticmethod
    def init(opt):
        # 要重置所有参数！！！
        Adjuster2.alpha = opt.init_alpha  # 1e-2
        Adjuster2.__alpha_up = None
        Adjuster2.__alpha_down = None
        Adjuster2.__alpha_up_counter = 0
        Adjuster2.__alpha_down_counter = 0
        Adjuster2.__alpha_up_flag = False
        Adjuster2.__alpha_down_flag = False
        Adjuster2.__early_stop_counter = 0
        Adjuster2.__loss_norm = float('inf')
        logger_message(Param.logger, f'===> init Adjuster', is_print=True)

    @staticmethod
    def _adjust_alpha(opt, acc):
        if acc >= opt.asr_threshold:
            Adjuster2.__alpha_up_counter += 1
            Adjuster2.__alpha_down_counter = 0
        else:
            Adjuster2.__alpha_up_counter = 0
            Adjuster2.__alpha_down_counter += 1

        pre_alpha = Adjuster2.alpha
        logger_adjustment = lambda direct: logger_message(
            Param.logger, f'===> {direct} alpha from {pre_alpha} to {Adjuster2.alpha}, '
                          f'search range [{Adjuster2.__alpha_down}, {Adjuster2.__alpha_up}]', is_print=True)
        if Adjuster2.__alpha_up_counter >= opt.alpha_change_max_counter:
            Adjuster2.__alpha_down = Adjuster2.alpha
            if Adjuster2.__alpha_up is None:
                Adjuster2.alpha *= 2
            else:
                Adjuster2.alpha = (Adjuster2.alpha + Adjuster2.__alpha_up) / 2
            logger_adjustment('up')
            Adjuster2.__alpha_up_flag = True
            Adjuster2.__alpha_up_counter = 0
            # 置为-1使得下面不管怎么样都会更新为0
            Adjuster2.__early_stop_counter = -1
        elif Adjuster2.__alpha_down_counter >= opt.alpha_change_max_counter:
            Adjuster2.__alpha_up = Adjuster2.alpha
            if Adjuster2.__alpha_down is None:
                Adjuster2.alpha /= 2
            else:
                Adjuster2.alpha = (Adjuster2.alpha + Adjuster2.__alpha_down) / 2
            logger_adjustment('down')
            Adjuster2.__alpha_down_flag = True
            Adjuster2.__alpha_down_counter = 0
            Adjuster2.__early_stop_counter = -1

    @staticmethod
    def check_early_stop(opt, acc):
        if not (Adjuster2.__alpha_up and Adjuster2.__alpha_down) or (
                Adjuster2.__alpha_up - Adjuster2.__alpha_down) > opt.search_range_threshold:
            Adjuster2._adjust_alpha(opt, acc)

        # loss没有下降或下降不明显则需要增加counter
        if BestResult.loss_norm >= Adjuster2.__loss_norm * opt.early_stop_threshold:
            Adjuster2.__early_stop_counter += 1
        else:
            Adjuster2.__early_stop_counter = 0
        Adjuster2.__loss_norm = BestResult.loss_norm

        if (Adjuster2.__alpha_down_flag and Adjuster2.__alpha_up_flag
                and Adjuster2.__early_stop_counter >= opt.early_stop_max_counter):
            return True
        return False


Adjuster = Adjuster2

if __name__ == "__main__":
    pass
