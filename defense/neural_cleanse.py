from os.path import join

import torch
from torch import optim, nn
from tqdm import tqdm

from dataloader import get_dataloader
from tools import pretty, logger_message, get_rundir_board_logger, MODE, NUM_CLASS
from .nc_common import Param, MaskPattern, Adjuster, BestResult


def train_neural_cleanse(opt):
    ckpt = torch.load(opt.ckpt_path, map_location=opt.device)
    opt.data_name = ckpt['opt'].data_name
    run_dir, log_file = get_rundir_board_logger(opt, tensorboard=False)
    Param.run_dir = run_dir
    with open(log_file, 'w') as logger:
        Param.logger = logger
        for rounds in range(1, 1 + opt.nc_rounds):
            Param.rounds = rounds
            masks = []
            for check_label in range(NUM_CLASS[opt.data_name]):
                Param.check_label = check_label
                logger_message(logger, '=' * 50 + f' check label: [{rounds}]-{check_label} ' + '=' * 50, is_print=True)
                train_per_label(opt, ckpt['C'])
                masks.append(BestResult.mask)
            logger_message(logger, '\n', is_print=True)
            norm_list = torch.stack([torch.sum(torch.abs(m)) for m in masks])
            # 检查的只是mask的L1范数
            outlier_detection(opt, norm_list)


def train_per_label(opt, state_dict):
    loader = get_dataloader(opt, mode=MODE.test_normal)
    NC = MaskPattern(opt, state_dict).to(opt.device)
    CE = nn.CrossEntropyLoss()
    optimizer = optim.Adam(NC.parameters(), lr=opt.lr, betas=(0.5, 0.9))

    for epoch in range(1, 1 + opt.epochs):
        total, corr, loss_ce_tot, loss_norm_tot = 0, 0, 0, 0
        desc = 'epoch {} - acc {:.2%} loss_ce {:.6f} loss_norm {:.6f}'
        run_tqdm = tqdm(loader)
        for X, _ in run_tqdm:
            bs, total = len(X), total + len(X)
            X = X.to(opt.device)
            y = torch.full((bs,), Param.check_label).to(opt.device)
            y_pred = NC(X)

            loss_ce = CE(y_pred, y)
            loss_norm = torch.norm(NC.get_mask(), 2)
            loss = loss_ce + Adjuster.alpha * loss_norm
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_ce_tot += loss_ce * bs
            loss_norm_tot += loss_norm * bs
            corr += torch.sum(torch.argmax(y_pred, dim=1) == y).detach()
            run_tqdm.set_description(desc.format(epoch, corr / total, loss_ce, loss_norm))

        acc, loss_ce_avg, loss_norm_avg = corr / total, loss_ce_tot / total, loss_norm_tot / total
        # 达到目标ASR要求，且触发器更小
        if acc >= opt.asr_threshold and loss_norm_avg < BestResult.loss_norm:
            BestResult.update(NC, loss_norm_avg, epoch)
        show_message = (desc + ' loss_norm_best {:.6f}').format(epoch, acc, loss_ce_avg, loss_norm_avg,
                                                                BestResult.loss_norm)
        logger_message(Param.logger, show_message, is_print=True)

        if Adjuster.check_early_stop(opt, acc):
            logger_message(Param.logger, "===> Early_stop !!!", is_print=True)
            break


def outlier_detection(opt, norm_list):
    save_file = join(Param.run_dir, 'outlier.txt')
    with open(save_file, 'a') as logger:
        print('===> outlier detection begin...')
        median = torch.median(norm_list)
        MAD = torch.median(torch.abs(norm_list - median)) * 1.4826
        anomaly_index = (norm_list - median) / MAD

        num_class = NUM_CLASS[opt.data_name]
        # 总宽度为9
        print('   '.join(['label', ' norm_list ', 'anomaly_index']))
        for label in range(num_class):
            print('   '.join([pretty(label, len('label')),
                              pretty(f'{norm_list[label].item():.5f}', len(' norm_list ')),
                              pretty(f'{anomaly_index[label].item():.5f}', len('anomaly_index'))]))
        logger_message(logger, '【  norm_list  】\n' + str(norm_list.tolist()))
        logger_message(logger, '【anomaly_index】\n' + str(anomaly_index.tolist()))

        backdoor_list = []
        for label in range(num_class):
            if anomaly_index[label] < -2:
                backdoor_list.append(label)
        if backdoor_list:
            logger_message(logger, f'backdoor label found: {backdoor_list}\n', is_print=True)
        else:
            logger_message(logger, 'backdoor label not found, the model clean\n', is_print=True)


if __name__ == "__main__":
    pass
