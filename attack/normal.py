import os
from os.path import join

import torch
from torch import optim, nn
from tqdm import tqdm

from dataloader import get_dataloader
from model import get_model
from tools import get_rundir_board_logger, logger_message, MODE, LOWEST_ACC, ATTACKER, DATASET
from .evaluate import evaluate_generator, evaluate_compare


def train_common(opt, train_loader, test_loader, C, CE, optimizer, scheduler,
                 G=None, is_fine_tune=False, is_generator=False):
    if is_fine_tune:
        evaluate_generator(opt, G, C) if is_generator else evaluate_compare(opt, C)
    run_dir, board, log_file = get_rundir_board_logger(opt)
    best_acc_norm = 0
    with open(log_file, 'w') as logger:
        for epoch in range(1, 1 + opt.epochs):
            # =====================================train=====================================
            C.train()
            total, corr = 0, 0
            loss_tot = 0

            desc = 'train - epoch {:d} loss {:.6f} acc {:.2%}'
            run_tqdm = tqdm(train_loader)
            for X, y in run_tqdm:
                bs, total = len(X), total + len(X)
                X, y = X.to(opt.device), y.to(opt.device)

                y_pred = C(X)
                loss = CE(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_tot += loss * bs
                corr += (torch.argmax(y_pred, dim=1) == y).sum().item()
                show_message = desc.format(epoch, loss, corr / total)
                run_tqdm.set_description(show_message)
            scheduler.step()
            logger_message(logger, show_message)
            board.add_scalars('loss', {'train': loss_tot / total}, epoch)
            board.add_scalars('acc', {'train': corr / total * 100}, epoch)

            if is_fine_tune:
                evaluate_generator(opt, G, C) if is_generator else evaluate_compare(opt, C)
            else:
                # =====================================test=====================================
                C.eval()
                total, corr = 0, 0
                loss_tot = 0

                desc = ' test - epoch {:d} loss {:.6f} acc {:.2%}'
                run_tqdm = tqdm(test_loader)
                for i, (X, y) in enumerate(run_tqdm):
                    bs, total = len(X), total + len(X)
                    X, y = X.to(opt.device), y.to(opt.device)

                    with torch.no_grad():
                        y_pred = C(X)
                        loss = CE(y_pred, y)
                        loss_tot += loss * bs
                    corr += (torch.argmax(y_pred, dim=1) == y).sum().item()

                    show_message = desc.format(epoch, loss, corr / total)
                    run_tqdm.set_description(show_message)
                logger_message(logger, show_message)
                board.add_scalars('loss', {'test': loss_tot / total}, epoch)
                board.add_scalars('acc', {'test': corr / total * 100}, epoch)

                # ===================================checkpoint===================================
                acc_norm = corr / total * 100
                if acc_norm > max(best_acc_norm, LOWEST_ACC[opt.data_name]):
                    ckpt = {
                        'C': C.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'opt': opt,
                        'epoch': epoch,
                        'best_acc_norm': best_acc_norm,
                    }
                    ckpt_dir = join(run_dir, 'ckpt')
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    best_acc_norm = acc_norm
                    ckpt_file = join(ckpt_dir, f'benign_model【{epoch}_{best_acc_norm:.2f}】.pth')
                    torch.save(ckpt, ckpt_file)


def train_normal(opt, ckpt=None):
    is_fine_tune = ckpt is not None
    train_loader = get_dataloader(opt, mode=MODE.train_normal)
    test_loader = get_dataloader(opt, mode=MODE.test_normal)

    G, C, _ = get_model(opt)
    is_generator = False
    if is_fine_tune:
        is_generator = opt.attacker == ATTACKER.generator
        C.load_state_dict(ckpt['C'])
        if is_generator:
            G.load_state_dict(ckpt['G'])
    CE = nn.CrossEntropyLoss()
    optimizer = optim.SGD(C.parameters(), lr=opt.lr, weight_decay=5e-4, momentum=0.9)
    if opt.data_name in [DATASET.celeba, DATASET.imagenet]:
        optimizer = optim.Adam(C.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, opt.scheduler, 0.1)

    train_common(opt, train_loader, test_loader, C, CE, optimizer, scheduler, G, is_fine_tune, is_generator)


if __name__ == '__main__':
    pass
