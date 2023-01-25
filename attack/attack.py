import os
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim, nn
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from dataloader import get_dataloader, Denormalize, get_generated_dataloader
from model import get_model
from tools import (get_rundir_board_logger, get_attack_num_class, logger_message, MODE, SAVE_SIZE, DATASET, ATTACKER,
                   LOWEST_ACC)


def train_generator(opt):
    """
    需要判断是否不使用判别器D——no_D，是否不使用cover样本——no_cover
    """
    train_loader = get_dataloader(opt, mode=MODE.train_backdoor)
    normal_loader = get_dataloader(opt, mode=MODE.test_normal)
    backdoor_loader = get_dataloader(opt, mode=MODE.test_backdoor)
    denorm = Denormalize(opt.data_name)

    G, C, D = get_model(opt)
    CE = nn.CrossEntropyLoss().to(opt.device)
    MSE = nn.MSELoss().to(opt.device)
    BCE = nn.BCEWithLogitsLoss().to(opt.device)
    # LPIPS = lpips.LPIPS(verbose=False).to(opt.device)
    optimizer_G = optim.SGD(G.parameters(), lr=opt.lr_G, weight_decay=5e-4, momentum=0.9)
    optimizer_C = optim.SGD(C.parameters(), lr=opt.lr_C, weight_decay=5e-4, momentum=0.9)
    optimizer_D = optim.SGD(D.parameters(), lr=opt.lr_D, weight_decay=5e-4, momentum=0.9)
    if opt.data_name in [DATASET.celeba, DATASET.imagenet]:
        optimizer_G = optim.Adam(G.parameters(), lr=opt.lr_G, betas=(0.5, 0.9))
        # optimizer_C = optim.Adam(C.parameters(), lr=opt.lr_C, betas=(0.5, 0.9))
        # optimizer_D = optim.Adam(D.parameters(), lr=opt.lr_D, betas=(0.5, 0.9))
    scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, opt.scheduler_G, 0.1)
    scheduler_C = optim.lr_scheduler.MultiStepLR(optimizer_C, opt.scheduler_C, 0.1)
    scheduler_D = optim.lr_scheduler.MultiStepLR(optimizer_D, opt.scheduler_D, 0.1)

    run_dir, board, log_file = get_rundir_board_logger(opt)
    # 调整loss权重提高正常准确率。scale_cls：1 -> 1.5
    scale_adv, scale_img, scale_msg, scale_cls = 1e-3, 0.7, 1, opt.alpha
    best_acc_norm, best_acc_back, acc_save_list = 0, 0, []
    with open(log_file, 'w') as logger:
        for epoch in range(1, 1 + opt.epochs):
            # =====================================train=====================================
            G.train(), D.train(), C.train()
            total, total_bd, corr_norm, corr_back, corr_cover = 0, 1e-8, 0, 0, 0
            loss_img_tot, loss_back_tot, loss_norm_tot = 0, 0, 0

            # [real fake adv] [img msg cls] [normal attack cover]
            desc = 'train - epoch {:d}:'
            if not opt.no_D:
                desc += ' {:.3f} {:.3f} {:.3f}'
            desc += '【{:.6f} {:.6f} {:.6f}】{:.2%} {:.2%}'
            if not opt.no_cover:
                desc += ' {:.2%}'
            run_tqdm = tqdm(train_loader)
            for backdoor_label, X, y in run_tqdm:
                is_backdoor = backdoor_label != -1
                bd, total_bd = is_backdoor.sum(), total_bd + is_backdoor.sum()
                bs, total = len(X) - bd, total + len(X) - bd
                X, y, backdoor_label = X.to(opt.device), y.to(opt.device), backdoor_label.to(opt.device)

                # =================================train symbol=================================
                train_symbol = True
                real_loss = fake_loss = adv_loss = img_loss = back_loss = torch.tensor(0.0)
                source = torch.cat([X[is_backdoor], X[is_backdoor]], dim=0) if not opt.no_cover else X[is_backdoor]
                if bd:
                    target, gen_y = make_backdoor(y[is_backdoor], backdoor_label[is_backdoor],
                                                  get_attack_num_class(opt), opt.no_cover)
                    gen_x = G(source, target)
                    gen_pred = C(gen_x) if len(source) else gen_x

                    real = torch.full((len(source), 1), 1.0).to(opt.device)
                    fake = torch.full((len(source), 1), 0.0).to(opt.device)
                    if not opt.no_D:
                        real_loss = BCE(D(source), real)
                        fake_loss = BCE(D(gen_x.detach()), fake)
                        loss_D = real_loss + fake_loss
                        optimizer_D.zero_grad()
                        loss_D.backward()
                        optimizer_D.step()

                        adv_loss = BCE(D(gen_x), real) * scale_adv
                    img_loss = MSE(gen_x, source) * scale_img
                    back_loss = CE(gen_pred, gen_y) * scale_msg

                X, y = X[~ is_backdoor], y[~ is_backdoor]
                y_pred = C(X)
                norm_loss = CE(y_pred, y) * scale_cls
                loss_GC = adv_loss + img_loss + back_loss + norm_loss
                optimizer_G.zero_grad()
                optimizer_C.zero_grad()
                loss_GC.backward()
                optimizer_G.step()
                optimizer_C.step()

                loss_img_tot += img_loss.item() * len(source)
                loss_back_tot += back_loss.item() * len(source)
                loss_norm_tot += norm_loss.item() * bs
                corr_norm += (torch.argmax(y_pred, dim=1) == y).sum().item()
                if bd:
                    gen_pred = torch.argmax(gen_pred, dim=1)
                    corr_back += (gen_pred[:bd] == gen_y[:bd]).sum().item()
                    if not opt.no_cover:
                        corr_cover += (gen_pred[bd:] == gen_y[bd:]).sum().item()
                assert train_symbol
                # =================================train symbol=================================
                show_message = [epoch]
                if not opt.no_D:
                    show_message += [real_loss, fake_loss, adv_loss / scale_adv]
                show_message += [img_loss, back_loss, norm_loss, corr_norm / total, corr_back / total_bd]
                if not opt.no_cover:
                    show_message += [corr_cover / total_bd]
                show_message = desc.format(*show_message)
                run_tqdm.set_description(show_message)
            scheduler_G.step(), scheduler_C.step()
            if not opt.no_D:
                scheduler_D.step()
            logger_message(logger, show_message)
            board.add_scalars('loss/train',
                              {'img_mse': loss_img_tot / total_bd / 2, 'back_ce': loss_back_tot / total_bd / 2,
                               'norm_ce': loss_norm_tot / total}, epoch)
            board.add_scalars('acc', {'train_normal': corr_norm / total * 100,
                                      'train_backdoor': corr_back / total_bd * 100}, epoch)
            if not opt.no_cover:
                board.add_scalars('acc', {'train_cover': corr_cover / total_bd * 100}, epoch)

            # =====================================test normal=====================================
            G.eval(), D.eval(), C.eval()
            total, corr_norm, loss_tot = 0, 0, 0

            desc = ' norm - epoch {:d}:'
            if not opt.no_D:
                desc += '                  '
            desc += '【                  {:.6f}】{:.2%}       '
            if not opt.no_cover:
                desc += '       '
            run_tqdm = tqdm(normal_loader)
            for i, (X, y) in enumerate(run_tqdm):
                bs, total = len(X), total + len(X)
                X, y = X.to(opt.device), y.to(opt.device)

                with torch.no_grad():
                    y_pred = C(X)
                    loss = CE(y_pred, y) * scale_cls

                loss_tot += loss.item() * bs
                corr_norm += (torch.argmax(y_pred, dim=1) == y).sum().item()

                show_message = desc.format(epoch, loss, corr_norm / total)
                run_tqdm.set_description(show_message)

            logger_message(logger, show_message)
            board.add_scalars('loss/test', {'norm_ce': loss_tot / total}, epoch)
            board.add_scalars('acc', {'test_normal': corr_norm / total * 100}, epoch)

            # =====================================test backdoor=====================================
            G.eval(), D.eval(), C.eval()
            total_bd, corr_back, corr_cover = 0, 0, 0
            loss_img_tot, loss_back_tot = 0, 0

            desc = ' back - epoch {:d}:'
            if not opt.no_D:
                desc += ' {:.3f} {:.3f} {:.3f}'
            desc += '【{:.6f} {:.6f}         】       {:.2%}'
            if not opt.no_cover:
                desc += ' {:.2%}'
            run_tqdm = tqdm(backdoor_loader)
            for i, (backdoor_label, X, y) in enumerate(run_tqdm):
                bd, total_bd = len(X), total_bd + len(X)
                X, y, backdoor_label = X.to(opt.device), y.to(opt.device), backdoor_label.to(opt.device)

                with torch.no_grad():
                    source = torch.cat([X, X], dim=0) if not opt.no_cover else X
                    target, gen_y = make_backdoor(y, backdoor_label, get_attack_num_class(opt), opt.no_cover)
                    gen_x = G(source, target)
                    gen_pred = C(gen_x)

                    real = torch.full((len(source), 1), 1.0).to(opt.device)
                    fake = torch.full((len(source), 1), 0.0).to(opt.device)
                    adv_loss = 0
                    if not opt.no_D:
                        real_loss = BCE(D(source), real)
                        fake_loss = BCE(D(gen_x), fake)
                        adv_loss = BCE(D(gen_x), real) * scale_adv
                    img_loss = MSE(gen_x, source) * scale_img
                    back_loss = CE(gen_pred, gen_y) * scale_msg

                loss_img_tot += img_loss.item() * len(source)
                loss_back_tot += back_loss.item() * len(source)
                gen_pred = torch.argmax(gen_pred, dim=1)
                corr_back += (gen_pred[:bd] == gen_y[:bd]).sum().item()
                if not opt.no_cover:
                    corr_cover += (gen_pred[bd:] == gen_y[bd:]).sum().item()

                show_message = [epoch]
                if not opt.no_D:
                    show_message += [real_loss, fake_loss, adv_loss / scale_adv]
                show_message += [img_loss, back_loss, corr_back / total_bd]
                if not opt.no_cover:
                    show_message += [corr_cover / total_bd]
                show_message = desc.format(*show_message)
                run_tqdm.set_description(show_message)

                if i == 0:
                    img_dir = join(run_dir, 'img')
                    if not os.path.exists(img_dir):
                        os.makedirs(img_dir)
                    choice = np.random.choice(len(source), 8, replace=False)
                    source, gen_x = denorm(source[choice]), denorm(gen_x[choice])
                    source = F.interpolate(source, size=SAVE_SIZE)
                    gen_x = F.interpolate(gen_x, size=SAVE_SIZE)
                    if opt.data_name == DATASET.mnist:
                        diff = source - gen_x
                    else:
                        diff = transforms.Grayscale(num_output_channels=3)(source - gen_x)
                    images = torch.cat([source, gen_x, 20 * diff], dim=0)
                    save_image(images, join(img_dir, f'{epoch:03d}.png'), nrow=8)

            logger_message(logger, show_message)
            board.add_scalars('loss/test',
                              {'img_mse': loss_img_tot / total_bd / 2, 'back_ce': loss_back_tot / total_bd / 2}, epoch)
            board.add_scalars('acc', {'test_backdoor': corr_back / total_bd * 100,
                                      'test_cover': corr_cover / total_bd * 100}, epoch)

            # ===================================checkpoint===================================
            acc_norm, acc_back = corr_norm / total * 100, corr_back / total_bd * 100
            if (LOWEST_ACC[opt.data_name] < acc_norm and LOWEST_ACC[opt.data_name] < acc_back and
                    all([acc[0] < acc_norm or acc[1] < acc_back for acc in acc_save_list])):
                acc_save_list.append((acc_norm, acc_back))
                ckpt = {
                    'G': G.state_dict(),
                    'C': C.state_dict(),
                    'D': D.state_dict(),
                    'optimizer_G': optimizer_G.state_dict(),
                    'optimizer_C': optimizer_C.state_dict(),
                    'optimizer_D': optimizer_D.state_dict(),
                    'opt': opt,
                    'epoch': epoch,
                    'best_acc_norm': best_acc_norm,
                    'best_acc_back': best_acc_back,
                }
                best_acc_norm, best_acc_back = save_ckpt(epoch, run_dir, ckpt, acc_norm, acc_back, best_acc_norm,
                                                         best_acc_back)


def make_backdoor(y, backdoor_label, num_class, no_cover):
    target = F.one_hot(backdoor_label, num_class)
    gen_y = backdoor_label.clone().detach()
    if not no_cover:
        cover_target = torch.tensor(np.random.choice([0.0, 1.0], (len(y), num_class)), dtype=torch.float32).to(y.device)
        # 保证cover_target不是one-hot编码
        for i in range(len(y)):
            while cover_target[i].sum() == 1:
                cover_target[i] = torch.tensor(np.random.choice([0.0, 1.0], (num_class,)), dtype=torch.float32)
        target = torch.cat([target, cover_target], dim=0)
        gen_y = torch.cat([gen_y, y], dim=0)
    return target, gen_y


def train_compare(opt):
    get_backdoor_loader = get_generated_dataloader if opt.attacker == ATTACKER.dataonly else get_dataloader
    train_loader = get_backdoor_loader(opt, mode=MODE.train_backdoor)
    normal_loader = get_dataloader(opt, mode=MODE.test_normal)
    backdoor_loader = get_backdoor_loader(opt, mode=MODE.test_backdoor)
    denorm = Denormalize(opt.data_name)

    _, C, _ = get_model(opt)
    CE = nn.CrossEntropyLoss().to(opt.device)
    optimizer_C = optim.SGD(C.parameters(), lr=opt.lr_C, weight_decay=5e-4, momentum=0.9)
    if opt.data_name in [DATASET.celeba, DATASET.imagenet]:
        optimizer_C = optim.Adam(C.parameters(), lr=opt.lr_C, betas=(0.5, 0.9))
    scheduler_C = optim.lr_scheduler.MultiStepLR(optimizer_C, opt.scheduler_C, 0.1)

    run_dir, board, log_file = get_rundir_board_logger(opt)
    best_acc_norm, best_acc_back, acc_save_list = 0, 0, []
    with open(log_file, 'w') as logger:
        for epoch in range(1, 1 + opt.epochs):
            # =====================================train=====================================
            C.train()
            total_norm, total_back, corr_norm, corr_back = 0, 1e-8, 0, 0
            total, loss_tot = 0, 0

            desc = 'train - epoch {:d}: loss {:.6f} acc_norm {:.2%} acc_back {:.2%}'
            run_tqdm = tqdm(train_loader)
            for backdoor_label, X, y in run_tqdm:
                is_backdoor = backdoor_label != -1
                bs, total = len(X), total + len(X)
                X, y, backdoor_label = X.to(opt.device), y.to(opt.device), backdoor_label.to(opt.device)
                y[is_backdoor] = backdoor_label[is_backdoor]

                # =================================train symbol=================================
                train_symbol = True
                y_pred = C(X)
                loss = CE(y_pred, y)
                optimizer_C.zero_grad()
                loss.backward()
                optimizer_C.step()

                loss_tot += loss.item() * bs
                y_pred = torch.argmax(y_pred, dim=1)
                corr_back += (y_pred[is_backdoor] == y[is_backdoor]).sum().item()
                total_back += is_backdoor.sum().item()
                corr_norm += (y_pred == y).sum().item() - (y_pred[is_backdoor] == y[is_backdoor]).sum().item()
                total_norm += bs - is_backdoor.sum().item()
                assert train_symbol
                # =================================train symbol=================================
                show_message = desc.format(epoch, loss_tot / total, corr_norm / total_norm, corr_back / total_back)
                run_tqdm.set_description(show_message)
            scheduler_C.step()
            logger_message(logger, show_message)
            board.add_scalars('loss', {'train': loss_tot / total}, epoch)
            board.add_scalars('acc', {'train_normal': corr_norm / total_norm * 100,
                                      'train_backdoor': corr_back / total_back * 100, }, epoch)

            # =====================================test normal=====================================
            C.eval()
            total, corr_norm, loss_tot = 0, 0, 0

            desc = ' norm - epoch {:d}: loss {:.6f} acc_norm {:.2%}                '
            run_tqdm = tqdm(normal_loader)
            for X, y in run_tqdm:
                bs, total = len(X), total + len(X)
                X, y = X.to(opt.device), y.to(opt.device)

                with torch.no_grad():
                    y_pred = C(X)
                    loss = CE(y_pred, y)

                loss_tot += loss.item() * bs
                corr_norm += (torch.argmax(y_pred, dim=1) == y).sum().item()

                show_message = desc.format(epoch, loss_tot / total, corr_norm / total)
                run_tqdm.set_description(show_message)
            logger_message(logger, show_message)
            board.add_scalars('loss', {'test_normal': loss_tot / total}, epoch)
            board.add_scalars('acc', {'test_normal': corr_norm / total * 100}, epoch)

            # =====================================test backdoor=====================================
            C.eval()
            total, corr_back, loss_tot = 0, 0, 0

            desc = ' back - epoch {:d}: loss {:.6f}                 acc_back {:.2%}'
            run_tqdm = tqdm(backdoor_loader)
            for i, (y, X, _) in enumerate(run_tqdm):
                bs, total = len(X), total + len(X)
                X, y = X.to(opt.device), y.to(opt.device)

                with torch.no_grad():
                    y_pred = C(X)
                    loss = CE(y_pred, y)

                loss_tot += loss.item() * bs
                corr_back += (torch.argmax(y_pred, dim=1) == y).sum().item()

                show_message = desc.format(epoch, loss_tot / total, corr_back / total)
                run_tqdm.set_description(show_message)

                if i == 0:
                    img_dir = join(run_dir, 'img')
                    if not os.path.exists(img_dir):
                        os.makedirs(img_dir)
                    choice = np.random.choice(len(X), 9, replace=False)
                    images = F.interpolate(denorm(X[choice]), size=SAVE_SIZE)
                    save_image(images, join(img_dir, f'{epoch:03d}.png'), nrow=3)

            logger_message(logger, show_message)
            board.add_scalars('loss', {'test_backdoor': loss_tot / total}, epoch)
            board.add_scalars('acc', {'test_backdoor': corr_back / total * 100, }, epoch)

            # ===================================checkpoint===================================
            acc_norm, acc_back = corr_norm / total * 100, corr_back / total * 100
            if (LOWEST_ACC[opt.data_name] < acc_norm and LOWEST_ACC[opt.data_name] < acc_back and
                    all([acc[0] < acc_norm or acc[1] < acc_back for acc in acc_save_list])):
                acc_save_list.append((acc_norm, acc_back))
                ckpt = {
                    'C': C.state_dict(),
                    'optimizer_C': optimizer_C.state_dict(),
                    'opt': opt,
                    'epoch': epoch,
                    'best_acc_norm': best_acc_norm,
                    'best_acc_back': best_acc_back,
                }
                best_acc_norm, best_acc_back = save_ckpt(epoch, run_dir, ckpt, acc_norm, acc_back, best_acc_norm,
                                                         best_acc_back)


def save_ckpt(epoch, run_dir, ckpt, acc_norm, acc_back, best_acc_norm, best_acc_back):
    ckpt_dir = join(run_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_file = join(ckpt_dir, f'epoch_{epoch}_{acc_norm:.2f}_{acc_back:.2f}.pth')
    torch.save(ckpt, ckpt_file)
    if acc_norm > best_acc_norm:
        best_acc_norm = acc_norm
        ckpt_file = join(ckpt_dir, f'best_acc_norm_{acc_norm:.2f}_{acc_back:.2f}.pth')
        torch.save(ckpt, ckpt_file)
    if acc_back > best_acc_back:
        best_acc_back = acc_back
        ckpt_file = join(ckpt_dir, f'best_acc_back_{acc_norm:.2f}_{acc_back:.2f}.pth')
        torch.save(ckpt, ckpt_file)
    return best_acc_norm, best_acc_back


if __name__ == '__main__':
    pass
