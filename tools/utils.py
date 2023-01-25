import os
import random
import shutil
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
from os.path import join

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from tqdm import tqdm

from .constant import DATASET, IMG_SIZE, NUM_CLASS


def init_seed(opt):
    if opt.seed is None:
        seed = int(random.random() * 2 ** 32)
        print(f'===> init random seed {seed}')
    else:
        seed = opt.seed
        print(f'===> use given seed {seed}...')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['STUDY_SEED'] = str(seed)


def get_root():
    project_name = 'StudyOne'
    file_path = os.path.dirname(__file__)
    root_path = file_path[:file_path.find(project_name) + len(project_name)]
    return root_path


def get_dataset_root():
    return join(get_root(), 'dataset')


def get_blended_root():
    return join(get_dataset_root(), 'blended')


def get_save_root():
    return join(get_root(), 'save')


def get_result_root():
    return join(get_save_root(), 'result')


def get_issba_root():
    return join(get_root(), 'issba')


def get_ftd_ckpt():
    return join(get_root(), 'defense', 'ckpt', 'ftd.pth')


def get_csv_path(root):
    for file in os.listdir(root):
        if file.endswith(".csv"):
            return join(root, file)


def get_attack_num_class(opt):
    if opt.attack_num == -1:
        return NUM_CLASS[opt.data_name]
    return opt.attack_num


def calcu_mean_std(loader):
    data_name = loader.dataset.data_name
    assert data_name != DATASET.gtsrb
    n = len(loader.dataset) * IMG_SIZE[data_name] ** 2
    rgb = torch.zeros(3)
    for X, _ in tqdm(loader):
        for i in range(3):
            rgb[i] += X[:, i].sum()
    mean = rgb / n
    rgb = torch.zeros(3)
    for X, _ in tqdm(loader):
        for i in range(3):
            rgb[i] += ((X[:, i] - mean[i]) ** 2).sum()
    std = np.sqrt(rgb / n)
    print(f'mean = {list(map(lambda x: float(f"{x:.4f}"), mean.tolist()))}')
    print(f' std = {list(map(lambda x: float(f"{x:.4f}"), std.tolist()))}')
    return mean, std


def get_rundir_board_logger(opt, log_name='log', tensorboard=True):
    run_dir = join(get_save_root(), opt.data_name, opt.run_name)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)
    if tensorboard:
        log_dir = join(run_dir, 'log')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    log_file = join(run_dir, f'{log_name}.txt')
    os.mknod(join(run_dir, f"{os.environ['STUDY_SEED']}.seed"))
    if tensorboard:
        return run_dir, SummaryWriter(log_dir=log_dir), log_file
    return run_dir, log_file


def logger_message(logger, message, is_print=False):
    if is_print:
        print(message)
    logger.write(f'{message}\n')
    logger.flush()


def pretty(s, total):
    left = (total - len(str(s))) // 2
    right = total - left - len(str(s))
    return ' ' * left + str(s) + ' ' * right


def send_complete_notification():
    from_ = ''
    pswd = ''
    to = ''
    try:
        msg = MIMEText('模型跑完了，请及时查看结果', 'plain', 'utf-8')
        msg['From'] = formataddr(["科研小助手", from_])
        msg['To'] = formataddr(["苦逼研究生", to])
        msg['Subject'] = "模型跑完了"
        server = smtplib.SMTP_SSL("smtp.qq.com", 465)
        server.login(from_, pswd)
        server.sendmail(from_, [to, ], msg.as_string())
        server.quit()
    except Exception:
        print('通知邮件发送失败！！！')
    finally:
        print('通知邮件发送成功！！！')


if __name__ == '__main__':
    pass
