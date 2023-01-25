import os
from os.path import join

from tools import ATTACKER, DEFENSER, DATASET
from .fine_pruning import test_fine_pruning
from .neural_cleanse import train_neural_cleanse
from .strip import test_strip


def test_defense(opt):
    def test_one_dir(root):
        # 不用walk的原因是文件夹下可能保存了一些其他配置的旧结果
        for file in sorted(os.listdir(root)):
            if file in DATASET.__dict__.values():
                test_one_dir(join(root, file))
            elif file.endswith('.pth'):
                data_name, attacker, attack_num = file[:-len('.pth')].split('_')[:3]
                if defenser in [DEFENSER.strip, DEFENSER.fine_pruning] and attacker == ATTACKER.normal:
                    continue
                if defenser in [DEFENSER.gradcam] and data_name == DATASET.mnist:
                    continue
                if attack_num == 'all':
                    opt.run_name = f'{defenser}_{attacker}_all_{run_name}'
                else:
                    opt.run_name = f'{defenser}_{attacker}_{run_name}'
                opt.ckpt_path = join(root, file)
                print('\n' + '=' * 45 + f' check {file} ' + '=' * 45 + '\n')
                run_defense(opt)

    defenser = opt.command
    run_defense = {DEFENSER.neural_cleanse: train_neural_cleanse, DEFENSER.strip: test_strip,
                   DEFENSER.fine_pruning: test_fine_pruning}[defenser]
    if os.path.isdir(opt.ckpt_path):
        root, run_name = opt.ckpt_path, opt.run_name
        test_one_dir(root)
    else:
        run_defense(opt)
