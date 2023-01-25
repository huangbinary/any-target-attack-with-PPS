import argparse

from tools import init_seed, DATASET, ATTACKER


def parse_normal(parser):
    parser.add_argument('run_name', type=str, help='保存模型、日志的目录')
    parser.add_argument('--device', '-d', type=str, default='cuda:0', help='使用的GPU')
    parser.add_argument('--data_name', '-n', type=str, default=DATASET.cifar10, help='数据集名称')
    parser.add_argument('--seed', '-s', type=int, default=None, help='随机数种子')

    # 训练的超参数
    parser.add_argument('--batch_size', type=int, default=128, help='批量大小')
    parser.add_argument('--epochs', type=int, default=310, help='训练轮次')
    parser.add_argument('--lr', type=float, default=1e-2, help='分类器学习率')
    # 注意i从1开始，不然一开始就会衰减
    parser.add_argument("--scheduler", type=list, default=[100 * i for i in range(1, 10)])


def parse_backdoor(parser):
    parser.add_argument('run_name', type=str, help='保存模型、日志的文目录')
    parser.add_argument('--device', '-d', type=str, default='cuda:0', help='使用的GPU')
    parser.add_argument('--data_name', '-n', type=str, default=DATASET.cifar10, help='数据集名称')
    parser.add_argument('--seed', type=int, default=None, help='随机数种子')
    # 攻击的超参数
    parser.add_argument('--attacker', '-a', type=str, default=ATTACKER.generator, help='攻击方法')
    # 只在ImageNet数据集时> 1才有效，表示攻击类别[0, attack_num)，若为-1则表示攻击所有类别
    parser.add_argument('--attack_num', type=int, default=-1, help='攻击的目标类别数量')
    # 只在attack_num=1时有效，表示单类别攻击的目标类别
    parser.add_argument('--attack_label', '-l', type=int, default=0, help='攻击的目标类别')
    parser.add_argument('--ratio', '-r', type=float, default=0.1,
                        help='所有类别的后门样本比例，若为单目标攻击，则目标类别的后门样本比例只有1/num_class')
    parser.add_argument('--no_cover', action='store_true', help='是否使用cover样本')
    parser.add_argument('--no_D', action='store_true', help='是否使用判别器D')
    parser.add_argument('--alpha', type=float, default=2.0, help='正常样本分类损失的权重')

    # 训练的超参数
    parser.add_argument('--batch_size', type=int, default=128, help='批量大小')
    parser.add_argument('--epochs', type=int, default=310, help='训练轮次')
    parser.add_argument('--lr_G', type=float, default=1e-2, help='生成器学习率')
    parser.add_argument('--lr_C', type=float, default=1e-2, help='分类器学习率')
    parser.add_argument('--lr_D', type=float, default=1e-2, help='判别器学习率')
    # 注意i从1开始，不然一开始就会衰减
    parser.add_argument("--scheduler_G", type=list, default=[100 * i for i in range(1, 10)])
    parser.add_argument("--scheduler_C", type=list, default=[100 * i for i in range(1, 10)])
    parser.add_argument("--scheduler_D", type=list, default=[100 * i for i in range(1, 10)])


def parse_evaluate(parser):
    parser.add_argument('ckpt_path', type=str, help='需要测试的ckpt路径')
    parser.add_argument('--device', '-d', type=str, default='cuda:0', help='使用的GPU')
    parser.add_argument('--seed', type=int, default=None, help='随机数种子')


def parse_neural_cleanse(parser):
    parser.add_argument('--nc_rounds', type=int, default=1, help='运行NC的次数')
    parser.add_argument('--epochs', type=int, default=300, help='每次运行训练的轮次')
    parser.add_argument('--batch_size', type=int, default=128, help='批量大小')
    parser.add_argument('--lr', type=float, default=1e-1, help='学习率')
    # 正则项系数特征相关
    parser.add_argument("--init_alpha", type=float, default=1e-1, help='loss_norm初始权重')
    parser.add_argument("--alpha_change_max_counter", type=int, default=5, help='修改alpha的条件')
    parser.add_argument("--early_stop_max_counter", type=int, default=15, help='early stop生效的条件')
    parser.add_argument("--search_range_threshold", type=float, default=1e-3, help='停止更新alpha的条件')
    parser.add_argument("--early_stop_threshold", type=float, default=0.99, help='early_stop_counter增加的条件')
    parser.add_argument("--asr_threshold", type=float, default=0.99, help='攻击成功的ASR指标')


def parse_strip(parser):
    parser.add_argument('--sp_rounds', type=int, default=3, help='运行strip的次数')
    # 采样相关的超参数
    parser.add_argument("--num_background", type=int, default=1000, help='测试的正常/后门样本数')
    parser.add_argument("--num_superposition", type=int, default=100, help='叠加在background上的样本数')
    parser.add_argument("--frr_threshold", type=float, default=0.01, help='错误拒绝率')
    parser.add_argument("--far_threshold", type=float, default=0.01, help='错误接收率')


def parse_defense_common(parser):
    parser.add_argument('run_name', type=str, help='保存模型、日志的文目录')
    # 当ckpt_path为文件夹时表示检测里面的所有模型
    # 此时run_name只是个后缀，真实的run_name为'nc_attacker_runname'
    parser.add_argument('ckpt_path', type=str, help='需要防御的ckpt路径')
    parser.add_argument('--device', '-d', type=str, default='cuda:0', help='使用的GPU')
    parser.add_argument('--seed', type=int, default=None, help='随机数种子')


def parse_param():
    parser = argparse.ArgumentParser()
    # [n:normal, b: backdoor, e: evaluate, d: defense]
    sub_parser = parser.add_subparsers(dest='command', required=True, help='需要运行的功能')

    # ========================================== normal ===========================================
    norm_parser = sub_parser.add_parser('n', help='训练正常模型')
    parse_normal(norm_parser)
    # ========================================= backdoor ==========================================
    back_parser = sub_parser.add_parser('a', help='训练后门攻击模型')
    parse_backdoor(back_parser)
    # ========================================= evaluate ==========================================
    eval_parser = sub_parser.add_parser('e', help='测试模型')
    parse_evaluate(eval_parser)
    # ====================================== neural_cleanse =======================================
    nc_parser = sub_parser.add_parser('nc', help='neural_cleanse防御方法')
    parse_neural_cleanse(nc_parser)
    # ========================================== strip ============================================
    sp_parser = sub_parser.add_parser('sp', help='strip防御方法')
    parse_strip(sp_parser)
    # ======================================= fine_pruning ========================================
    fp_parser = sub_parser.add_parser('fp', help='fine_pruning防御方法')
    for defense_parser in [nc_parser, sp_parser, fp_parser]:
        parse_defense_common(defense_parser)

    opt = parser.parse_args()
    init_seed(opt)
    return opt


if __name__ == '__main__':
    pass
