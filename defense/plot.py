from glob import glob
from os.path import join

import matplotlib.pyplot as plt

from tools import get_result_root, DATASET


def plot_neural_cleanse_histogram(root, is_show=False):
    mapping = {
        'badnet_all': 'BadNets*',
        'blended_all': 'Blended*',
        # 'issba': 'ISSBA',
        'generator': 'ours'
    }

    all_path = sorted(glob(join(root, 'nc*')))
    anomaly_index, attacker = [], []
    for path in all_path:
        attacker.append('_'.join(path.split('/')[-1].split('_')[1:-1]))
        attacker.append(mapping[attacker.pop()])
        with open(join(path, 'outlier.txt')) as f:
            anomaly_index.append(-min(eval(f.readlines()[3])))
    # attacker[2], attacker[3] = attacker[3], attacker[2]
    # anomaly_index[2], anomaly_index[3] = anomaly_index[3], anomaly_index[2]

    plt.bar(attacker, anomaly_index)
    plt.ylabel('anomaly index', fontsize=20)
    # plt.xticks(rotation=45)
    plt.tick_params(labelsize=20)
    plt.tight_layout()
    plt.savefig(join(root, f'result.pdf'), format='pdf')
    if is_show:
        plt.show()
    plt.close()


def plot_strip_entropy_histogram(root, rounds, is_show=False):
    file = join(root, f'entropy_{rounds}.txt')
    with open(file, 'r') as f:
        lines = f.readlines()
    entropy_benign, entropy_trojan = eval(lines[1]), eval(lines[3])
    if is_show:
        print(f'【entropy_benign】min: {min(entropy_benign)}, max: {max(entropy_benign)}')
        print(f'【entropy_trojan】min: {min(entropy_trojan)}, max: {max(entropy_trojan)}')

    n, bins = len(entropy_benign), 30
    plt.hist(entropy_benign, bins, alpha=0.8, weights=[1 / n] * n, label='benign')
    plt.hist(entropy_trojan, bins, alpha=0.8, weights=[1 / n] * n, label='BadNets*')
    plt.title('normalized entropy', fontsize=20)
    plt.xlabel('Entropy', fontsize=20)
    plt.ylabel('Probability (%)', fontsize=20)
    plt.legend(loc='upper right', fontsize=15)
    plt.tick_params(labelsize=20)
    plt.tight_layout()
    plt.savefig(join(root, f'result_{rounds}.pdf'), format='pdf')
    if is_show:
        plt.show()
    plt.close()


def plot_pruning_acc_line(root, is_show=False):
    file = join(root, 'accuracy.txt')
    with open(file, 'r') as f:
        lines = f.readlines()
    acc_norm_list, acc_back_list = eval(lines[1]), eval(lines[3])

    plt.plot(range(len(acc_norm_list)), acc_norm_list, label='Benign Accuracy')
    plt.plot(range(len(acc_norm_list)), acc_back_list, label='Attack Success Rate')
    plt.xlabel('pruned channels', fontsize=20)
    plt.ylabel('accuracy', fontsize=20)
    plt.legend(loc='lower left', fontsize=15)
    plt.tick_params(labelsize=20)
    plt.tight_layout()
    plt.savefig(join(root, f'result.pdf'), format='pdf')
    if is_show:
        plt.show()
    plt.close()


if __name__ == '__main__':
    # plot_neural_cleanse_histogram(join(get_result_root(), 'neural_cleanse', DATASET.cifar10), True)
    plot_pruning_acc_line(join(get_result_root(), 'fine_pruning', DATASET.cifar10, 'fp_generator_auto'), True)
    # plot_strip_entropy_histogram(join(get_result_root(), 'strip', DATASET.cifar10, 'sp_generator_auto'), 1, True)
    # plot_strip_entropy_histogram(join(get_result_root(), 'strip', DATASET.cifar10, 'sp_badnet_auto'), 1, True)
    # plot_strip_entropy_histogram(join(get_result_root(), 'strip', DATASET.cifar10, 'sp_blended_all_auto'), 1, True)
