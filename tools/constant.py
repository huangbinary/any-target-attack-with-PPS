class DATASET:
    mnist = 'mnist'
    cifar10 = 'cifar10'
    gtsrb = 'gtsrb'
    imagenet = 'imagenet'
    celeba = 'celeba'


SAVE_SIZE = (128, 128)
IMG_SIZE = {DATASET.mnist: 28, DATASET.cifar10: 32, DATASET.gtsrb: 32, DATASET.imagenet: 64, DATASET.celeba: 64}
NUM_CHANNEL = {DATASET.mnist: 1, DATASET.cifar10: 3, DATASET.gtsrb: 3, DATASET.imagenet: 3, DATASET.celeba: 3}
NUM_CLASS = {DATASET.mnist: 10, DATASET.cifar10: 10, DATASET.gtsrb: 43, DATASET.imagenet: 200, DATASET.celeba: 8}

MEAN = {DATASET.mnist: [0.5], DATASET.cifar10: [0.4914, 0.4822, 0.4465], DATASET.gtsrb: [0.4914, 0.4822, 0.4465],
        DATASET.imagenet: [0.4824, 0.4495, 0.3981], DATASET.celeba: [0.5063, 0.4258, 0.3832]}
STD = {DATASET.mnist: [0.5], DATASET.cifar10: [0.247, 0.2435, 0.2616], DATASET.gtsrb: [0.247, 0.2435, 0.2616],
       DATASET.imagenet: [0.277, 0.2693, 0.2829], DATASET.celeba: [0.3043, 0.2839, 0.2834]}

# LOWEST_ACC = {DATASET.mnist: 90, DATASET.cifar10: 80, DATASET.gtsrb: 80, DATASET.imagenet: 0}
LOWEST_ACC = {DATASET.mnist: 0, DATASET.cifar10: 0, DATASET.gtsrb: 0, DATASET.imagenet: 0, DATASET.celeba: 0}


class NUM_TOTAL:
    train = {DATASET.mnist: 60000, DATASET.cifar10: 50000, DATASET.gtsrb: 39209,
             DATASET.imagenet: 100000, DATASET.celeba: 162770}
    test = {DATASET.mnist: 10000, DATASET.cifar10: 10000, DATASET.gtsrb: 12630,
            DATASET.imagenet: 10000, DATASET.celeba: 19962}


class MODE:
    train_normal = 'train_normal'
    train_backdoor = 'train_backdoor'
    test_normal = 'test_normal'
    test_backdoor = 'test_backdoor'
    train = [train_normal, train_backdoor]
    test = [test_normal, test_backdoor]
    normal = [train_normal, test_normal]
    backdoor = [train_backdoor, test_backdoor]


class ATTACKER:
    normal = 'normal'
    badnet = 'badnet'
    blended = 'blended'
    issba = 'issba'
    generator = 'generator'
    dataonly = 'dataonly'


class DEFENSER:
    neural_cleanse = 'nc'
    strip = 'sp'
    fine_pruning = 'fp'
    gradcam = 'gc'
    activation_clustering = 'ac'


GENERATOR_PATH = lambda data_name: f'result/{data_name}/models/defense/{data_name}_generator_a2_r0.1.pth'


class BADNET_TRIGGER:
    mnist = [
        ([3, 4, 3, 5], [3, 4, 5, 3]),
        ([3, 3, 4, 4], [10, 11, 10, 11]),
        ([3, 3, 4, 4], [16, 17, 16, 17]),
        ([3, 4, 3, 5], [24, 23, 22, 24]),
        ([13, 14, 13, 14], [3, 3, 4, 4]),
        ([13, 14, 13, 14], [23, 23, 24, 24]),
        ([24, 23, 24, 22], [3, 4, 5, 3]),
        ([23, 23, 24, 24], [10, 11, 10, 11]),
        ([23, 23, 24, 24], [16, 17, 16, 17]),
        ([24, 23, 24, 22], [24, 23, 22, 24]),
    ]
    __cifar10_border = [
        *[[[i, j], [k, l]] for i, j in [[3, 6], [26, 29]] for k, l in [[3, 6], [11, 14], [18, 21], [26, 29]]],
        *[[[14, 17], [i, j]] for i, j in [[3, 6], [26, 29]]]
    ]
    # 前面range是行，后面range是列
    cifar10 = [([i for i in range(row[0], row[1]) for _ in range(col[0], col[1])],
                [i for _ in range(row[0], row[1]) for i in range(col[0], col[1])]) for row, col in __cifar10_border]
    __gtsrb_border = [
        *[[[i, j], [k, l]] for i, j in [[3, 6], [27, 30]] for k, l in [[m, m + 3] for m in range(3, 30, 3)]],
        *[[[k, l], [i, j]] for i, j in [[3, 6], [27, 30]] for k, l in [[m, m + 3] for m in range(6, 27, 3)]],
        *[[[i, i + 1], [k, l]] for i in [3, 28] for k, l in [[m, m + 9] for m in range(3, 30, 9)]],
        *[[[k, l], [i, i + 1]] for i in [3, 28] for k, l in [[m, m + 9] for m in range(3, 30, 9)]]
    ]
    gtsrb = [([i for i in range(row[0], row[1]) for _ in range(col[0], col[1])],
              [i for _ in range(row[0], row[1]) for i in range(col[0], col[1])]) for row, col in __gtsrb_border]
    # 暂时只有1个
    imagenet = [([i for i in range(3, 9) for _ in range(3, 9)], [i for _ in range(3, 9) for i in range(3, 9)])]
    __celeba_border = [[[i, j], [k, l]] for i, j in [[3, 9], [29, 35], [55, 61]]
                       for k, l in [[3, 9], [29, 35], [55, 61]]]
    celeba = [([i for i in range(row[0], row[1]) for _ in range(col[0], col[1])],
               [i for _ in range(row[0], row[1]) for i in range(col[0], col[1])]) for row, col in __celeba_border]


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt


    class TRIGGER:
        @staticmethod
        def test_trigger(data_name, show_first=False):
            trigger = getattr(BADNET_TRIGGER, data_name)
            img = np.zeros((IMG_SIZE[data_name],) * 2)
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if (i + j) % 2:
                        img[i, j] = 127
            print(trigger)
            if show_first:
                img[trigger[0]] = 255
            else:
                for t in trigger:
                    img[t] = 255
            plt.imshow(img, cmap='gray')
            plt.show()


    TRIGGER.test_trigger(DATASET.celeba, show_first=False)
