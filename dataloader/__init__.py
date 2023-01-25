from .common import get_transform, Normalize, Denormalize, ToTensor, ToNumpy


def worker_init(worker_id):
    import os
    import random

    import numpy as np

    seed = int(os.environ['STUDY_SEED']) - 10 + worker_id
    random.seed(seed)
    np.random.seed(seed)


def get_dataloader(opt, mode, num_sample=None, no_normalize=False):
    from torch.utils.data import DataLoader

    from tools import MODE, DATASET
    from .cifar10 import Cifar10
    from .gtsrb import Gtsrb
    from .imagenet import Imagenet
    from .mnist import Mnist
    from .celeba import Celeba

    assert mode in MODE.__dict__.values()
    loaders = {DATASET.mnist: Mnist, DATASET.cifar10: Cifar10, DATASET.gtsrb: Gtsrb,
               DATASET.imagenet: Imagenet, DATASET.celeba: Celeba}
    return DataLoader(loaders[opt.data_name](opt, mode, num_sample, no_normalize),
                      batch_size=opt.batch_size if num_sample is None else num_sample,
                      shuffle=(mode in MODE.train), num_workers=2, worker_init_fn=worker_init)


def get_generated_dataloader(opt, mode):
    from torch.utils.data import DataLoader

    from .dataset import GeneratedDataset
    from tools import MODE

    assert mode in MODE.backdoor
    return DataLoader(GeneratedDataset(opt, mode), batch_size=opt.batch_size, shuffle=(mode in MODE.train),
                      num_workers=2, worker_init_fn=worker_init)
