from .common import TVLoss


def get_model(opt):
    from .classifier import PreActResNet18 as ResNet18, SimpleCNN
    from .discriminator import Discriminator
    from .generator import Generator
    from tools import get_attack_num_class, DATASET, ATTACKER
    G, D = None, None
    if 'attacker' in opt and opt.attacker == ATTACKER.generator:
        G = Generator(opt.data_name, get_attack_num_class(opt)).to(opt.device)
        D = Discriminator(opt.data_name).to(opt.device)
    C = (SimpleCNN() if opt.data_name == DATASET.mnist else ResNet18(opt.data_name)).to(opt.device)
    return G, C, D
