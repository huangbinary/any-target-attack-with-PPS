from os.path import join

import bchlib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from tools import get_issba_root

BCH_POLYNOMIAL = 137
BCH_BITS = 5


def encode_image(image, secret, device):
    encoder_path = join(get_issba_root(), 'ckpt', f'encoder_{image.shape[1]}.pth')

    encoder = torch.load(encoder_path, map_location=device)
    encoder.eval()
    encoder = encoder.to(device)

    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    data = bytearray(secret + ' ' * (7 - len(secret)), 'utf-8')
    ecc = bch.encode(data)
    packet = data + ecc

    packet_binary = ''.join(format(x, '08b') for x in packet)
    secret = [int(x) for x in packet_binary]
    secret.extend([0, 0, 0, 0])
    secret = torch.tensor(secret, dtype=torch.float).unsqueeze(0)
    secret = secret.to(device)

    to_tensor = transforms.ToTensor()

    with torch.no_grad():
        image = Image.fromarray(image)
        image = to_tensor(image).unsqueeze(0)
        image = image.to(device)

        residual = encoder((secret, image))
        encoded = image + residual
        residual = residual.cpu()
        encoded = encoded.cpu()
        encoded = np.array(encoded.squeeze(0) * 255, dtype=np.uint8).transpose((1, 2, 0))

        residual = residual[0] + .5
        residual = np.array(residual.squeeze(0) * 255, dtype=np.uint8).transpose((1, 2, 0))
    return encoded


if __name__ == "__main__":
    pass
