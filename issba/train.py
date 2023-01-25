import os
from os.path import join

import lpips
import yaml
from easydict import EasyDict
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from tools import get_issba_root, IMG_SIZE
from .dataset import StegaData
from .model import *
from .utils import *

with open(join(get_issba_root(), 'setting.yaml'), 'r') as f:
    args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))


def train_issba(data_name):
    train_path, img_size = f'dataset/{data_name}/all_images', IMG_SIZE[data_name]
    dataset = StegaData(train_path, args.secret_size, size=(img_size, img_size))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    encoder = StegaStampEncoder(img_size)
    decoder = StegaStampDecoder(img_size, secret_size=args.secret_size)
    discriminator = Discriminator()
    lpips_alex = lpips.LPIPS(net="alex", verbose=False)
    encoder = encoder.to(args.cuda)
    decoder = decoder.to(args.cuda)
    discriminator = discriminator.to(args.cuda)
    lpips_alex.to(args.cuda)

    d_vars = discriminator.parameters()
    g_vars = [{'params': encoder.parameters()},
              {'params': decoder.parameters()}]

    optimize_loss = optim.Adam(g_vars, lr=args.lr)
    optimize_secret_loss = optim.Adam(g_vars, lr=args.lr)
    optimize_dis = optim.RMSprop(d_vars, lr=0.00001)

    total_steps = len(dataset) // args.batch_size + 1
    global_step = 0

    while global_step < args.num_steps:
        desc = 'loss = {:.4f}'
        run_tqdm = tqdm(range(min(total_steps, args.num_steps - global_step)), desc=desc.format(0))
        for _ in run_tqdm:
            image_input, secret_input = next(iter(dataloader))
            image_input = image_input.to(args.cuda)
            secret_input = secret_input.to(args.cuda)
            no_im_loss = global_step < args.no_im_loss_steps
            l2_loss_scale = min(args.l2_loss_scale * global_step / args.l2_loss_ramp, args.l2_loss_scale)
            lpips_loss_scale = min(args.lpips_loss_scale * global_step / args.lpips_loss_ramp, args.lpips_loss_scale)
            secret_loss_scale = min(args.secret_loss_scale * global_step / args.secret_loss_ramp,
                                    args.secret_loss_scale)
            G_loss_scale = min(args.G_loss_scale * global_step / args.G_loss_ramp, args.G_loss_scale)
            l2_edge_gain = 0
            if global_step > args.l2_edge_delay:
                l2_edge_gain = min(args.l2_edge_gain * (global_step - args.l2_edge_delay) / args.l2_edge_ramp,
                                   args.l2_edge_gain)

            rnd_tran = min(args.rnd_trans * global_step / args.rnd_trans_ramp, args.rnd_trans)
            rnd_tran = np.random.uniform() * rnd_tran

            global_step += 1
            Ms = get_rand_transform_matrix(img_size, np.floor(img_size * rnd_tran), args.batch_size)
            Ms = Ms.to(args.cuda)

            loss_scales = [l2_loss_scale, lpips_loss_scale, secret_loss_scale, G_loss_scale]
            yuv_scales = [args.y_scale, args.u_scale, args.v_scale]
            loss, secret_loss, D_loss, bit_acc, str_acc = build_model(encoder, decoder, discriminator, lpips_alex,
                                                                      secret_input, image_input,
                                                                      args.l2_edge_gain, args.borders,
                                                                      args.secret_size, Ms, loss_scales,
                                                                      yuv_scales, args, global_step)
            if no_im_loss:
                optimize_secret_loss.zero_grad()
                secret_loss.backward()
                optimize_secret_loss.step()
            else:
                optimize_loss.zero_grad()
                loss.backward()
                optimize_loss.step()
                if not args.no_gan:
                    optimize_dis.zero_grad()
                    optimize_dis.step()

            # if global_step % 10 == 0:
            run_tqdm.set_description(desc.format(loss))
    torch.save(encoder, os.path.join(args.saved_models, f"encoder_{img_size}.pth"))
    torch.save(decoder, os.path.join(args.saved_models, f"decoder_{img_size}.pth"))


if __name__ == '__main__':
    pass
