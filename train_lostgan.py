import argparse
import os
import pickle
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

from utils.util import *
from data.cocostuff_loader import *
from data.vg import *
from model.plgan_128 import LostGANGenerator128
from model.plgan_256 import LostGANGenerator256
from model.discriminator_cal2i import *
from model.sync_batchnorm import DataParallelWithCallback
from utils.logger import setup_logger
from utils.util import one_hot_to_rgb


def get_dataset(args, num_obj):
    data_dir = args.data_dir
    dataset = args.dataset
    img_size = args.input_size
    if dataset == "coco":
        data = CocoSceneGraphDataset(image_dir=os.path.join(data_dir, 'images'),
                                     instances_json=os.path.join(data_dir, 'annotations/instances_train2017.json'),
                                     stuff_json=os.path.join(data_dir, 'annotations/stuff_train2017.json'),
                                     stuff_only=True, image_size=(img_size, img_size),
                                     max_objects_per_image=num_obj, left_right_flip=True)
    elif dataset == 'vg':
        with open(os.path.join(data_dir, 'vocab.json'), 'r') as load_f:
            vocab = json.load(load_f)
        data = VgSceneGraphDataset(vocab=vocab, h5_path=os.path.join(data_dir, 'train.h5'),
                                   image_dir=os.path.join(data_dir, 'VG'),
                                   image_size=(img_size, img_size), max_objects=num_obj-1, left_right_flip=True)
    else :
        raise ValueError('Dataset {} is not involved...'.format(dataset))

    return data


def main(args):
    # parameters
    img_size = args.input_size
    z_dim = 128
    lamb_obj = 1.0
    lamb_img = 0.1
    num_classes = 184 if args.dataset == 'coco' else 179
    num_obj = 8 if args.dataset == 'coco' else 31
    instance_threshold = 92 if args.dataset == 'coco' else 130
    colors = torch.randn(1024, 3).cuda()

    # data loader
    train_data = get_dataset(args, num_obj)

    dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, pin_memory=True,
        drop_last=True, shuffle=True, num_workers=args.workers)

    # Load model
    if args.input_size == 128:
        netG = LostGANGenerator128(num_classes=num_classes, output_dim=3, instance_threshold=instance_threshold).cuda()
        netD = CombineDiscriminator128(num_classes=num_classes).cuda()
    elif args.input_size == 256:
        netG = LostGANGenerator256(num_classes=num_classes, output_dim=3, instance_threshold=instance_threshold).cuda()
        netD = CombineDiscriminator256(num_classes=num_classes).cuda()
    else:
        raise ValueError('input_size %s must be in [64, 128, 256, 512]'%args.input_size)

    parallel = True
    if parallel:
        netG = DataParallelWithCallback(netG)
        netD = nn.DataParallel(netD)

    g_lr, d_lr = args.g_lr, args.d_lr
    gen_parameters = []
    for key, value in dict(netG.named_parameters()).items():
        if value.requires_grad:
            if 'mapping' in key:
                gen_parameters += [{'params': [value], 'lr': g_lr * 0.1}]
            else:
                gen_parameters += [{'params': [value], 'lr': g_lr}]
    g_optimizer = torch.optim.Adam(gen_parameters, betas=(0, 0.999))

    dis_parameters = []
    for key, value in dict(netD.named_parameters()).items():
        if value.requires_grad:
            dis_parameters += [{'params': [value], 'lr': d_lr}]
    d_optimizer = torch.optim.Adam(dis_parameters, betas=(0, 0.999))

    # make dirs
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    if not os.path.exists(os.path.join(args.out_path, 'model/')):
        os.mkdir(os.path.join(args.out_path, 'model/'))
    writer = SummaryWriter(os.path.join(args.out_path, 'log'))

    logger = setup_logger("lostGAN", args.out_path, 0)
    logger.info(netG)
    logger.info(netD)

    start_time = time.time()
    vgg_loss = VGGLoss()
    seg_feat_loss = FeatLoss()
    vgg_loss = nn.DataParallel(vgg_loss)
    l1_loss = nn.DataParallel(nn.L1Loss())

    if args.ckpt != '':
        checkpoint = torch.load(args.ckpt)
        start_epoch = checkpoint['epoch']
        netG.load_state_dict(checkpoint['netG'])
        netD.load_state_dict(checkpoint['netD'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer'])
    else:
        start_epoch = 0
    for epoch in range(start_epoch, args.total_epoch):
        netG.train()
        netD.train()

        for idx, data in enumerate(dataloader):
            real_images, label, bbox, filename, attributes = data
            real_images, label, bbox = real_images.cuda(), label.long().cuda().unsqueeze(-1), bbox.float().cuda()
            bbox_wh_gt = bbox[:, :, 2:]
            attributes = attributes.cuda()

            # update D network
            netD.zero_grad()
            real_images, label = real_images.cuda(), label.long().cuda()
            d_out_real, d_out_robj = netD(real_images, bbox, label)
            d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
            d_loss_robj = torch.nn.ReLU()(1.0 - d_out_robj).mean()

            z = torch.randn(real_images.size(0), num_obj, z_dim).cuda()
            # fake_images = netG(z, bbox, y=label.squeeze(dim=-1))
            fake_images, bbox_wh, mask, res_vis \
                = netG(z, bbox, attr=attributes, y=label.squeeze(dim=-1), vis=True)
            d_out_fake, d_out_fobj = netD(fake_images.detach(), bbox, label)
            d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
            d_loss_fobj = torch.nn.ReLU()(1.0 + d_out_fobj).mean()

            d_loss = lamb_obj * (d_loss_robj+d_loss_fobj) + lamb_img * (d_loss_real + d_loss_fake)
            d_loss.backward()
            if not clip_grad_value_(netD.parameters(), 1.1):
                logger.info('Iteration Skip...')
                continue
            d_optimizer.step()

            # update G network
            if (idx % 1) == 0:
                netG.zero_grad()
                g_out_fake, g_out_obj = netD(fake_images, bbox, label)
                g_loss_fake = - g_out_fake.mean()
                g_loss_obj = - g_out_obj.mean()

                pixel_loss = l1_loss(fake_images, real_images).mean()
                feat_loss = vgg_loss(fake_images, real_images).mean()

                y_mask = (label>0).float()
                bbox_loss = torch.sum((bbox_wh-bbox_wh_gt)**2*y_mask)/\
                            (torch.sum(y_mask)+1e-6)

                g_loss = g_loss_obj * lamb_obj + g_loss_fake * lamb_img \
                         + pixel_loss + feat_loss + bbox_loss
                g_loss.backward()
                if not clip_grad_value_(netG.parameters(), 1.1):
                    logger.info('Iteration Skip...')
                    continue
                g_optimizer.step()

            if (idx) % 100 == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                logger.info("Time Elapsed: [{}]".format(elapsed))
                logger.info("Step[{}/{}], d_out_real: {:.4f}, d_out_fake: {:.4f}, g_out_fake: {:.4f} ".format(epoch + 1,
                                                                                                              idx + 1,
                                                                                                              d_loss_real.item(),
                                                                                                              d_loss_fake.item(),
                                                                                                              g_loss_fake.item()))
                logger.info("             d_obj_real: {:.4f}, d_obj_fake: {:.4f}, g_obj_fake: {:.4f} ".format(
                    d_loss_robj.item(),
                    d_loss_fobj.item(),
                    g_loss_obj.item()))
                logger.info(
                    "             pixel_loss: {:.4f}, feat_loss: {:.4f}".format(pixel_loss.item(), feat_loss.item()))
                logger.info(
                    "             bbox_loss: {:.4f}".format(bbox_loss.item()))
                vis_num = 16
                writer.add_image("real images", make_grid(real_images[:vis_num].cpu().data * 0.5 + 0.5, nrow=4),
                                 epoch * len(dataloader) + idx + 1)
                writer.add_image("fake images", make_grid(fake_images[:vis_num].cpu().data * 0.5 + 0.5, nrow=4),
                                 epoch * len(dataloader) + idx + 1)
                for key in res_vis:
                    writer.add_image(key, make_grid(one_hot_to_rgb(res_vis[key][:vis_num], colors).cpu().data, nrow=4),
                                     epoch * len(dataloader) + idx + 1)

        # save model
        if (epoch + 1) % 2 == 0:
            torch.save({"netG": netG.state_dict(),
                        "netD": netD.state_dict(),
                        # "netM": netM.state_dict(),
                        "g_optimizer": g_optimizer.state_dict(),
                        "d_optimizer": d_optimizer.state_dict(),
                        'epoch': epoch},
                       os.path.join(args.out_path, 'model/', 'ckpt_%d.pth' % (epoch + 1)))
            # torch.save(netG.state_dict(), os.path.join(args.out_path, 'model/', 'G_%d.pth' % (epoch + 1)))
            # torch.save(netD.state_dict(), os.path.join(args.out_path, 'model/', 'D_%d.pth' % (epoch + 1)))

def clip_grad_value_(parameters, clip_value):
    r"""Clips gradient of an iterable of parameters at specified value.
    Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        clip_value (float or int): maximum allowed value of the gradients.
            The gradients are clipped in the range
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    clip_value = float(clip_value)
    for p in filter(lambda p: p.grad is not None, parameters):
        # if not torch.isfinite(p.grad).all():
        #     print('clip_grad_value_', torch.isfinite(p.grad).all())
        # p.grad.data.clamp_(min=-clip_value, max=clip_value)
        if not torch.isfinite(p.grad).all():
            return False
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco',
                        help='training dataset')
    parser.add_argument('--data_dir', type=str,
                        help='dataset directory')
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--batch_size', type=int, default=128,
                        help='mini-batch size of training data. Default: 32')
    parser.add_argument('--input_size', type=int, default=128,
                        help='input size of training data. Default: 128')
    parser.add_argument('--total_epoch', type=int, default=200,
                        help='number of total training epoch')
    parser.add_argument('--d_lr', type=float, default=0.0001,
                        help='learning rate for discriminator')
    parser.add_argument('--g_lr', type=float, default=0.0001,
                        help='learning rate for generator')
    parser.add_argument('--out_path', type=str, default='',
                        help='path to output files')
    parser.add_argument('--ckpt', type=str, default='',
                        help='path to output files')
    args = parser.parse_args()
    main(args)
