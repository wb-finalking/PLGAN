import argparse
import os
import numpy as np
from scipy import misc
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import json
from collections import OrderedDict
from skimage import img_as_ubyte

from data.cocostuff_loader import *
from data.vg import *
from model.plgan_256 import LostGANGenerator256Freeze
from utils.util import *


def get_color_table(class_num):
    '''
    return :  list of (r, g, b) color
    '''
    color_table = []
    for i in range(class_num):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        color_table.append((b, g, r))
    return color_table

def constrait(x, start, end):
    '''
    return:x    ,start <= x <= end
    '''
    if x < start:
        return start
    elif x > end:
        return end
    else:
        return x

def draw_img(img, boxes, label, word_dict, color_table, ):
    '''
    img : cv2.img [416, 416, 3]
    boxes:[V, 4], x_min, y_min, x_max, y_max
    score:[V], score of corresponding box
    label:[V], label of corresponding box
    word_dict: dictionary of  id=>name
    return : a image after draw the boxes
    '''
    # img = np.zeros_like(img)
    # img = img.reshape((128,128,3))
    img = np.ones((512,512,3))*255
    w = img.shape[1]
    h = img.shape[0]
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # boxes = boxes/128.0
    for i in range(len(boxes)):
        boxes[i][0] = constrait(boxes[i][0], 0, 1)
        boxes[i][1] = constrait(boxes[i][1], 0, 1)
        boxes[i][2] = constrait(boxes[i][2], 0, 1)
        boxes[i][3] = constrait(boxes[i][3], 0, 1)
        x_min = int(boxes[i][0] * w)
        x_max = int((boxes[i][0]+boxes[i][2])* w)
        y_min = int(boxes[i][1] * h)
        y_max = int((boxes[i][1] + boxes[i][3]) * h)

        curr_label = label[i] if label is not None else 0
        curr_color = color_table[curr_label] if color_table is not None else (0, 125, 255)

        if int(curr_label[0]) == 0:
            continue

        curr_color = (int(curr_color[0,0]), int(curr_color[0,1]), int(curr_color[0,2]))
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), curr_color, thickness=2)
        # draw font
        if word_dict is not None:
            text_name = "{}".format(word_dict[int(curr_label[0])])
            cv2.putText(img, text_name, (x_min, y_min+25), font, 1, curr_color, 2)
        # if score is not None:
        #     text_score = "{:2d}%".format(int(score[i] * 100))
        #     cv2.putText(img, text_score, (x_min, y_min+25), font, 1, curr_color)
    return img

def get_dataloader(args, num_obj):
    data_dir = args.data_dir
    dataset = args.dataset
    img_size = args.input_size

    if args.dump_bbox_dir is not None:
        with open(args.dump_bbox_dir, 'r') as f:
            dump_bbox_dict = json.load(f)
    else:
        dump_bbox_dict = None
    if dataset == "coco":
        data = CocoSceneGraphDataset(image_dir=os.path.join(data_dir, 'val2017'),
                                     instances_json=os.path.join(data_dir, 'annotations/instances_{}2017.json'.format(args.set)),
                                     stuff_json=os.path.join(data_dir, 'annotations/stuff_{}2017.json'.format(args.set)),
                                     stuff_only=True, image_size=(img_size, img_size),
                                     max_objects_per_image=num_obj, dump_bbox_dict=dump_bbox_dict,
                                     filter_mode=args.filter_mode, left_right_flip=True)
    elif dataset == 'vg':
        with open(os.path.join(data_dir, 'vocab.json'), 'r') as load_f:
            vocab = json.load(load_f)
        data = VgSceneGraphDataset(vocab=vocab, h5_path=os.path.join(data_dir, '{}.h5'.format(args.set)),
                                   image_dir=os.path.join(data_dir, 'VG'),
                                   dump_bbox_dict=dump_bbox_dict,
                                   image_size=(img_size, img_size), max_objects=num_obj-1, left_right_flip=True)
    else :
        raise ValueError('Dataset {} is not involved...'.format(dataset))

    dataloader = torch.utils.data.DataLoader(
        data, batch_size=1,
        drop_last=True, shuffle=False, num_workers=1)

    return dataloader


def main(args):
    num_classes = 184 if args.dataset == 'coco' else 179
    num_o = 8 if args.dataset == 'coco' else 31
    instance_threshold = 92 if args.dataset == 'coco' else 130

    dataloader = get_dataloader(args, num_o)
    if args.dataset == 'coco':
        with open('data/coco_vocab.json', 'r') as f:
            import json
            vocab = json.load(f)
        word_dict = vocab['object_idx_to_name']
    else:
        with open('data/vg_vocab.json', 'r') as f:
            import json
            vocab = json.load(f)
        word_dict = vocab['object_idx_to_name']

    # Load model
    netG = LostGANGenerator256Freeze(num_classes=num_classes, output_dim=3, instance_threshold=instance_threshold).cuda()

    if not os.path.isfile(args.model_path):
        return
    print('==>loading ', args.model_path)
    state_dict = torch.load(args.model_path)['netG']

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`nvidia
        new_state_dict[name] = v

    # model_dict = netG.state_dict()
    # pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    netG.load_state_dict(new_state_dict)

    netG.cuda()
    netG.eval()
    color_table = torch.FloatTensor(np.load('./color.npy')).cuda()
    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)
    thres=2.0
    if args.set == 'train':
        set = 'train'
    else:
        set = 'test'
    for sample_idx in range(args.sample_times):
        for idx, data in enumerate(dataloader):
            real_images, label, bbox, filename, attributes = data
            # layouts = draw_img(real_images[0].numpy(), bbox[0].numpy(), label[0].numpy(), word_dict, color_table, )
            filename = os.path.splitext(os.path.basename(filename[0]))[0]

            real_images, label, bbox = real_images.cuda(), label.long().cuda().unsqueeze(-1), bbox.float().cuda()
            attributes = attributes.cuda()

            z_obj = torch.from_numpy(truncted_random(num_o=num_o, thres=thres)).float().cuda()
            z_im = torch.from_numpy(truncted_random(num_o=1, thres=thres)).view(1, -1).float().cuda()

            with torch.no_grad():
                fake_images, bbox_wh, mask = netG.forward(z_obj, bbox[:, :, :2], attributes, z_im, label.squeeze(dim=-1))

                # freeze
                def remove_hooks(model):
                    model._backward_hooks = OrderedDict()
                    model._forward_hooks = OrderedDict()
                    model._forward_pre_hooks = OrderedDict()
                    for child in model.children():
                        remove_hooks(child)

                remove_hooks(netG)
                trace = torch.jit.trace(netG,
                                        (z_obj, bbox[:, :, :2], attributes, z_im, label.squeeze(dim=-1)))
                torch.jit.save(trace, os.path.join(args.sample_path, args.output_name))
                print(os.path.join(args.sample_path, args.output_name))
            exit()

            # load
            # netG = torch.jit.load(os.path.join(args.sample_path, 'netG.pt'))
            # fake_images, bbox_wh, panoptic_bbox = netG(z_obj, bbox[:, :, :2], attributes, z_im, label.squeeze(dim=-1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco',
                        help='training dataset')
    parser.add_argument('--data_dir', type=str,
                        help='dataset directory')
    parser.add_argument('--set', type=str, default='val',
                        help='dataset part')
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--input_size', type=int, default=256,
                        help='input size of training data. Default: 128')
    parser.add_argument('--filter_mode', type=str, default='LostGAN',
                        help='dataset')
    parser.add_argument('--model_path', type=str, default='../models/lost_gan/plgan_256/model/ckpt_34.pth',
                        help='which epoch to load')
    parser.add_argument('--sample_path', type=str, default='../res_vis/bf_gan/base',
                        help='path to save generated images')
    parser.add_argument('--sample_times', type=int, default=5,
                        help='')
    parser.add_argument('--gt_bb', action='store_true', help='whether to use gt bbox')
    parser.add_argument('--dump_bbox', action='store_true', help='whether to dump pred bbox')
    parser.add_argument('--dump_bbox_dir', type=str, default=None,
                        help='whether to use dumped bbox')
    parser.add_argument('--bbox_dir', type=str, default='./netGv2_coco128.pth',
                        help='pred bbox path')
    parser.add_argument('--dump_input', action='store_true', help='whether to dump input')
    parser.add_argument('--output_name', type=str, help='')
    args = parser.parse_args()
    main(args)
