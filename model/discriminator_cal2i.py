import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .roi_layers import ROIAlign, ROIPool
from utils.util import *
from utils.bilinear import *


def conv2d(in_feat, out_feat, kernel_size=3, stride=1, pad=1, spectral_norm=True):
    conv = nn.Conv2d(in_feat, out_feat, kernel_size, stride, pad)
    if spectral_norm:
        return nn.utils.spectral_norm(conv, eps=1e-4)
    else:
        return conv

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        norm_layer = nn.InstanceNorm2d
        use_bias = True
        # if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
        #     use_bias = norm_layer.func == nn.InstanceNorm2d
        # else:
        #     use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class ResnetDiscriminator128(nn.Module):
    def __init__(self, num_classes=0, input_dim=3, ch=64):
        super(ResnetDiscriminator128, self).__init__()
        self.num_classes = num_classes
        self.instance_threshold = 92

        self.block1 = OptimizedBlock(3, ch, downsample=True)
        self.block2 = ResBlock(ch, ch * 2, downsample=True)
        self.block3 = ResBlock(ch * 2, ch * 4, downsample=True)
        self.block4 = ResBlock(ch * 4, ch * 8, downsample=True)
        self.block5 = ResBlock(ch * 8, ch * 16, downsample=True)
        self.block6 = ResBlock(ch * 16, ch * 16, downsample=False)
        self.l7 = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        # self.l7 = conv2d(ch * 16, 1, 1, 1, 0)
        # self.d_img = NLayerDiscriminator(3)
        self.activation = nn.ReLU()

        self.roi_align = ROIAlign((64, 64), 1.0, int(0))
        self.roi_align_s = ROIAlign((8, 8), 1.0 / 4.0, int(0))
        self.roi_align_l = ROIAlign((8, 8), 1.0 / 8.0, int(0))

        self.block_obj1 = OptimizedBlock(3, ch, downsample=True)
        self.block_obj2 = ResBlock(ch, ch * 2, downsample=True)
        self.block_obj3 = ResBlock(ch * 2, ch * 4, downsample=True)
        self.block_obj4 = ResBlock(ch * 4, ch * 8, downsample=True)
        self.block_obj5 = ResBlock(ch * 8, ch * 16, downsample=False)
        self.l_obj = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.l_y = nn.utils.spectral_norm(nn.Embedding(num_classes, ch * 16))
        # self.l_ac = nn.utils.spectral_norm(nn.Linear(ch * 16, num_classes))

        # self.seg_dis = SegDiscriminator(num_classes, ndf=64, n_layers=3)
        # self.mask_dis = MaskDiscriminator(num_classes, self.instance_threshold, ndf=64, n_layers=4)

    def forward(self, x, y=None, bbox=None, seg=None):
        b = bbox.size(0)
        x_obj = x
        # 128x128
        x = self.block1(x)
        # 64x64
        x1 = self.block2(x)
        # 32x32
        x2 = self.block3(x1)
        # 16x16
        x = self.block4(x2)
        # 8x8
        x = self.block5(x)
        # 4x4
        x = self.block6(x)
        x = self.activation(x)
        x = torch.sum(x, dim=(2, 3))
        out_im = self.l7(x)
        # out_im = self.d_img(x)

        # obj path
        # # seperate different path
        # # obj_feat = self.roi_align(x_obj, bbox)
        # # obj_feat = self.block_obj1(obj_feat)
        # # obj_feat = self.block_obj2(obj_feat)
        # # obj_feat = self.block_obj3(x1)
        # obj_feat = self.block_obj4(x2)
        # obj_feat = self.roi_align_l(obj_feat, bbox)
        # obj_feat = self.block_obj5(obj_feat)
        # obj_feat = self.activation(obj_feat)
        # obj_feat = torch.sum(obj_feat, dim=(2, 3))
        # out_obj = self.l_obj(obj_feat)
        # out_ac = self.l_ac(obj_feat)
        # out_ac = F.cross_entropy(out_ac, y)

        s_idx = ((bbox[:, 3] - bbox[:, 1]) < 64) * ((bbox[:, 4] - bbox[:, 2]) < 64)
        bbox_l, bbox_s = bbox[~s_idx], bbox[s_idx]
        y_l, y_s = y[~s_idx], y[s_idx]
        obj_feat_s = self.block_obj3(x1)
        obj_feat_s = self.block_obj4(obj_feat_s)
        obj_feat_s = self.roi_align_s(obj_feat_s, bbox_s)
        obj_feat_l = self.block_obj4(x2)
        obj_feat_l = self.roi_align_l(obj_feat_l, bbox_l)
        obj_feat = torch.cat([obj_feat_l, obj_feat_s], dim=0)
        y = torch.cat([y_l, y_s], dim=0)
        obj_feat = self.block_obj5(obj_feat)
        obj_feat = self.activation(obj_feat)
        obj_feat = torch.sum(obj_feat, dim=(2, 3))
        out_obj = self.l_obj(obj_feat)
        out_obj = out_obj + torch.sum(self.l_y(y).view(b, -1) * obj_feat.view(b, -1), dim=1, keepdim=True)
        # out_ac = self.l_ac(obj_feat)
        # out_ac = F.cross_entropy(out_ac, y)

        return out_im, out_obj


class ResnetDiscriminator64(nn.Module):
    def __init__(self, num_classes=0, input_dim=3, ch=64):
        super(ResnetDiscriminator64, self).__init__()
        self.num_classes = num_classes

        self.block1 = OptimizedBlock(input_dim, ch, downsample=False)
        self.block2 = ResBlock(ch, ch * 2, downsample=False)
        self.block3 = ResBlock(ch * 2, ch * 4, downsample=True)
        self.block4 = ResBlock(ch * 4, ch * 8, downsample=True)
        self.block5 = ResBlock(ch * 8, ch * 16, downsample=True)
        self.l_im = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.activation = nn.ReLU()

        # object path
        self.roi_align = ROIAlign((8, 8), 1.0 / 2.0, 0)
        self.block_obj4 = ResBlock(ch * 4, ch * 8, downsample=True)
        self.l_obj = nn.utils.spectral_norm(nn.Linear(ch * 8, 1))
        self.l_y = nn.utils.spectral_norm(nn.Embedding(num_classes, ch * 8))
        # self.l_ac = nn.utils.spectral_norm(nn.Linear(ch * 8, num_classes))

        self.init_parameter()

    def forward(self, x, y=None, bbox=None):
        b = bbox.size(0)
        # 64x64
        x = self.block1(x)
        # 64x64
        x = self.block2(x)
        # 32x32
        x1 = self.block3(x)
        # 16x16
        x = self.block4(x1)
        # 8x8
        x = self.block5(x)
        x = self.activation(x)
        x = torch.mean(x, dim=(2, 3))
        out_im = self.l_im(x)

        # obj path
        obj_feat = self.roi_align(x1, bbox)
        obj_feat = self.block_obj4(obj_feat)
        obj_feat = self.activation(obj_feat)
        obj_feat = torch.sum(obj_feat, dim=(2, 3))
        out_obj = self.l_obj(obj_feat)
        out_obj = out_obj + torch.sum(self.l_y(y).view(b, -1) * obj_feat.view(b, -1), dim=1, keepdim=True)
        # out_ac = self.l_ac(obj_feat)
        # out_ac = F.cross_entropy(out_ac, y)

        return out_im, out_obj

    def init_parameter(self):
        for k in self.named_parameters():
            if k[1].dim() > 1:
                torch.nn.init.orthogonal_(k[1])
            if k[0][-4:] == 'bias':
                torch.nn.init.constant_(k[1], 0)


class ResnetDiscriminator256(nn.Module):
    def __init__(self, num_classes=0, input_dim=3, ch=64):
        super(ResnetDiscriminator256, self).__init__()
        self.num_classes = num_classes

        self.block1 = OptimizedBlock(3, ch, downsample=True)
        self.block2 = ResBlock(ch, ch * 2, downsample=True)
        self.block3 = ResBlock(ch * 2, ch * 4, downsample=True)
        self.block4 = ResBlock(ch * 4, ch * 8, downsample=True)
        self.block5 = ResBlock(ch * 8, ch * 8, downsample=True)
        self.block6 = ResBlock(ch * 8, ch * 16, downsample=True)
        self.block7 = ResBlock(ch * 16, ch * 16, downsample=False)
        self.l8 = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.activation = nn.ReLU()

        self.roi_align_s = ROIAlign((8, 8), 1.0 / 8.0, int(0))
        self.roi_align_l = ROIAlign((8, 8), 1.0 / 16.0, int(0))

        self.block_obj4 = ResBlock(ch * 4, ch * 8, downsample=False)
        self.block_obj5 = ResBlock(ch * 8, ch * 8, downsample=False)
        self.block_obj6 = ResBlock(ch * 8, ch * 16, downsample=True)
        self.l_obj = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.l_y = nn.utils.spectral_norm(nn.Embedding(num_classes, ch * 16))
        # self.l_ac = nn.utils.spectral_norm(nn.Linear(ch * 16, num_classes))

    def forward(self, x, y=None, bbox=None):
        b = bbox.size(0)
        # 256x256
        x = self.block1(x)
        # 128x128
        x = self.block2(x)
        # 64x64
        x1 = self.block3(x)
        # 32x32
        x2 = self.block4(x1)
        # 16x16
        x = self.block5(x2)
        # 8x8
        x = self.block6(x)
        # 4x4
        x = self.block7(x)
        x = self.activation(x)
        x = torch.sum(x, dim=(2, 3))
        out_im = self.l8(x)

        # obj path
        # seperate different path
        s_idx = ((bbox[:, 3] - bbox[:, 1]) < 128) * ((bbox[:, 4] - bbox[:, 2]) < 128)
        bbox_l, bbox_s = bbox[~s_idx], bbox[s_idx]
        y_l, y_s = y[~s_idx], y[s_idx]

        obj_feat_s = self.block_obj4(x1)
        obj_feat_s = self.block_obj5(obj_feat_s)
        obj_feat_s = self.roi_align_s(obj_feat_s, bbox_s)

        obj_feat_l = self.block_obj5(x2)
        obj_feat_l = self.roi_align_l(obj_feat_l, bbox_l)

        obj_feat = torch.cat([obj_feat_l, obj_feat_s], dim=0)
        y = torch.cat([y_l, y_s], dim=0)
        obj_feat = self.block_obj6(obj_feat)
        obj_feat = self.activation(obj_feat)
        obj_feat = torch.sum(obj_feat, dim=(2, 3))
        out_obj = self.l_obj(obj_feat)
        # out_ac = self.l_ac(obj_feat)
        # out_ac = F.cross_entropy(out_ac, y)
        out_obj = out_obj + torch.sum(self.l_y(y).view(b, -1) * obj_feat.view(b, -1), dim=1, keepdim=True)

        return out_im, out_obj

class ResnetDiscriminator512(nn.Module):
    def __init__(self, num_classes=0, input_dim=3, ch=64):
        super(ResnetDiscriminator512, self).__init__()
        self.num_classes = num_classes

        self.block1 = OptimizedBlock(3, ch, downsample=True)
        self.block2 = ResBlock(ch, ch * 2, downsample=True)
        self.block3 = ResBlock(ch * 2, ch * 4, downsample=True)
        self.block4 = ResBlock(ch * 4, ch * 4, downsample=True)
        self.block5 = ResBlock(ch * 4, ch * 8, downsample=True)
        self.block6 = ResBlock(ch * 8, ch * 8, downsample=True)
        self.block7 = ResBlock(ch * 8, ch * 16, downsample=True)
        self.block8 = ResBlock(ch * 16, ch * 16, downsample=False)
        self.l8 = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.activation = nn.ReLU()

        self.roi_align_s = ROIAlign((8, 8), 1.0 / 16.0, int(0))
        self.roi_align_l = ROIAlign((8, 8), 1.0 / 32.0, int(0))

        self.block_obj4 = ResBlock(ch * 4, ch * 8, downsample=False)
        self.block_obj5 = ResBlock(ch * 8, ch * 8, downsample=False)
        self.block_obj6 = ResBlock(ch * 8, ch * 16, downsample=False)
        # self.block_obj7 = ResBlock(ch * 8, ch * 16, downsample=True)
        self.l_obj = nn.utils.spectral_norm(nn.Linear(ch * 16, 1))
        self.l_y = nn.utils.spectral_norm(nn.Embedding(num_classes, ch * 16))
        # self.l_ac = nn.utils.spectral_norm(nn.Linear(ch * 16, num_classes))

    def forward(self, x, y=None, bbox=None):
        b = bbox.size(0)
        # 256x256
        x = self.block1(x)
        # 128x128
        x = self.block2(x)
        # 64x64
        x = self.block3(x)
        # 32x32
        x1 = self.block4(x)
        # 16x16
        x2 = self.block5(x1)
        # 8x8
        x = self.block6(x2)
        # 4x4
        x = self.block7(x)
        x = self.block8(x)
        x = self.activation(x)
        x = torch.sum(x, dim=(2, 3))
        out_im = self.l8(x)

        # obj path
        # seperate different path
        s_idx = ((bbox[:, 3] - bbox[:, 1]) < 256) * ((bbox[:, 4] - bbox[:, 2]) < 256)
        bbox_l, bbox_s = bbox[~s_idx], bbox[s_idx]
        y_l, y_s = y[~s_idx], y[s_idx]

        obj_feat_s = self.block_obj4(x1)
        obj_feat_s = self.block_obj5(obj_feat_s)
        obj_feat_s = self.roi_align_s(obj_feat_s, bbox_s)

        obj_feat_l = self.block_obj5(x2)
        obj_feat_l = self.roi_align_l(obj_feat_l, bbox_l)

        obj_feat = torch.cat([obj_feat_l, obj_feat_s], dim=0)
        y = torch.cat([y_l, y_s], dim=0)
        obj_feat = self.block_obj6(obj_feat)
        obj_feat = self.activation(obj_feat)
        obj_feat = torch.sum(obj_feat, dim=(2, 3))
        out_obj = self.l_obj(obj_feat)
        # out_ac = self.l_ac(obj_feat)
        # out_ac = F.cross_entropy(out_ac, y)
        out_obj = out_obj + torch.sum(self.l_y(y).view(b, -1) * obj_feat.view(b, -1), dim=1, keepdim=True)

        return out_im, out_obj


class OptimizedBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, pad=1, downsample=False):
        super(OptimizedBlock, self).__init__()
        self.conv1 = conv2d(in_ch, out_ch, ksize, 1, pad)
        self.conv2 = conv2d(out_ch, out_ch, ksize, 1, pad)
        self.c_sc = conv2d(in_ch, out_ch, 1, 1, 0)
        self.activation = nn.ReLU()
        self.downsample = downsample

    def forward(self, in_feat):
        x = in_feat
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x + self.shortcut(in_feat)

    def shortcut(self, x):
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return self.c_sc(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, pad=1, downsample=False):
        super(ResBlock, self).__init__()
        self.conv1 = conv2d(in_ch, out_ch, ksize, 1, pad)
        self.conv2 = conv2d(out_ch, out_ch, ksize, 1, pad)
        self.activation = nn.ReLU()
        self.downsample = downsample
        self.learnable_sc = (in_ch != out_ch) or downsample
        if self.learnable_sc:
            self.c_sc = conv2d(in_ch, out_ch, 1, 1, 0)

    def residual(self, in_feat):
        x = in_feat
        x = self.conv1(self.activation(x))
        x = self.conv2(self.activation(x))
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                x = F.avg_pool2d(x, 2)
        return x

    def forward(self, in_feat):
        return self.residual(in_feat) + self.shortcut(in_feat)


class CombineDiscriminator512(nn.Module):
    def __init__(self, num_classes=81):
        super(CombineDiscriminator512, self).__init__()
        self.obD = ResnetDiscriminator512(num_classes=num_classes, input_dim=3)

    def forward(self, images, bbox, label, mask=None):
        idx = torch.arange(start=0, end=images.size(0),
                           device=images.device).view(images.size(0),
                                                      1, 1).expand(-1, bbox.size(1), -1).float()
        # bbox[:, :, 2] = bbox[:, :, 2] + bbox[:, :, 0]
        # bbox[:, :, 3] = bbox[:, :, 3] + bbox[:, :, 1]
        bbox = torch.cat([bbox[:, :, :1], bbox[:, :, 1:2],
                          bbox[:, :, 2:3] + bbox[:, :, :1],
                          bbox[:, :, 3:] + bbox[:, :, 1:2]], 2)
        bbox = bbox * images.size(2)
        bbox = torch.cat((idx, bbox.float()), dim=2)
        bbox = bbox.view(-1, 5)
        label = label.view(-1)

        idx = (label != 0).nonzero().view(-1)
        bbox = bbox[idx]
        label = label[idx]
        d_out_img, d_out_obj = self.obD(images, label, bbox)
        return d_out_img, d_out_obj

class CombineDiscriminator256(nn.Module):
    def __init__(self, num_classes=81):
        super(CombineDiscriminator256, self).__init__()
        self.obD = ResnetDiscriminator256(num_classes=num_classes, input_dim=3)

    def forward(self, images, bbox, label, mask=None):
        idx = torch.arange(start=0, end=images.size(0),
                           device=images.device).view(images.size(0),
                                                      1, 1).expand(-1, bbox.size(1), -1).float()
        # bbox[:, :, 2] = bbox[:, :, 2] + bbox[:, :, 0]
        # bbox[:, :, 3] = bbox[:, :, 3] + bbox[:, :, 1]
        bbox = torch.cat([bbox[:, :, :1], bbox[:, :, 1:2],
                          bbox[:, :, 2:3] + bbox[:, :, :1],
                          bbox[:, :, 3:] + bbox[:, :, 1:2]], 2)
        bbox = bbox * images.size(2)
        bbox = torch.cat((idx, bbox.float()), dim=2)
        bbox = bbox.view(-1, 5)
        label = label.view(-1)

        idx = (label != 0).nonzero().view(-1)
        bbox = bbox[idx]
        label = label[idx]
        d_out_img, d_out_obj = self.obD(images, label, bbox)
        return d_out_img, d_out_obj


class CombineDiscriminator128(nn.Module):
    def __init__(self, num_classes=81):
        super(CombineDiscriminator128, self).__init__()
        self.obD = ResnetDiscriminator128(num_classes=num_classes, input_dim=3)

    def forward(self, images, bbox, label, mask=None):
        idx = torch.arange(start=0, end=images.size(0),
                           device=images.device).view(images.size(0),
                                                      1, 1).expand(-1, bbox.size(1), -1).float()
        # bbox[:, :, 2] = bbox[:, :, 2] + bbox[:, :, 0]
        # bbox[:, :, 3] = bbox[:, :, 3] + bbox[:, :, 1]
        bbox = torch.cat([bbox[:, :, :1], bbox[:, :, 1:2],
                          bbox[:, :, 2:3] + bbox[:, :, :1],
                          bbox[:, :, 3:] + bbox[:, :, 1:2]], 2)
        bbox = bbox * images.size(2)
        bbox = torch.cat((idx, bbox.float()), dim=2)
        bbox = bbox.view(-1, 5)
        label = label.view(-1)

        idx = (label != 0).nonzero().view(-1)
        bbox = bbox[idx]
        label = label[idx]
        d_out_img, d_out_obj = self.obD(images, label, bbox)
        return d_out_img, d_out_obj


class CombineDiscriminator64(nn.Module):
    def __init__(self, num_classes=81):
        super(CombineDiscriminator64, self).__init__()
        self.obD = ResnetDiscriminator64(num_classes=num_classes, input_dim=3)

    def forward(self, images, bbox, label, mask=None):
        idx = torch.arange(start=0, end=images.size(0),
                           device=images.device).view(images.size(0),
                                                      1, 1).expand(-1, bbox.size(1), -1).float()
        # bbox[:, :, 2] = bbox[:, :, 2] + bbox[:, :, 0]
        # bbox[:, :, 3] = bbox[:, :, 3] + bbox[:, :, 1]
        bbox = torch.cat([bbox[:, :, :1], bbox[:, :, 1:2],
                          bbox[:, :, 2:3] + bbox[:, :, :1],
                          bbox[:, :, 3:] + bbox[:, :, 1:2]], 2)
        bbox = bbox * images.size(2)
        bbox = torch.cat((idx, bbox.float()), dim=2)
        bbox = bbox.view(-1, 5)
        label = label.view(-1)

        idx = (label != 0).nonzero().view(-1)
        bbox = bbox[idx]
        label = label[idx]
        d_out_img, d_out_obj = self.obD(images, label, bbox)
        return d_out_img, d_out_obj


class SegDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, use_sigmoid=False):
        super(SegDiscriminator, self).__init__()
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        for n in range(len(sequence)):
            setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        res = [input]
        for n in range(self.n_layers + 2):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]

class MaskDiscriminator(nn.Module):
    def __init__(self, num_classes, instance_threshold, ndf=64, n_layers=4, norm_layer=nn.InstanceNorm2d):
        super(MaskDiscriminator, self).__init__()
        self.n_layers = n_layers
        self.num_classes = num_classes
        self.instance_threshold = instance_threshold

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(1, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        # nf_prev = nf
        # nf = min(nf * 2, 512)
        # sequence += [[
        #     nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
        #     norm_layer(nf),
        #     nn.LeakyReLU(0.2, True)
        # ]]

        # sequence += [[nn.Conv2d(nf, nf, kernel_size=kw, stride=1, padding=padw)]]

        for n in range(len(sequence)):
            setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))

        self.fc = nn.utils.spectral_norm(nn.Linear(nf, 128))
        # self.l_y = nn.utils.spectral_norm(nn.Embedding(num_classes, 128))
        # self.l_y = Variable(torch.Tensor(num_classes, 128), requires_grad=True)
        self.l_y = nn.Parameter(torch.Tensor(num_classes, 128))
        self.l_y.data.uniform_()
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)

    def forward(self, bbox, y):
        b, o, w, h = bbox.size(0), bbox.size(1), bbox.size(2), bbox.size(3)
        bbox = bbox.view(b*o, 1, w, h)
        y = y.view(b * o)
        for n in range(self.n_layers):
            model = getattr(self, 'model' + str(n))
            bbox = model(bbox)
        obj_feat = torch.sum(bbox, dim=(2, 3))
        obj_feat = obj_feat
        obj_feat = self.fc(obj_feat)

        loss = self.cos(self.l_y.unsqueeze(0), obj_feat.unsqueeze(1))
        # print('===>', torch.sum(loss).item(), torch.sum(self.l_y).item(), torch.sum(obj_feat).item())
        loss = F.softmax(loss, 1)

        loss = loss[torch.arange(b * o), y[torch.arange(b * o)]]
        y_instance_mask = (y.view(b * o, 1) < self.instance_threshold).float() * (y.view(b * o, 1) > 0).float()
        loss = torch.sum(loss*y_instance_mask)/(torch.sum(y_instance_mask)+1e-6)

        return loss
