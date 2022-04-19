import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .norm_module import *
from .mask_regression import *
from .sync_batchnorm import SynchronizedBatchNorm2d
from .roi_layers import ROIAlign, ROIPool

class CAI2LGenerator256(nn.Module):
    def __init__(self, ch=64, z_dim=128, num_classes=10, output_dim=3, instance_threshold=92):
        super(CAI2LGenerator256, self).__init__()
        self.num_classes = num_classes
        self.map_size = 64
        self.instance_threshold = instance_threshold
        self.mask_embedding = nn.Embedding(num_classes, 180)
        self.label_embedding = nn.Embedding(num_classes, 180)

        num_w = 128+180
        self.ref = False
        self.fc = nn.utils.spectral_norm(nn.Linear(z_dim, 4*4*16*ch))
        self.res1 = ResBlock(ch*16, ch*16, upsample=True, num_w=num_w, predict_mask=self.ref, psp_module=False,
                             Norm=SpatialAdaptiveSynBatchNorm2d)
        self.res2 = ResBlock(ch*16, ch*8, upsample=True, num_w=num_w, predict_mask=self.ref, psp_module=False,
                             Norm=SpatialAdaptiveSynBatchNorm2d)
        self.res3 = ResBlock(ch*8, ch*8, upsample=True, num_w=num_w, predict_mask=self.ref, psp_module=False,
                             Norm=SpatialAdaptiveSynBatchNorm2d)
        self.res4 = ResBlock(ch*8, ch*4, upsample=True, num_w=num_w, predict_mask=self.ref, psp_module=False,
                             Norm=SpatialAdaptiveSynBatchNorm2d)
        self.res5 = ResBlock(ch*4, ch*2, upsample=True, num_w=num_w, predict_mask=self.ref, psp_module=False,
                             Norm=SpatialAdaptiveSynBatchNorm2d)
        self.res6 = ResBlock(ch * 2, ch * 1, upsample=True, num_w=num_w, predict_mask=False, psp_module=False,
                             Norm=SpatialAdaptiveSynBatchNorm2d)
        self.final = nn.Sequential(SynchronizedBatchNorm2d(ch),
                                   nn.ReLU(),
                                   conv2d(ch, output_dim, 3, 1, 1, spectral_norm=True),
                                   nn.Tanh())

        self.m_cgf2 = ISA(num_classes, in_ch=ch * 16, radius=3, norm=SynchronizedBatchNorm2d)
        self.m_cgf3 = ISA(num_classes, in_ch=ch * 8, radius=3, norm=SynchronizedBatchNorm2d)
        self.m_cgf4 = ISA(num_classes, in_ch=ch * 8, radius=3, norm=SynchronizedBatchNorm2d)
        self.m_cgf5 = ISA(num_classes, in_ch=ch * 4, radius=3, norm=SynchronizedBatchNorm2d)
        self.m_cgf6 = ISA(num_classes, in_ch=ch * 2, radius=3, norm=SynchronizedBatchNorm2d)

        # seg
        self.seg_embedding = nn.Embedding(num_classes, 180)
        self.fc_layout = nn.utils.spectral_norm(nn.Linear(z_dim, 4 * 4 * 16 * ch))
        self.res1_layout = ResBlock(ch * 16, ch * 16, upsample=True, num_w=num_classes, Norm=SPADENorm2d)
        self.res2_layout = ResBlock(ch * 16, ch * 8, upsample=True, num_w=num_classes, Norm=SPADENorm2d)
        self.res3_layout = ResBlock(ch * 8, ch * 4, upsample=True, num_w=num_classes, Norm=SPADENorm2d)
        self.res4_layout = ResBlock(ch * 4, ch * 4, upsample=True, num_w=num_classes, Norm=VNorm2d)
        self.final_layout = nn.Sequential(SynchronizedBatchNorm2d(ch * 4),
                                          nn.ReLU(),
                                          conv2d(ch * 4, num_classes, 3, 1, 1))
        # bbox
        self.bbox_embedding = nn.Embedding(num_classes, 180)
        self.bbox_fc1 = nn.Linear(180+35, 128)
        self.bbox_activation1 = nn.ReLU()
        self.bbox_fc2 = nn.Linear(128, 64)
        self.bbox_activation2 = nn.ReLU()
        self.bbox_fc3 = nn.Linear(64, 2)

        # mapping function
        mapping = list()
        self.mapping = nn.Sequential(*mapping)

        self.mask_regress = MaskRegressNet16(num_w)
        # self.mask_regress = MaskRegressNetv2(num_w)
        self.init_parameter()
        self.roi_align = ROIAlign((32, 32), 1.0, int(0))

    def forward(self, z, bbox=None, bbox_lt=None, attr=None, z_im=None, y=None, vis=False):
        b, o = z.size(0), z.size(1)
        if z_im is None:
            z_im = torch.randn((b, 128), device=z.device)

        # embedding
        if y.dim() == 3:
            _, _, num_label = y.size()
            label_embedding = []
            for idx in range(num_label):
                label_embedding.append(self.label_embedding[idx](y[:, :, idx]))
            label_embedding = torch.cat(label_embedding, dim=-1)
        else:
            label_embedding = self.label_embedding(y)
        z = z.view(b * o, -1)
        label_embedding = label_embedding.view(b * o, -1)
        latent_vector = torch.cat((z, label_embedding), dim=1).view(b, o, -1)
        w = self.mapping(latent_vector.view(b * o, -1))
        bbox_in = bbox

        # instance layouts
        # generate bbox width and height
        w_bbox = self.bbox_embedding(y)
        bbox_wh = self.bbox_fc1(torch.cat([w_bbox.view(-1, 180), attr.view(-1, 35)], 1))
        bbox_wh = self.bbox_activation1(bbox_wh)
        bbox_wh = self.bbox_fc2(bbox_wh)
        bbox_wh = self.bbox_activation1(bbox_wh)
        bbox_wh = self.bbox_fc3(bbox_wh)
        bbox_wh = torch.clamp(bbox_wh.view(b, o, 2), 0.001, 1.0)
        # generate mask
        if self.training:
            center = bbox[:, :, :2] + 0.5 * bbox[:, :, 2:]
            bbox, mask = self.mask_regress(w, bbox)
        else:
            if bbox is not None:
                center = bbox[:, :, :2] + 0.5 * bbox[:, :, 2:]
                bbox, mask = self.mask_regress(w, bbox)
            else:
                center = bbox_lt + 0.5 * bbox_wh
                bbox, mask = self.mask_regress(w, torch.cat([bbox_lt, bbox_wh], 2))
                bbox_in = torch.cat([bbox_lt, bbox_wh], 2)

        # stuff layouts
        # coarse stuff
        size_attribute, location_attribute = attr[:, :, :10], attr[:, :, 10:]
        size_index = torch.argmax(size_attribute, dim=2)
        size_list = torch.FloatTensor((np.arange(10)+1)/10).cuda()
        wh = torch.index_select(size_list, 0, size_index.view(-1)).view(b, o, 1)
        center = center - 0.5 * wh
        coarse_bbox = torch.cat([center, wh, wh], 2)
        coarse_masks = masks_to_layout(
            coarse_bbox, torch.ones([b, o, 16, 16]).cuda(), self.map_size).view(b, o, self.map_size, self.map_size)
        coarse_layout = torch.zeros([b, self.num_classes, self.map_size, self.map_size]).cuda()
        for idx in range(b):
            coarse_layout[idx].index_add_(0, y[idx], coarse_masks[idx])
        coarse_layout[coarse_layout > 1] = 1
        # refine stuff
        w_seg = self.seg_embedding(y)
        seg = self.fc_layout(z_im).view(b, -1, 4, 4)
        seg = self.res1_layout(seg, w_seg, coarse_layout)
        seg = self.res2_layout(seg, w_seg, coarse_layout)
        seg = self.res3_layout(seg, w_seg, coarse_layout)
        seg = self.res4_layout(seg, w_seg, coarse_layout)
        seg = self.final_layout(seg)
        y_mask = (y > 0).float()
        y_ = (y.float() * y_mask).long()
        objs_one_hot = torch.zeros(b, self.num_classes).cuda()
        for idx in range(b):
            objs_one_hot[idx, y_[idx].long()] = 1
        seg_out = self.indexed_softmax(seg, objs_one_hot.view(b, self.num_classes, 1, 1), 1)
        # postprocess
        y_stuff_mask = (y>=self.instance_threshold).float() * (y>0).float()
        objs_one_hot = torch.zeros(b, self.num_classes).cuda()
        y_ = (y.float()*y_stuff_mask).long()
        for idx in range(b):
            objs_one_hot[idx, y_[idx].long()] = 1
            objs_one_hot[idx, 0] = 0
        seg_index = self.indexed_softmax(seg, objs_one_hot.view(b, self.num_classes, 1, 1), 1)
        seg = torch.cat([seg_index[i, y[i]].unsqueeze(0) for i in range(b)], 0)

        # image generation
        y_instance_mask = (y < self.instance_threshold).float() * (y > 0).float()
        # 4x4
        x = self.fc(z_im).view(b, -1, 4, 4)
        bbox_mask_ = bbox_mask(z, bbox_in, 64, 64)
        # 8x8
        instance_mask = (torch.sum(y_instance_mask.view(b, o, 1, 1) * bbox.detach(), 1, keepdim=True) > 0.5).float()
        panoptic_bbox1 = y_instance_mask.view(b, o, 1, 1) * bbox * instance_mask + seg * (1 - instance_mask)
        x = self.res1(x, w, panoptic_bbox1)
        # 16x16
        stage_bbox, panoptic_bbox2 = self.m_cgf2(x, bbox, bbox_mask_, b, o, seg, y_instance_mask)
        x = self.res2(x, w, panoptic_bbox2)
        # 32x32
        stage_bbox, panoptic_bbox3 = self.m_cgf3(x, bbox, bbox_mask_, b, o, seg, y_instance_mask)
        x = self.res3(x, w, panoptic_bbox3)
        # 64x64
        stage_bbox, panoptic_bbox4 = self.m_cgf4(x, bbox, bbox_mask_, b, o, seg, y_instance_mask)
        x = self.res4(x, w, panoptic_bbox4)
        # 128x128
        stage_bbox, panoptic_bbox5 = self.m_cgf5(x, stage_bbox, bbox_mask_, b, o, seg, y_instance_mask)
        x = self.res5(x, w, panoptic_bbox5)
        # 256x256
        stage_bbox, panoptic_bbox6 = self.m_cgf6(x, stage_bbox, bbox_mask_, b, o, seg, y_instance_mask)
        x = self.res6(x, w, panoptic_bbox6)
        # to RGB
        x = self.final(x)

        if vis:
            res_vis = {
                'seg_out': seg_out,
                'seg': seg,
                'stage_bbox': stage_bbox,
                'mask': panoptic_bbox6,
                'mask1': panoptic_bbox1,
                'mask2': panoptic_bbox2,
                'mask3': panoptic_bbox3,
                'mask4': panoptic_bbox4,
                'mask5': panoptic_bbox5,
                'mask6': panoptic_bbox6,
                'bbox': bbox,
                'coarse_layout': coarse_layout,
                'instance_mask': instance_mask,
            }
            return x, bbox_wh, mask, res_vis
        return x, bbox_wh, panoptic_bbox6

    def indexed_softmax(self, x, index, axis):
        index = index.float()
        x = x - self.min(x, axis, keepdim=True)
        x_max = self.max(x*index, axis, keepdim=True)
        x = (x - x_max)*index
        softmax = (torch.exp(x)*index)/(torch.sum(torch.exp(x)*index, axis, keepdim=True)+1e-6)
        return softmax

    def min(self, x, axis, keepdim):
        value, min_indices = torch.min(x, axis, keepdim=keepdim)
        return value

    def max(self, x, axis, keepdim):
        value, min_indices = torch.max(x, axis, keepdim=keepdim)
        return value

    def update_mask(self, alpha1, x, stage_mask, bbox, bbox_mask_, y, b, o, seg, y_instance_mask):
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha1 = torch.gather(self.sigmoid(alpha1).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        bbox = F.interpolate(bbox, size=(hh, ww), mode='bilinear') * (1 - alpha1) + seman_bbox * alpha1
        instance_mask = (torch.sum(y_instance_mask.view(b, o, 1, 1) * bbox.detach(), 1, keepdim=True) > 0.2).float()
        seg = F.interpolate(seg, size=(hh, ww), mode='bilinear')
        panoptic_bbox = y_instance_mask.view(b, o, 1, 1) * bbox * instance_mask + seg * (1 - instance_mask)
        return bbox, panoptic_bbox

    def init_parameter(self):
        for k in self.named_parameters():
            if k[1].dim() > 1:
                torch.nn.init.orthogonal_(k[1])
            if k[0][-4:] == 'bias':
                torch.nn.init.constant_(k[1], 0)

class LostGANGenerator256(nn.Module):
    def __init__(self, ch=64, z_dim=128, num_classes=10, output_dim=3, instance_threshold=92):
        super(LostGANGenerator256, self).__init__()
        self.num_classes = num_classes
        self.map_size = 64
        self.instance_threshold = instance_threshold
        self.mask_embedding = nn.Embedding(num_classes, 180)
        self.label_embedding = nn.Embedding(num_classes, 180)

        num_w = 128+180
        self.ref = False
        self.fc = nn.utils.spectral_norm(nn.Linear(z_dim, 4*4*16*ch))
        self.res1 = ResBlock(ch*16, ch*16, upsample=True, num_w=num_w, predict_mask=self.ref, psp_module=False,
                             Norm=SpatialAdaptiveSynBatchNorm2d)
        self.res2 = ResBlock(ch*16, ch*8, upsample=True, num_w=num_w, predict_mask=self.ref, psp_module=False,
                             Norm=SpatialAdaptiveSynBatchNorm2d)
        self.res3 = ResBlock(ch*8, ch*8, upsample=True, num_w=num_w, predict_mask=self.ref, psp_module=False,
                             Norm=SpatialAdaptiveSynBatchNorm2d)
        self.res4 = ResBlock(ch*8, ch*4, upsample=True, num_w=num_w, predict_mask=self.ref, psp_module=False,
                             Norm=SpatialAdaptiveSynBatchNorm2d)
        self.res5 = ResBlock(ch*4, ch*2, upsample=True, num_w=num_w, predict_mask=self.ref, psp_module=False,
                             Norm=SpatialAdaptiveSynBatchNorm2d)
        self.res6 = ResBlock(ch * 2, ch * 1, upsample=True, num_w=num_w, predict_mask=False, psp_module=False,
                             Norm=SpatialAdaptiveSynBatchNorm2d)
        self.final = nn.Sequential(SynchronizedBatchNorm2d(ch),
                                   nn.ReLU(),
                                   conv2d(ch, output_dim, 3, 1, 1, spectral_norm=True),
                                   nn.Tanh())

        self.m_cgf2 = ISA(num_classes, in_ch=ch * 16, radius=3, norm=SynchronizedBatchNorm2d)
        self.m_cgf3 = ISA(num_classes, in_ch=ch * 8, radius=3, norm=SynchronizedBatchNorm2d)
        self.m_cgf4 = ISA(num_classes, in_ch=ch * 8, radius=3, norm=SynchronizedBatchNorm2d)
        self.m_cgf5 = ISA(num_classes, in_ch=ch * 4, radius=3, norm=SynchronizedBatchNorm2d)
        self.m_cgf6 = ISA(num_classes, in_ch=ch * 2, radius=3, norm=SynchronizedBatchNorm2d)

        # seg
        self.seg_embedding = nn.Embedding(num_classes, 180)
        self.fc_layout = nn.utils.spectral_norm(nn.Linear(z_dim, 4 * 4 * 16 * ch))
        self.res1_layout = ResBlock(ch * 16, ch * 16, upsample=True, num_w=num_classes, Norm=SPADENorm2d)
        self.res2_layout = ResBlock(ch * 16, ch * 8, upsample=True, num_w=num_classes, Norm=SPADENorm2d)
        self.res3_layout = ResBlock(ch * 8, ch * 4, upsample=True, num_w=num_classes, Norm=SPADENorm2d)
        self.res4_layout = ResBlock(ch * 4, ch * 4, upsample=True, num_w=num_classes, Norm=VNorm2d)
        self.final_layout = nn.Sequential(SynchronizedBatchNorm2d(ch * 4),
                                          nn.ReLU(),
                                          conv2d(ch * 4, num_classes, 3, 1, 1))
        # bbox
        self.bbox_embedding = nn.Embedding(num_classes, 180)
        self.bbox_fc1 = nn.Linear(180+35, 128)
        self.bbox_activation1 = nn.ReLU()
        self.bbox_fc2 = nn.Linear(128, 64)
        self.bbox_activation2 = nn.ReLU()
        self.bbox_fc3 = nn.Linear(64, 2)

        # mapping function
        mapping = list()
        self.mapping = nn.Sequential(*mapping)

        self.mask_regress = MaskRegressNet16(num_w)
        self.init_parameter()
        self.roi_align = ROIAlign((32, 32), 1.0, int(0))

    def forward(self, z, bbox=None, bbox_lt=None, attr=None, z_im=None, y=None, vis=False):
        b, o = z.size(0), z.size(1)
        # embedding
        if y.dim() == 3:
            _, _, num_label = y.size()
            label_embedding = []
            for idx in range(num_label):
                label_embedding.append(self.label_embedding[idx](y[:, :, idx]))
            label_embedding = torch.cat(label_embedding, dim=-1)
        else:
            label_embedding = self.label_embedding(y)
        z = z.view(b * o, -1)
        label_embedding = label_embedding.view(b * o, -1)
        latent_vector = torch.cat((z, label_embedding), dim=1).view(b, o, -1)
        w = self.mapping(latent_vector.view(b * o, -1))
        bbox_in = bbox

        # instance layouts
        # generate bbox width and height
        w_bbox = self.bbox_embedding(y)
        bbox_wh = self.bbox_fc1(torch.cat([w_bbox.view(-1, 180), attr.view(-1, 35)], 1))
        bbox_wh = self.bbox_activation1(bbox_wh)
        bbox_wh = self.bbox_fc2(bbox_wh)
        bbox_wh = self.bbox_activation1(bbox_wh)
        bbox_wh = self.bbox_fc3(bbox_wh)
        bbox_wh = torch.clamp(bbox_wh.view(b, o, 2), 0.001, 1.0)
        # generate mask
        if self.training:
            center = bbox[:, :, :2] + 0.5 * bbox[:, :, 2:]
            bbox, mask = self.mask_regress(w, bbox)
        else:
            if bbox is not None:
                center = bbox[:, :, :2] + 0.5 * bbox[:, :, 2:]
                bbox, mask = self.mask_regress(w, bbox)
            else:
                center = bbox_lt + 0.5 * bbox_wh
                bbox, mask = self.mask_regress(w, torch.cat([bbox_lt, bbox_wh], 2))
                bbox_in = torch.cat([bbox_lt, bbox_wh], 2)

        if z_im is None:
            z_im = torch.randn((b, 128), device=z.device)

        # stuff layouts
        # coarse stuff
        size_attribute, location_attribute = attr[:, :, :10], attr[:, :, 10:]
        size_index = torch.argmax(size_attribute, dim=2)
        size_list = torch.FloatTensor((np.arange(10)+1)/10).cuda()
        wh = torch.index_select(size_list, 0, size_index.view(-1)).view(b, o, 1)
        center = center - 0.5 * wh
        coarse_bbox = torch.cat([center, wh, wh], 2)
        coarse_masks = masks_to_layout(
            coarse_bbox, torch.ones([b, o, 16, 16]).cuda(), self.map_size).view(b, o, self.map_size, self.map_size)
        coarse_layout = torch.zeros([b, self.num_classes, self.map_size, self.map_size]).cuda()
        for idx in range(b):
            coarse_layout[idx].index_add_(0, y[idx], coarse_masks[idx])
        coarse_layout[coarse_layout > 1] = 1
        # refine stuff
        w_seg = self.seg_embedding(y)
        seg = self.fc_layout(z_im).view(b, -1, 4, 4)
        seg = self.res1_layout(seg, w_seg, coarse_layout)
        seg = self.res2_layout(seg, w_seg, coarse_layout)
        seg = self.res3_layout(seg, w_seg, coarse_layout)
        seg = self.res4_layout(seg, w_seg, coarse_layout)
        seg = self.final_layout(seg)
        y_mask = (y > 0).float()
        y_ = (y.float() * y_mask).long()
        objs_one_hot = torch.zeros(b, self.num_classes).cuda()
        for idx in range(b):
            objs_one_hot[idx, y_[idx].long()] = 1
        seg_out = self.indexed_softmax(seg, objs_one_hot.view(b, self.num_classes, 1, 1), 1)
        # postprocess
        y_stuff_mask = (y>=self.instance_threshold).float() * (y>0).float()
        objs_one_hot = torch.zeros(b, self.num_classes).cuda()
        y_ = (y.float()*y_stuff_mask).long()
        for idx in range(b):
            objs_one_hot[idx, y_[idx].long()] = 1
            objs_one_hot[idx, 0] = 0
        seg_index = self.indexed_softmax(seg, objs_one_hot.view(b, self.num_classes, 1, 1), 1)
        seg = torch.cat([seg_index[i, y[i]].unsqueeze(0) for i in range(b)], 0)

        # image generation
        y_instance_mask = (y < self.instance_threshold).float() * (y > 0).float()
        # 4x4
        x = self.fc(z_im).view(b, -1, 4, 4)
        bbox_mask_ = bbox_mask(z, bbox_in, 64, 64)
        # 8x8
        instance_mask = (torch.sum(y_instance_mask.view(b, o, 1, 1) * bbox.detach(), 1, keepdim=True) > 0.5).float()
        panoptic_bbox1 = y_instance_mask.view(b, o, 1, 1) * bbox * instance_mask + seg * (1 - instance_mask)
        x = self.res1(x, w, panoptic_bbox1)
        # 16x16
        stage_bbox, panoptic_bbox2 = self.m_cgf2(x, bbox, bbox_mask_, b, o, seg, y_instance_mask)
        x = self.res2(x, w, panoptic_bbox2)
        # 32x32
        stage_bbox, panoptic_bbox3 = self.m_cgf3(x, bbox, bbox_mask_, b, o, seg, y_instance_mask)
        x = self.res3(x, w, panoptic_bbox3)
        # 64x64
        stage_bbox, panoptic_bbox4 = self.m_cgf4(x, bbox, bbox_mask_, b, o, seg, y_instance_mask)
        x = self.res4(x, w, panoptic_bbox4)
        # 128x128
        stage_bbox, panoptic_bbox5 = self.m_cgf5(x, stage_bbox, bbox_mask_, b, o, seg, y_instance_mask)
        x = self.res5(x, w, panoptic_bbox5)
        # 256x256
        stage_bbox, panoptic_bbox6 = self.m_cgf6(x, stage_bbox, bbox_mask_, b, o, seg, y_instance_mask)
        x = self.res6(x, w, panoptic_bbox6)
        # to RGB
        x = self.final(x)

        if vis:
            res_vis = {
                'seg_out': seg_out,
                'seg': seg,
                'stage_bbox': stage_bbox,
                'mask': panoptic_bbox6,
                'mask1': panoptic_bbox1,
                'mask2': panoptic_bbox2,
                'mask3': panoptic_bbox3,
                'mask4': panoptic_bbox4,
                'mask5': panoptic_bbox5,
                'mask6': panoptic_bbox6,
                'bbox': bbox,
                'coarse_layout': coarse_layout,
                'instance_mask': instance_mask,
            }
            return x, bbox_wh, mask, res_vis
        return x, bbox_wh, panoptic_bbox6

    def indexed_softmax(self, x, index, axis):
        index = index.float()
        x = x - self.min(x, axis, keepdim=True)
        x_max = self.max(x*index, axis, keepdim=True)
        x = (x - x_max)*index
        softmax = (torch.exp(x)*index)/(torch.sum(torch.exp(x)*index, axis, keepdim=True)+1e-6)
        return softmax

    def min(self, x, axis, keepdim):
        value, min_indices = torch.min(x, axis, keepdim=keepdim)
        return value

    def max(self, x, axis, keepdim):
        value, min_indices = torch.max(x, axis, keepdim=keepdim)
        return value

    def update_mask(self, alpha1, x, stage_mask, bbox, bbox_mask_, y, b, o, seg, y_instance_mask):
        hh, ww = x.size(2), x.size(3)
        seman_bbox = batched_index_select(stage_mask, dim=1, index=y.view(b, o, 1, 1))
        seman_bbox = torch.sigmoid(seman_bbox) * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        alpha1 = torch.gather(self.sigmoid(alpha1).expand(b, -1, -1), dim=1, index=y.view(b, o, 1)).unsqueeze(-1)
        bbox = F.interpolate(bbox, size=(hh, ww), mode='bilinear') * (1 - alpha1) + seman_bbox * alpha1
        instance_mask = (torch.sum(y_instance_mask.view(b, o, 1, 1) * bbox.detach(), 1, keepdim=True) > 0.2).float()
        seg = F.interpolate(seg, size=(hh, ww), mode='bilinear')
        panoptic_bbox = y_instance_mask.view(b, o, 1, 1) * bbox * instance_mask + seg * (1 - instance_mask)
        return bbox, panoptic_bbox

    def init_parameter(self):
        for k in self.named_parameters():
            if k[1].dim() > 1:
                torch.nn.init.orthogonal_(k[1])
            if k[0][-4:] == 'bias':
                torch.nn.init.constant_(k[1], 0)

class LostGANGenerator256Freeze(LostGANGenerator256):
    def forward(self, z_obj, bbox, attributes, z_im, label):
        return super(LostGANGenerator256Freeze, self).forward(z_obj, bbox_lt=bbox, attr=attributes, z_im=z_im, y=label, vis=False)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1, upsample=False,
                 num_w=128, Norm=SpatialAdaptiveSynBatchNorm2d,
                 predict_mask=False, psp_module=False, num_class=184):
        super(ResBlock, self).__init__()
        self.upsample = upsample
        self.h_ch = h_ch if h_ch else out_ch
        self.conv1 = conv2d(in_ch, self.h_ch, ksize, pad=pad)
        self.conv2 = conv2d(self.h_ch, out_ch, ksize, pad=pad)
        self.b1 = Norm(in_ch, num_w=num_w, batchnorm_func=SynchronizedBatchNorm2d)
        self.b2 = Norm(self.h_ch, num_w=num_w, batchnorm_func=SynchronizedBatchNorm2d)

        self.predict_mask = predict_mask
        if self.predict_mask:
            if psp_module:
                self.conv_mask = nn.Sequential(PSPModule(out_ch, 100),
                                               nn.Conv2d(100, num_class, kernel_size=1))
            else:
                self.conv_mask = nn.Sequential(nn.Conv2d(out_ch, 100, 3, 1, 1),
                                               SynchronizedBatchNorm2d(100),
                                               nn.ReLU(),
                                               nn.Conv2d(100, num_class, 1, 1, 0, bias=True))

        self.learnable_sc = in_ch != out_ch or upsample
        if self.learnable_sc:
            self.c_sc = conv2d(in_ch, out_ch, 1, 1, 0)
        self.activation = nn.ReLU()

    def residual(self, in_feat, w, bbox, vis=False):
        '''
        :param in_feat: z latent variable
        :param w: object latent variable
        :param bbox: object mask
        :return:
        '''
        x = in_feat
        x = self.b1(x, w, bbox)
        x = self.activation(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        if vis:
            x, weight, bias = self.b2(x, w, bbox, vis)
        else:
            x = self.b2(x, w, bbox)
        x = self.activation(x)
        x = self.conv2(x)
        if vis:
            return x, weight, bias
        return x

    def shortcut(self, x):
        if self.learnable_sc:
            if self.upsample:
                x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = self.c_sc(x)
        return x

    def forward(self, in_feat, w, bbox, vis=False):
        if vis:
            x, weight, bias = self.residual(in_feat, w, bbox, vis)
            out_feat = x + self.shortcut(in_feat)
        else:
            out_feat = self.residual(in_feat, w, bbox) + self.shortcut(in_feat)
        if self.predict_mask:
            mask = self.conv_mask(out_feat)
            return out_feat, mask
        else:
            if vis:
                return out_feat, weight, bias
            return out_feat


def conv2d(in_feat, out_feat, kernel_size=3, stride=1, pad=1, spectral_norm=True):
    conv = nn.Conv2d(in_feat, out_feat, kernel_size, stride, pad)
    if spectral_norm:
        return nn.utils.spectral_norm(conv, eps=1e-4)
    else:
        return conv

def batched_index_select(input, dim, index):
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            SynchronizedBatchNorm2d(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1)
            )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_features)
        return nn.Sequential(prior, conv, bn, nn.ReLU())

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle

def bbox_mask(x, bbox, H, W):
    b, o, _ = bbox.size()
    N = b * o

    bbox_1 = bbox.float().view(-1, 4)
    x0, y0 = bbox_1[:, 0], bbox_1[:, 1]
    ww, hh = bbox_1[:, 2], bbox_1[:, 3]

    x0 = x0.contiguous().view(N, 1).expand(N, H)
    ww = ww.contiguous().view(N, 1).expand(N, H)
    y0 = y0.contiguous().view(N, 1).expand(N, W)
    hh = hh.contiguous().view(N, 1).expand(N, W)

    X = torch.linspace(0, 1, steps=W).view(1, W).expand(N, W).cuda(device=x.device)
    Y = torch.linspace(0, 1, steps=H).view(1, H).expand(N, H).cuda(device=x.device)

    X = (X - x0) / ww
    Y = (Y - y0) / hh

    X_out_mask = ((X < 0) + (X > 1)).view(N, 1, W).expand(N, H, W)
    Y_out_mask = ((Y < 0) + (Y > 1)).view(N, H, 1).expand(N, H, W)

    out_mask = 1 - (X_out_mask + Y_out_mask).float().clamp(max=1)
    return out_mask.view(b, o, H, W)

def batched_index_select(input, dim, index):
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def bbox_mask(x, bbox, H, W):
    b, o, _ = bbox.size()
    N = b * o

    bbox_1 = bbox.float().view(-1, 4)
    x0, y0 = bbox_1[:, 0], bbox_1[:, 1]
    ww, hh = bbox_1[:, 2], bbox_1[:, 3]

    x0 = x0.contiguous().view(N, 1).expand(N, H)
    ww = ww.contiguous().view(N, 1).expand(N, H)
    y0 = y0.contiguous().view(N, 1).expand(N, W)
    hh = hh.contiguous().view(N, 1).expand(N, W)

    X = torch.linspace(0, 1, steps=W).view(1, W).expand(N, W).cuda(device=x.device)
    Y = torch.linspace(0, 1, steps=H).view(1, H).expand(N, H).cuda(device=x.device)

    X = (X - x0) / ww
    Y = (Y - y0) / hh

    X_out_mask = ((X < 0) + (X > 1)).view(N, 1, W).expand(N, H, W)
    Y_out_mask = ((Y < 0) + (Y > 1)).view(N, H, 1).expand(N, H, W)

    out_mask = 1 - (X_out_mask + Y_out_mask).float().clamp(max=1)
    return out_mask.view(b, o, H, W)

class ISA(nn.Module):
    def __init__(self, num_classes, box_ks=3, in_ch=3, radius=1, norm=nn.BatchNorm2d):
        super(ISA, self).__init__()

        self.in_ch = in_ch
        padding = int(radius*(box_ks-1)/2)
        self.box_filter = nn.Conv2d(3, 3, kernel_size=box_ks, padding=padding, dilation=radius, bias=False, groups=3)
        self.box_filter.weight.data[...] = 1.0

        self.conv_a = nn.Sequential(nn.Conv2d(6, 32, kernel_size=1, bias=False),
                                    norm(32),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, kernel_size=1, bias=False),
                                    norm(32),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 3, kernel_size=1, bias=False))
        self.guided_map = nn.Sequential(
            nn.Conv2d(in_ch, 32, 1, bias=False),
            norm(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, bbox, bbox_mask_, b, o, seg, y_instance_mask):
        hh, ww = x.size(2), x.size(3)
        bbox = F.interpolate(bbox, size=(hh, ww), mode='bilinear')

        # x: [b, c, h, w], bbox: [b, o, h, w]
        bbox_ = []
        for i in range(o):
            bbox_in = bbox[:, i:i + 1]
            bbox_.append(self.mask_filter(x.detach(), bbox_in).clamp(0, 1))
        bbox = torch.cat(bbox_, 1)
        bbox = bbox * F.interpolate(bbox_mask_, size=(hh, ww), mode='nearest')
        instance_mask = (torch.sum(y_instance_mask.view(b, o, 1, 1) * bbox.detach(), 1, keepdim=True) > 0.2).float()
        seg = F.interpolate(seg, size=(hh, ww), mode='bilinear')
        panoptic_bbox = y_instance_mask.view(b, o, 1, 1) * bbox * instance_mask + seg * (1 - instance_mask)

        return bbox, panoptic_bbox

    def mask_filter(self, guide, mask):
        guide = self.guided_map(guide)
        mask_ = mask.expand(-1, 3, -1, -1)
        _, _, h_lrx, w_lrx = guide.size()

        N = self.box_filter(guide.data.new().resize_((1, 3, h_lrx, w_lrx)).fill_(1.0))

        ## mean_x
        mean_x = self.box_filter(guide)/(N + 1e-8)
        ## mean_y
        mean_y = self.box_filter(mask_)/(N + 1e-8)
        ## cov_xy
        cov_xy = self.box_filter(guide * mask_)/(N + 1e-8) - mean_x * mean_y
        ## var_x
        var_x  = self.box_filter(guide * guide)/(N + 1e-8) - mean_x * mean_x

        ## A
        A = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        ## b
        b = mean_y - A * mean_x

        mask_filtered = torch.mean(A * mask + b, dim=1, keepdim=True)

        return mask_filtered

'''
    Contextual aware
    https://arxiv.org/abs/2103.11897
'''
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def BoxRelationalEmbedding(f_g, dim_g=64, wave_len=1000, trignometric_embedding=True):
    """
    Given a tensor with bbox coordinates for detected objects on each batch image,
    this function computes a matrix for each image
    with entry (i,j) given by a vector representation of the
    displacement between the coordinates of bbox_i, and bbox_j
    input: np.array of shape=(batch_size, max_nr_bounding_boxes, 4)
    output: np.array of shape=(batch_size, max_nr_bounding_boxes, max_nr_bounding_boxes, 64)
    """
    # returns a relational embedding for each pair of bboxes, with dimension = dim_g
    # follow implementation of https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1014-L1055

    f_g = f_g.cuda()

    batch_size = f_g.size(0)

    x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=-1)

    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    # cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
    delta_x = cx - cx.view(batch_size, 1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))

    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)

    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).cuda()
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
    return(embedding)

def box_attention(query, key, value, box_relation_embds_matrix, mask=None, dropout=None):
    '''
    Compute 'Scaled Dot Product Attention as in paper Relation Networks for Object Detection'.
    Follow the implementation in https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1026-L1055
    '''
    if mask is not None:
        mask = mask.unsqueeze(1).expand(mask.size(0), query.size(1), mask.size(1))

    N = value.size()[:2]
    dim_k = key.size(-1)
    dim_g = box_relation_embds_matrix.size()[-1]

    # print(mask.shape)

    w_q = query
    w_k = key.transpose(-2, -1)
    w_v = value
    # print(w_q.shape)
    # print(w_k.shape)
    # attention weights
    scaled_dot = torch.matmul(w_q, w_k)
    # print(scaled_dot.shape)
    scaled_dot = scaled_dot / np.sqrt(dim_k)
    if mask is not None:
        scaled_dot = scaled_dot.masked_fill(mask == 0, -1e9)

    #w_g = box_relation_embds_matrix.view(N,N)
    w_g = box_relation_embds_matrix.squeeze(1)
    w_a = scaled_dot
    #w_a = scaled_dot.view(N,N)
    # print(w_g.shape)
    # print(w_a.shape)

    # multiplying log of geometric weights by feature weights
    w_mn = torch.log(torch.clamp(w_g, min=1e-6)) + w_a
    w_mn = torch.nn.Softmax(dim=-1)(w_mn)
    if dropout is not None:
        w_mn = dropout(w_mn)

    output = torch.matmul(w_mn, w_v)

    return output, w_mn

class BoxMultiHeadedAttention(nn.Module):
    '''
    Self-attention layer with relative position weights.
    Following the paper "Relation Networks for Object Detection" in https://arxiv.org/pdf/1711.11575.pdf
    '''

    def __init__(self, h, d_model, trignometric_embedding=True, legacy_extra_skip=False, dropout=0.1):
        "Take in model size and number of heads."
        super(BoxMultiHeadedAttention, self).__init__()

        assert d_model % h == 0
        self.trignometric_embedding = trignometric_embedding
        self.legacy_extra_skip = legacy_extra_skip

        # We assume d_v always equals d_k
        self.h = h
        self.d_k = d_model // h
        self.d_v = d_model // h
        if self.trignometric_embedding:
            self.dim_g = 64
        else:
            self.dim_g = 4
        geo_feature_dim = self.dim_g

        # matrices W_q, W_k, W_v, and one last projection layer
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.WGs = clones(nn.Linear(geo_feature_dim, 1, bias=True), h)
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm0 = nn.LayerNorm(d_model)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_query, input_key, input_value, input_box, mask=None):
        "Implements Figure 2 of Relation Network for Object Detection"
        # if mask is not None:
        # Same mask applied to all h heads.
        #    mask = mask.unsqueeze(1)

        d_k, d_v, n_head = self.d_k, self.d_v, self.h
        sz_b0, len_q, _ = input_query.size()
        sz_b, len_k, _ = input_key.size()
        sz_b, len_v, _ = input_value.size()

        nbatches = input_query.size(0)

        residual = input_query

        # tensor with entries R_mn given by a hardcoded embedding of the relative position between bbox_m and bbox_n
        relative_geometry_embeddings = BoxRelationalEmbedding(input_box, trignometric_embedding=self.trignometric_embedding)
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, self.dim_g)
        # print(flatten_relative_geometry_embeddings.shape)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k)
             for l, x in zip(self.linears, (input_query, input_key, input_value))]

        q = query.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = key.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = value.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        box_size_per_head.insert(1, 1)
        relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in self.WGs]
        relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
        relative_geometry_weights = F.relu(relative_geometry_weights)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.box_attn = box_attention(q, k, v, relative_geometry_weights, mask=mask,
                                         dropout=self.dropout)
        # print(x.shape)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)

        # An extra internal skip connection is added. This is only
        # kept here for compatibility with some legacy models. In
        # general, there is no advantage in using it, as there is
        # already an outer skip connection surrounding this layer.
        # if self.legacy_extra_skip:
        #    x = input_value + x
        # print(residual.shape)

        output = self.layer_norm0(x + residual)
        new_residual = output

        output = self.dropout(self.linears[-1](output))
        output = self.layer_norm(output + new_residual)

        return output
