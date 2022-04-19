#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from skimage.transform import resize as imresize

import numpy as np
import h5py
import PIL


class VgSceneGraphDataset(Dataset):
    def __init__(self, vocab, h5_path, image_dir, image_size=(256, 256),
                 normalize_images=True, max_objects=10, max_samples=None,
                 include_relationships=True, use_orphaned_objects=True,
                 left_right_flip=False, dump_bbox_dict=None):
        super(VgSceneGraphDataset, self).__init__()

        self.image_dir = image_dir
        self.image_size = image_size
        self.vocab = vocab
        self.num_objects = len(vocab['object_idx_to_name'])
        self.use_orphaned_objects = use_orphaned_objects
        self.max_objects = max_objects
        self.max_samples = max_samples
        self.left_right_flip = left_right_flip
        self.include_relationships = include_relationships
        self.size_attribute_len = 10
        self.location_attribute_len = 25
        self.dump_bbox_dict = dump_bbox_dict
        if dump_bbox_dict is not None:
            self.use_pred_bbox = True
        else:
            self.use_pred_bbox = False

        transform = [Resize(image_size), T.ToTensor()]
        if normalize_images:
            transform.append(imagenet_preprocess())
        self.transform = T.Compose(transform)

        self.data = {}
        with h5py.File(h5_path, 'r') as f:
            for k, v in f.items():
                if k == 'image_paths':
                    self.image_paths = list(v)
                else:
                    self.data[k] = torch.IntTensor(np.asarray(v))

    def __len__(self):
        num = self.data['object_names'].size(0)
        if self.max_samples is not None:
            return min(self.max_samples, num)
        if self.left_right_flip:
            return num * 2
        return num

    def __getitem__(self, index):
        """
        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
        - triples: LongTensor of shape (T, 3) where triples[t] = [i, p, j]
          means that (objs[i], p, objs[j]) is a triple.
        """
        flip = False
        if index >= self.data['object_names'].size(0):
            index = index - self.data['object_names'].size(0)
            flip = True

        img_path = os.path.join(self.image_dir, self.image_paths[index])
        filename = self.image_paths[index]
        with open(img_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                if flip:
                    image = PIL.ImageOps.mirror(image)
                WW, HH = image.size
                image = self.transform(image.convert('RGB'))

        H, W = self.image_size

        # Figure out which objects appear in relationships and which don't
        obj_idxs_with_rels = set()
        obj_idxs_without_rels = set(range(self.data['objects_per_image'][index].item()))
        for r_idx in range(self.data['relationships_per_image'][index]):
            s = self.data['relationship_subjects'][index, r_idx].item()
            o = self.data['relationship_objects'][index, r_idx].item()
            obj_idxs_with_rels.add(s)
            obj_idxs_with_rels.add(o)
            obj_idxs_without_rels.discard(s)
            obj_idxs_without_rels.discard(o)

        obj_idxs = list(obj_idxs_with_rels)
        obj_idxs_without_rels = list(obj_idxs_without_rels)
        if len(obj_idxs) > self.max_objects - 1:
            obj_idxs = random.sample(obj_idxs, self.max_objects)
        if len(obj_idxs) < self.max_objects - 1 and self.use_orphaned_objects:
            num_to_add = self.max_objects - 1 - len(obj_idxs)
            num_to_add = min(num_to_add, len(obj_idxs_without_rels))
            obj_idxs += random.sample(obj_idxs_without_rels, num_to_add)
        O = len(obj_idxs)

        objs = torch.LongTensor(self.max_objects+1).fill_(-1)

        boxes = torch.FloatTensor([[0, 0, 1, 1]]).repeat(self.max_objects+1, 1)
        obj_idx_mapping = {}
        size_attribute = []
        location_attribute = []
        l_root = self.location_attribute_len ** (.5)
        for i, obj_idx in enumerate(obj_idxs):
            objs[i] = self.data['object_names'][index, obj_idx].item()
            x, y, w, h = self.data['object_boxes'][index, obj_idx].tolist()
            # OW, OH = self.data['image_wh'][index].tolist()
            # x0 = float(x) / float(OW)
            # y0 = float(y) / float(OH)
            # x1 = float(w) / float(OW)
            # y1 = float(h) / float(OH)
            x0 = float(x) / WW
            y0 = float(y) / HH
            x1 = float(w) / WW
            y1 = float(h) / HH
            if flip:
                x0 = 1 - (x0 + x1)
            boxes[i] = torch.FloatTensor([x0, y0, x1, y1])
            obj_idx_mapping[obj_idx] = i

            # size_index = round((self.size_attribute_len - 1) * (w * h) / (float(OW) * float(OH)))
            size_index = round((self.size_attribute_len - 1) * (w * h) / (WW * HH))
            local_size_attr = np.zeros([self.size_attribute_len], dtype=np.float32)
            if size_index >= self.size_attribute_len:
                size_index = self.size_attribute_len - 1
                print('==>', x, y, w, h, WW, HH)
            local_size_attr[size_index] = 1.0
            size_attribute.append(local_size_attr)

            mean_x = x0 + 0.5 * x1
            mean_y = y0 + 0.5 * y1
            location_index = round(mean_x * (l_root - 1)) + l_root * round(mean_y * (l_root - 1))
            # print('==>', mean_x, mean_y, l_root, location_index)
            local_location_attr = np.zeros([self.location_attribute_len], dtype=np.float32)
            if location_index >= self.location_attribute_len:
                location_index = self.location_attribute_len - 1
            local_location_attr[int(location_index)] = 1.0
            location_attribute.append(local_location_attr)

        # The last object will be the special __image__ object
        # objs[O - 1] = self.vocab['object_name_to_idx']['__image__']

        for i in range(O, self.max_objects+1):
            # objs.append(self.vocab['object_name_to_idx']['__image__'])
            # boxes.append(np.array([-0.6, -0.6, 0.5, 0.5]))
            objs[i] = self.vocab['object_name_to_idx']['__image__']
            boxes[i] = torch.FloatTensor([-0.6, -0.6, 0.5, 0.5])

            size_attribute.append(np.zeros([self.size_attribute_len], dtype=np.float32))
            location_attribute.append(np.zeros([self.location_attribute_len], dtype=np.float32))

        # triples = []
        # for r_idx in range(self.data['relationships_per_image'][index].item()):
        #     if not self.include_relationships:
        #         break
        #     s = self.data['relationship_subjects'][index, r_idx].item()
        #     p = self.data['relationship_predicates'][index, r_idx].item()
        #     o = self.data['relationship_objects'][index, r_idx].item()
        #     s = obj_idx_mapping.get(s, None)
        #     o = obj_idx_mapping.get(o, None)
        #     if s is not None and o is not None:
        #         triples.append([s, p, o])
        #
        # # Add dummy __in_image__ relationships for all objects
        # in_image = self.vocab['pred_name_to_idx']['__in_image__']
        # for i in range(O - 1):
        #     triples.append([i, in_image, O - 1])
        #
        # triples = torch.LongTensor(triples)
        if self.use_pred_bbox:
            key_ = os.path.splitext(os.path.basename(filename))[0]
            boxes = np.array(self.dump_bbox_dict[key_], dtype=np.float)
            boxes = boxes[0]

        size_attribute = np.concatenate([item[None, :] for item in size_attribute], axis=0)
        location_attribute = np.concatenate([item[None, :] for item in location_attribute], axis=0)
        attributes = np.concatenate([size_attribute, location_attribute], axis=1)

        return image, objs, boxes, filename, attributes


class Resize(object):
    def __init__(self, size, interp=PIL.Image.BILINEAR):
        if isinstance(size, tuple):
              H, W = size
              self.size = (W, H)
        else:
            self.size = (size, size)
        self.interp = interp

    def __call__(self, img):
        return img.resize(self.size, self.interp)


IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]


def imagenet_preprocess():
    return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def vg_collate_fn(batch):
    """
    Collate function to be used when wrapping a VgSceneGraphDataset in a
    DataLoader. Returns a tuple of the following:
    
    - imgs: FloatTensor of shape (N, C, H, W)
    - objs: LongTensor of shape (O,) giving categories for all objects
    - boxes: FloatTensor of shape (O, 4) giving boxes for all objects
    - triples: FloatTensor of shape (T, 3) giving all triples, where
    triples[t] = [i, p, j] means that [objs[i], p, objs[j]] is a triple
    - obj_to_img: LongTensor of shape (O,) mapping objects to images;
    obj_to_img[i] = n means that objs[i] belongs to imgs[n]
    - triple_to_img: LongTensor of shape (T,) mapping triples to images;
    triple_to_img[t] = n means that triples[t] belongs to imgs[n].
    """
    # batch is a list, and each element is (image, objs, boxes, triples)
    all_imgs, all_objs, all_boxes, all_triples = [], [], [], []
    all_obj_to_img, all_triple_to_img = [], []
    obj_offset = 0
    for i, (img, objs, boxes, triples) in enumerate(batch):
        all_imgs.append(img[None])
        O, T = objs.size(0), triples.size(0)
        all_objs.append(objs)
        all_boxes.append(boxes)
        triples = triples.clone()
        triples[:, 0] += obj_offset
        triples[:, 2] += obj_offset
        all_triples.append(triples)

        all_obj_to_img.append(torch.LongTensor(O).fill_(i))
        all_triple_to_img.append(torch.LongTensor(T).fill_(i))
        obj_offset += O

    all_imgs = torch.cat(all_imgs)
    all_objs = torch.cat(all_objs)
    all_boxes = torch.cat(all_boxes)
    all_triples = torch.cat(all_triples)
    all_obj_to_img = torch.cat(all_obj_to_img)
    all_triple_to_img = torch.cat(all_triple_to_img)

    out = (all_imgs, all_objs, all_boxes, all_triples,
           all_obj_to_img, all_triple_to_img)
    return out


def vg_uncollate_fn(batch):
    """
    Inverse operation to the above.
    """
    imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
    out = []
    obj_offset = 0
    for i in range(imgs.size(0)):
        cur_img = imgs[i]
        o_idxs = (obj_to_img == i).nonzero().view(-1)
        t_idxs = (triple_to_img == i).nonzero().view(-1)
        cur_objs = objs[o_idxs]
        cur_boxes = boxes[o_idxs]
        cur_triples = triples[t_idxs].clone()
        cur_triples[:, 0] -= obj_offset
        cur_triples[:, 2] -= obj_offset
        obj_offset += cur_objs.size(0)
        out.append((cur_img, cur_objs, cur_boxes, cur_triples))
    return out
