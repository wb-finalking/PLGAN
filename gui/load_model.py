import argparse
import json
import math
import os
from datetime import datetime
from collections import OrderedDict
import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from imageio import imwrite

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True)
parser.add_argument('--output_dir', default='outputs')
args = parser.parse_args()


def get_model():
    if not os.path.isdir(os.path.join('src', args.output_dir)):
        print('Output directory "%s" does not exist; creating it' % args.output_dir)
        os.makedirs(os.path.join('src', args.output_dir))

    # device = torch.device('cuda:0')

    # Load the model, with a bit of care in case there are no GPUs
    model = torch.jit.load(args.model_path)
    model.colors = torch.FloatTensor(np.load('./color.npy')).cuda()
    # model.eval()
    # model.to(device)

    return model


def json_to_img(scene_graph, model):
    output_dir = args.output_dir
    scene_graphs = json_to_scene_graph(scene_graph)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    # Run the model forward
    with open('./data/coco_vocab.json', 'r') as f:
        vocab = json.load(f)
    bbox_lt, attributes, label = encode_scene_graphs(scene_graphs, vocab)
    with torch.no_grad():
        z_obj = torch.from_numpy(truncted_random(num_o=8, thres=2.0)).float().cuda()
        z_im = torch.from_numpy(truncted_random(num_o=1, thres=2.0)).view(1, -1).float().cuda()
        # imgs, bbox_wh, panoptic_bbox = model(z_obj, bbox_lt=bbox_lt, attr=attributes, y=label.squeeze(dim=-1))
        imgs, bbox_wh, layout = model(z_obj, bbox_lt[:, :, :2], attributes, z_im, label.squeeze(dim=-1))
        # imgs_org, bmask_org = org_model(z_obj, torch.cat([bbox_lt, bbox_wh], 2), y=label.squeeze(dim=-1))
    imgs = imagenet_deprocess_batch(imgs)
    # imgs_org = imagenet_deprocess_batch(imgs_org)

    # Save the generated image
    for i in range(imgs.shape[0]):
        img_np = imgs[i].numpy().transpose(1, 2, 0).astype('uint8')
        return_img_path = os.path.join(output_dir, 'img{}.png'.format(current_time))
        img_path = os.path.join('src', output_dir, 'img{}.png'.format(current_time))
        imwrite(img_path, img_np)

        # img_np_org = imgs_org[i].numpy().transpose(1, 2, 0).astype('uint8')
        # return_img_path_org = os.path.join('images', output_dir, 'img_org{}.png'.format(current_time))
        # img_path_org = os.path.join('scripts', 'gui', return_img_path_org)
        # imwrite(img_path_org, img_np_org)

    # Save the generated layouts image
    for i in range(imgs.shape[0]):
        # b, h, w = bmask_org.size(0), bmask_org.size(2), bmask_org.size(3)
        # bmask_org = torch.cat([torch.zeros([b, 1, h, w], device=bmask_org.device).float(), bmask_org], 1)
        # bmask_org = torch.argmax(bmask_org, 1)
        # num_class = 9
        # one_hot = F.one_hot(bmask_org, num_class).permute(0, 3, 1, 2).float()
        # one_hot[:, 0] = 0
        # img_layout_np = one_hot_to_rgb(one_hot, model.colors[:num_class, :])[0].numpy().transpose(1, 2, 0).astype(
        #     'uint8')
        # return_img_layout_path = os.path.join('images', output_dir, 'img_layout{}.png'.format(current_time))
        # img_layout_path = os.path.join('scripts', 'gui', return_img_layout_path)
        # # vis.add_boxes_to_layout(img_layout_np, scene_graphs[i]['objects'], boxes_pred, img_layout_path,
        # #                         colors=obj_colors)
        # imwrite(img_layout_path, img_layout_np)

        b, h, w = layout.size(0), layout.size(2), layout.size(3)
        layout = torch.cat([torch.zeros([b, 1, h, w], device=layout.device).float(), layout], 1)
        layout = torch.argmax(layout, 1)
        num_class = 9
        one_hot = F.one_hot(layout, num_class).permute(0, 3, 1, 2).float()
        one_hot[:, 0] = 0
        layout = one_hot_to_rgb(one_hot, model.colors[:num_class, :])[0].numpy().transpose(1, 2, 0).astype('uint8')
        return_layout_path = os.path.join(output_dir, 'layouts{}.png'.format(current_time))
        layout_path = os.path.join('src', output_dir, 'layouts{}.png'.format(current_time))
        imwrite(layout_path, layout)

    return return_img_path, return_layout_path


def truncted_random(num_o=8, thres=1.0, bs=1):
    z = np.ones((1, num_o, 128)) * 100
    for i in range(num_o):
        for j in range(128):
            while z[0, i, j] > thres or z[0, i, j] < - thres:
                z[0, i, j] = np.random.normal()
    z = np.tile(z, (bs, 1, 1))
    return z

def one_hot_to_rgb(one_hot, colors):
    one_hot_3d = torch.einsum('abcd,be->aecd', (one_hot.cpu(), colors.cpu()))
    one_hot_3d *= (255.0 / one_hot_3d.max())
    return one_hot_3d

def encode_scene_graphs(scene_graphs, vocab, num_objs=8):
    """
    Encode one or more scene graphs using this model's vocabulary. Inputs to
    this method are scene graphs represented as dictionaries like the following:
    {
      "objects": ["cat", "dog", "sky"],
      "relationships": [
        [0, "next to", 1],
        [0, "beneath", 2],
        [2, "above", 1],
      ]
    }
    This scene graph has three relationshps: cat next to dog, cat beneath sky,
    and sky above dog.
    Inputs:
    - scene_graphs: A dictionary giving a single scene graph, or a list of
      dictionaries giving a sequence of scene graphs.
    Returns a tuple of LongTensors (objs, triples, obj_to_img) that have the
    same semantics as self.forward. The returned LongTensors will be on the
    same device as the model parameters.
    """
    if isinstance(scene_graphs, dict):
        # We just got a single scene graph, so promote it to a list
        scene_graphs = [scene_graphs]

    objs, triples, obj_to_img = [], [], []
    all_attributes = []
    all_features = []
    obj_offset = 0
    bbox_lt = []
    bbox_center = []
    print(scene_graphs)
    for i, sg in enumerate(scene_graphs):
        attributes = torch.zeros([num_objs, 25 + 10], dtype=torch.float).cuda()
        # Insert dummy __image__ object and __in_image__ relationships
        boxes = sg['boxes']
        image_idx = len(sg['objects'])

        for bbox in boxes:
            sx0, sy0, sx1, sy1 = bbox
            bbox_lt.append([sx0, sy0])
            bbox_center.append([(sx0+sx1)/2, (sy0+sy1)/2])
        for obj in sg['objects']:
            obj_idx = vocab['object_name_to_idx'][obj]
            objs.append(obj_idx)
            obj_to_img.append(i)
        for i, size_attr in enumerate(sg['attributes']['size']):
            attributes[i, size_attr] = 1
        for i, location_attr in enumerate(sg['attributes']['location']):
            attributes[i, location_attr + 10] = 1

        for _ in range(len(boxes), num_objs):
            bbox_lt.append([-0.6, -0.6])
            bbox_center.append([-0.6, -0.6])
            objs.append(0)
        all_attributes.append(attributes.unsqueeze(0))

    objs = torch.tensor(objs, dtype=torch.int64).unsqueeze(0).cuda()
    attributes = torch.cat(all_attributes, 0)
    bbox_lt = torch.FloatTensor(bbox_lt).unsqueeze(0).cuda()
    bbox_center = torch.FloatTensor(bbox_center).unsqueeze(0).cuda()

    return bbox_lt, attributes, objs

def json_to_scene_graph(json_text):
    scene = json.loads(json_text)
    if len(scene) == 0:
        return []
    image_id = scene['image_id']
    scene = scene['objects']
    objects = [i['text'] for i in scene]
    relationships = []
    size = []
    location = []
    features = []
    bbox_lt = []
    attr = []
    boxes = []
    for i in range(0, len(objects)):
        obj_s = scene[i]
        # Check for inside / surrounding

        sx0 = obj_s['left']
        sy0 = obj_s['top']
        sx1 = obj_s['width'] + sx0
        sy1 = obj_s['height'] + sy0
        bbox_lt.append([sx0, sy0])
        boxes.append([sx0, sy0, sx1, sy1])

        margin = (obj_s['size'] + 1) / 10 / 2
        mean_x_s = 0.5 * (sx0 + sx1)
        mean_y_s = 0.5 * (sy0 + sy1)

        sx0 = max(0, mean_x_s - margin)
        sx1 = min(1, mean_x_s + margin)
        sy0 = max(0, mean_y_s - margin)
        sy1 = min(1, mean_y_s + margin)

        size.append(obj_s['size'])
        location.append(obj_s['location'])

        features.append(obj_s['feature'])
        if i == len(objects) - 1:
            continue

        obj_o = scene[i + 1]
        ox0 = obj_o['left']
        oy0 = obj_o['top']
        ox1 = obj_o['width'] + ox0
        oy1 = obj_o['height'] + oy0


        mean_x_o = 0.5 * (ox0 + ox1)
        mean_y_o = 0.5 * (oy0 + oy1)
        d_x = mean_x_s - mean_x_o
        d_y = mean_y_s - mean_y_o
        theta = math.atan2(d_y, d_x)

        margin = (obj_o['size'] + 1) / 10 / 2
        ox0 = max(0, mean_x_o - margin)
        ox1 = min(1, mean_x_o + margin)
        oy0 = max(0, mean_y_o - margin)
        oy1 = min(1, mean_y_o + margin)

        if sx0 < ox0 and sx1 > ox1 and sy0 < oy0 and sy1 > oy1:
            p = 'surrounding'
        elif sx0 > ox0 and sx1 < ox1 and sy0 > oy0 and sy1 < oy1:
            p = 'inside'
        elif theta >= 3 * math.pi / 4 or theta <= -3 * math.pi / 4:
            p = 'left of'
        elif -3 * math.pi / 4 <= theta < -math.pi / 4:
            p = 'above'
        elif -math.pi / 4 <= theta < math.pi / 4:
            p = 'right of'
        elif math.pi / 4 <= theta < 3 * math.pi / 4:
            p = 'below'
        relationships.append([i, p, i + 1])

    return [{'objects': objects, 'relationships': relationships, 'attributes': {'size': size, 'location': location},
             'features': features, 'image_id': image_id, 'boxes': boxes}]

def rescale(x):
    lo, hi = x.min(), x.max()
    return x.sub(lo).div(hi - lo)

def imagenet_deprocess(rescale_image=True, MEAN=[0.5, 0.5, 0.5], STD=[0.5, 0.5, 0.5]):
    INV_MEAN = [-m for m in MEAN]
    INV_STD = [1.0 / s for s in STD]
    transforms = [
        T.Normalize(mean=[0, 0, 0], std=INV_STD),
        T.Normalize(mean=INV_MEAN, std=[1.0, 1.0, 1.0]),
    ]
    if rescale_image:
        transforms.append(rescale)
    return T.Compose(transforms)

def imagenet_deprocess_batch(imgs, rescale=True):
    """
    Input:
    - imgs: FloatTensor of shape (N, C, H, W) giving preprocessed images
    Output:
    - imgs_de: ByteTensor of shape (N, C, H, W) giving deprocessed images
      in the range [0, 255]
    """
    if isinstance(imgs, torch.autograd.Variable):
        imgs = imgs.data
    imgs = imgs.cpu().clone()
    deprocess_fn = imagenet_deprocess(rescale_image=rescale)
    imgs_de = []
    for i in range(imgs.size(0)):
        img_de = deprocess_fn(imgs[i])[None]
        img_de = img_de.mul(255).clamp(0, 255)
        imgs_de.append(img_de)
    imgs_de = torch.cat(imgs_de, dim=0)
    return imgs_de
