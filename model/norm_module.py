import torch
import torch.nn as nn
import torch.nn.functional as F


# Adaptive instance normalization
# modified from https://github.com/NVlabs/MUNIT/blob/d79d62d99b588ae341f9826799980ae7298da553/networks.py#L453-L482
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, num_w=512, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

        # projection layer
        self.weight_proj = nn.Linear(num_w, num_features)
        self.bias_proj = nn.Linear(num_w, num_features)

    def forward(self, x, w):
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        weight, bias = self.weight_proj(w).contiguous().view(-1) + 1, self.bias_proj(w).contiguous().view(-1)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, weight, bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class SpatialAdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, num_w=512, eps=1e-5, momentum=0.1):
        super(SpatialAdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

        # projection layer
        self.weight_proj = nn.Linear(num_w, num_features)
        self.bias_proj = nn.Linear(num_w, num_features)

    def forward(self, x, w, bbox):
        b, c, h, w = x.size()
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        return x


class AdaptiveBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, num_w=512, eps=1e-5, momentum=0.1, affine=False, track_running_stats=True):
        super(AdaptiveBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        # projection layer
        self.weight_proj = nn.Linear(num_w, num_features)
        self.bias_proj = nn.Linear(num_w, num_features)

    def forward(self, x, w):
        self._check_input_dim(x)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        output = F.batch_norm(x, self.running_mean, self.running_var,
                              self.weight, self.bias,
                              self.training or not self.track_running_stats,
                              exponential_average_factor, self.eps)

        size = output.size()
        weight, bias = self.weight_proj(w) + 1, self.bias_proj(w)
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)

        return weight * output + bias

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class SpatialAdaptiveBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, num_w=512, eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=True):
        super(SpatialAdaptiveBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        # projection layer
        self.weight_proj = nn.Linear(num_w, num_features)
        self.bias_proj = nn.Linear(num_w, num_features)

    def forward(self, x, vector, bbox):
        """
        :param x: input feature map (b, c, h, w)
        :param vector: latent vector (b*o, dim_w)
        :param bbox: bbox map (b, o, h, w)
        :return:
        """
        self._check_input_dim(x)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        output = F.batch_norm(x, self.running_mean, self.running_var,
                              self.weight, self.bias,
                              self.training or not self.track_running_stats,
                              exponential_average_factor, self.eps)

        b, o, _, _ = bbox.size()
        _, _, h, w = x.size()
        bbox = F.interpolate(bbox, size=(h, w), mode='bilinear')
        # calculate weight and bias
        weight, bias = self.weight_proj(vector), self.bias_proj(vector)

        weight, bias = weight.view(b, o, -1), bias.view(b, o, -1)

        weight = torch.sum(bbox.unsqueeze(2) * weight.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=False) / \
                 (torch.sum(bbox.unsqueeze(2), dim=1, keepdim=False) + 1e-6) + 1
        bias = torch.sum(bbox.unsqueeze(2) * bias.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=False) / \
               (torch.sum(bbox.unsqueeze(2), dim=1, keepdim=False) + 1e-6)
        return weight * output + bias

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


from .sync_batchnorm import SynchronizedBatchNorm2d


class SpatialAdaptiveSynBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_w=512, batchnorm_func=SynchronizedBatchNorm2d, eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=True):
        super(SpatialAdaptiveSynBatchNorm2d, self).__init__()
        # projection layer
        self.num_features = num_features
        self.weight_proj = nn.utils.spectral_norm(nn.Linear(num_w, num_features))
        self.bias_proj = nn.utils.spectral_norm(nn.Linear(num_w, num_features))
        self.batch_norm2d = batchnorm_func(num_features, eps=eps, momentum=momentum,
                                           affine=affine)

    def forward(self, x, vector, bbox, vis=False):
        """
        :param x: input feature map (b, c, h, w)
        :param vector: latent vector (b*o, dim_w)
        :param bbox: bbox map (b, o, h, w)
        :return:
        """
        # self._check_input_dim(x)
        output = self.batch_norm2d(x)

        b, o, bh, bw = bbox.size()
        _, _, h, w = x.size()
        if bh != h or bw != w:
            bbox = F.interpolate(bbox, size=(h, w), mode='bilinear')
        # calculate weight and bias
        weight, bias = self.weight_proj(vector), self.bias_proj(vector)

        weight, bias = weight.view(b, o, -1), bias.view(b, o, -1)

        weight = torch.sum(bbox.unsqueeze(2) * weight.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=False) / \
                 (torch.sum(bbox.unsqueeze(2), dim=1, keepdim=False) + 1e-6) + 1
        bias = torch.sum(bbox.unsqueeze(2) * bias.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=False) / \
               (torch.sum(bbox.unsqueeze(2), dim=1, keepdim=False) + 1e-6)
        if vis:
            return weight * output + bias, weight, bias
        return weight * output + bias

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class SENorm2d(nn.Module):
    def __init__(self, num_features, num_w=512,
                 batchnorm_func=SynchronizedBatchNorm2d, eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=True):
        super(SENorm2d, self).__init__()
        # projection layer
        self.num_features = num_features
        self.weight_proj = conv2d(num_w, num_features, kernel_size=3, stride=1, pad=1)
        self.bias_proj = conv2d(num_w, num_features, kernel_size=3, stride=1, pad=1)
        self.batch_norm2d = batchnorm_func(num_features, eps=eps, momentum=momentum,
                                           affine=affine)

    def forward(self, x, vector, bbox, vis=False):
        """
        :param x: input feature map (b, c, h, w)
        :param vector: latent vector (b*o, dim_w)
        :param bbox: bbox map (b, o, h, w)
        :return:
        """
        # self._check_input_dim(x)
        output = self.batch_norm2d(x)

        b, o, bh, bw = bbox.size()
        _, _, h, w = x.size()
        if bh != h or bw != w:
            bbox = F.interpolate(bbox, size=(h, w), mode='bilinear')
        vector = F.interpolate(vector, size=(h, w), mode='bilinear', align_corners=True)
        # calculate weight and bias
        weight, bias = self.weight_proj(vector), self.bias_proj(vector)

        weight, bias = weight.view(b, o, -1, h, w), bias.view(b, o, -1, h, w)

        weight = torch.sum(bbox.unsqueeze(2) * weight, dim=1, keepdim=False) / \
                 (torch.sum(bbox.unsqueeze(2), dim=1, keepdim=False) + 1e-6) + 1
        bias = torch.sum(bbox.unsqueeze(2) * bias, dim=1, keepdim=False) / \
               (torch.sum(bbox.unsqueeze(2), dim=1, keepdim=False) + 1e-6)
        if vis:
            return weight * output + bias, weight, bias
        return weight * output + bias

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class SPADENorm2d(nn.Module):
    def __init__(self, num_features, num_w=512, batchnorm_func=SynchronizedBatchNorm2d, eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=True):
        super(SPADENorm2d, self).__init__()
        # projection layer
        self.num_features = num_features
        self.weight_proj = conv2d(num_w, num_features, kernel_size=1, stride=1, pad=0)
        self.bias_proj = conv2d(num_w, num_features, kernel_size=1, stride=1, pad=0)
        self.batch_norm2d = batchnorm_func(num_features, eps=eps, momentum=momentum,
                                           affine=affine)

    def forward(self, x, vector, bbox, vis=False):
        """
        :param x: input feature map (b, c, h, w)
        :param vector: latent vector (b*o, dim_w)
        :param bbox: bbox map (b, o, h, w)
        :return:
        """
        # self._check_input_dim(x)
        output = self.batch_norm2d(x)

        b, o, bh, bw = bbox.size()
        _, _, h, w = x.size()

        # vector = vector.view(b, o, -1)
        # bbox = torch.sum(bbox.unsqueeze(2) * vector.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=False) / \
        #        (torch.sum(bbox.unsqueeze(2), dim=1, keepdim=False) + 1e-6)
        if bh != h or bw != w:
            bbox = F.interpolate(bbox, size=(h, w), mode='bilinear')
        # calculate weight and bias
        weight, bias = self.weight_proj(bbox) + 1, self.bias_proj(bbox)

        if vis:
            return weight * output + bias, weight, bias
        return weight * output + bias

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class VNorm2d(nn.Module):
    def __init__(self, num_features, num_w=512, batchnorm_func=SynchronizedBatchNorm2d, eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=True):
        super(VNorm2d, self).__init__()
        # projection layer
        self.num_features = num_features
        self.weight_proj = conv2d(num_w, num_features, kernel_size=1, stride=1, pad=0)
        self.bias_proj = conv2d(num_w, num_features, kernel_size=1, stride=1, pad=0)
        self.batch_norm2d = batchnorm_func(num_features, eps=eps, momentum=momentum,
                                           affine=affine)

    def forward(self, x, vector, bbox, vis=False):
        """
        :param x: input feature map (b, c, h, w)
        :param vector: latent vector (b*o, dim_w)
        :param bbox: bbox map (b, o, h, w)
        :return:
        """
        # self._check_input_dim(x)
        output = self.batch_norm2d(x)
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

def conv2d(in_feat, out_feat, kernel_size=3, stride=1, pad=1, spectral_norm=True):
    conv = nn.Conv2d(in_feat, out_feat, kernel_size, stride, pad)
    if spectral_norm:
        return nn.utils.spectral_norm(conv, eps=1e-4)
    else:
        return conv
