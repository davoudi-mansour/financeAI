import torch
import torch.nn as nn

class Norm:
    def __init__(self, data, norm_type=None):
        self.norm_type = norm_type
        if self.norm_type is not None:
            self.mean = torch.mean(data, dim=0)
            self.std = torch.std(data, dim=0)
            self.min_val = torch.min(data, dim=0).values
            self.max_val = torch.max(data, dim=0).values
            self.eps = 1e-10

    def normalize(self, x):
        normalized_x = x
        if self.norm_type == 'normal_dist':
            normalized_x = (x - self.mean) / self.std
        elif self.norm_type == 'min_max':
            normalized_x = (x - self.min_val) / (self.max_val - self.min_val + self.eps)

        return normalized_x

    def denormalize(self, x):
        denormalized_x = x
        if self.norm_type == 'normal_dist':
            denormalized_x = (x * self.std) + self.mean
        elif self.norm_type == 'min_max':
            denormalized_x = x * (self.max_val - self.min_val + self.eps) + self.min_val

        return denormalized_x


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x