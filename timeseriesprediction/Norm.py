import torch

class Norm:
    def __init__(self, data, norm_type=None):
        self.norm_type = norm_type
        if self.norm_type is not None:
            self.mean = torch.mean(data, dim=0)
            self.std = torch.std(data, dim=0)
            self.min_val = torch.min(data, dim=0).values
            self.max_val = torch.max(data, dim=0).values

    def normalize(self, x):
        normalized_x = x
        if self.norm_type == 'normal_dist':
            normalized_x = (x - self.mean) / self.std
        elif self.norm_type == 'min_max':
            normalized_x = (x - self.min_val) / (self.max_val - self.min_val)

        return normalized_x

    def denormalize(self, x):
        denormalized_x = x
        if self.norm_type == 'normal_dist':
            denormalized_x = (x * self.std) + self.mean
        elif self.norm_type == 'min_max':
            denormalized_x = x * (self.max_val - self.min_val) + self.min_val

        return denormalized_x