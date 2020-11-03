import torch
import torch.nn as nn



class rotate90(nn.Module):
    def __init__(self, degree, dims=None):
        super(rotate90, self).__init__()
        self.degree = degree if degree >= 0 else 360 + degree
        self.k = self.degree // 90
        if dims:
            self.dims = dims
        else:
            self.dims = [2, 3]
    
    def forward(self, image, dims=None):
        if dims is None:
            return torch.rot90(image, self.k, self.dims)
        return torch.rot90(image, self.k, dims)


