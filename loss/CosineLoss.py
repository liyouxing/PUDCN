import torch
import torch.nn as nn


class CosineDimLoss(nn.Module):
    """ Cosine Embedding Loss"""

    def __init__(self, dim=-1):
        super(CosineDimLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=dim)

    def forward(self, x, y):
        loss = 1 - torch.mean(self.cos(x, y))
        return loss


class CosineLoss(nn.Module):
    """ Cosine Embedding Loss"""

    def __init__(self, ):
        super(CosineLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        b = x.shape[0]
        x_flatten = x.view(b, -1)
        y_flatten = y.view(b, -1)
        loss = 1 - torch.mean(self.cos(x_flatten, y_flatten))
        return loss
