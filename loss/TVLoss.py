import torch
import torch.nn as nn


class TVLoss_Charbonnier(nn.Module):
    """ Total Variation Loss, be used to denoise and smooth image"""

    def __init__(self):
        super(TVLoss_Charbonnier, self).__init__()

        self.e = 0.000001 ** 2

    def forward(self, x):
        h_tv = torch.abs((x[:, :, 1:, :] - x[:, :, :-1, :]))

        h_tv = torch.mean(torch.sqrt(h_tv ** 2 + self.e))

        w_tv = torch.abs((x[:, :, :, 1:] - x[:, :, :, :-1]))

        w_tv = torch.mean(torch.sqrt(w_tv ** 2 + self.e))

        return h_tv + w_tv
