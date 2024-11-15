import torch
import torch.nn as nn


class FrequencyLoss(nn.Module):
    """ MSE in frequency domain"""

    def __init__(self):
        super(FrequencyLoss, self).__init__()

    def forward(self, x, target):
        b, c, h, w = x.size()
        x = x.contiguous().view(-1, h, w)
        target = target.contiguous().view(-1, h, w)
        x_fft = torch.rfft(x, signal_ndim=2, normalized=False, onesided=True)
        target_fft = torch.rfft(target, signal_ndim=2, normalized=False, onesided=True)

        _, h, w, f = x_fft.size()
        x_fft = x_fft.view(b, c, h, w, f)
        target_fft = target_fft.view(b, c, h, w, f)
        diff = x_fft - target_fft
        return torch.mean(torch.sum(diff ** 2, (1, 2, 3, 4)))
