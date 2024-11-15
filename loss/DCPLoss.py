import torch
import torch.nn as nn
from torch.nn import L1Loss
from torch.autograd import Variable


def DCLoss(img, patch_size, use_gpu=True):
    """
     unsupervised loss, calculating dark channel of image, the image shape is of N*C*W*H
    """
    maxpool = nn.MaxPool3d((3, patch_size, patch_size), stride=1, padding=(0, patch_size // 2, patch_size // 2))
    dc = maxpool(0 - img[:, None, :, :, :])

    target = Variable(torch.FloatTensor(dc.shape).zero_())
    if use_gpu:
        target = target.cuda()

    loss = L1Loss(size_average=True)(-dc, target)
    return loss
