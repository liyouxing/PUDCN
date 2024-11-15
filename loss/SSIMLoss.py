import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


# caculate gaussian vector
def gaussian(window_size, sigma):
    """
    :param
        window_size:guassian vector size
    """
    # 高斯权值分布,利用数组index减去index中值来体现
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                          for x in range(window_size)])
    return gauss / gauss.sum()


# use two gaussian vector make a gaussian kernel
def create_window(window_size, channel):
    """
    :param window_size:  guassian vector size
    :param channel: guassian channel
    :return: guassian kernel
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(dim=1)  # e.g.[1].unsqueeze(1)=[[1]]
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """
    :param window: gaussian kernel
    :param window_size: kernel size
    :param channel: image channel
    :param size_average:
    note:
    you can understand nn.Conv2d is 2d convolution layer,but F.conv3d is 2d convolution operation
    torch.nn.functional.conv2d(input,weight,bias=None,stride=1,padding=0,dilation=1,groups=1)
    input--(m,C,H,W)
    weight--(output_channel,in_channel,H,W),first is out_channel,second is in_channel,weight_in_channel should
    equal to image_inchannel/groups,and
    """
    # F.conv2d(input tensor,weight,bias,...),not i_channel,out_channel
    # group img channel then conv,then output weight_out_channel
    # mu1 mu2 is Gauss weighted average
    mu1 = F.conv2d(img1, window, padding=window_size // 2,
                   groups=channel)  # img1(m,3,64,64),window(3,1,11,11),mu1(m,3,h,w)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)  # .pow only in tensor,mu1^2
    mu2_sq = mu2.pow(2)  # mu2^2
    mu1_mu2 = mu1 * mu2  # mu1*mu2

    # sigma is variance
    # can use historgram statistic variance caculate a weight block pixel variance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # a gaussian window will make a ssim,how much window in a image will make how much block ssim
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)  # mean(dim)


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        window = create_window(self.window_size, channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


"""def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)
"""
