import os
import glob
import time
import torch
import re

from datetime import datetime
from guided_filter_pytorch.guided_filter import GuidedFilter


# --- last info check --- #
def find_last_point(save_dir):
    file_list = glob.glob(os.path.join(save_dir, '*params_*.tar'))  # *表示任意长
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*params_(.*).tar.*", file_)  # re正则表达,.*匹配任意长度字符,()表示提取,
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


# --- print and save net info --- #
def print_network(net, log_dir):
    num_params = 0
    # txt记录用于文本查看
    log_txt = open(log_dir + "/net_info.txt", "a+")

    for index, params in enumerate(net.parameters()):
        num_params += params.numel()
        log_txt.write("index: {:0>4}\tparams: {:0>7}\ttime: {}\n".format(
            index, params.numel(), datetime.now()))

    log_txt.write("total: {:0>8}\ttime:{}\n".format(num_params, datetime.now()))
    log_txt.close()

    # print(net)
    print('Total number of parameters: %d' % num_params)


def print_log(log_dir, epoch, num_epochs, epoch_time, val_rmse, loss, lr):
    print('Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Val_rmse:{4:.4f}, loss:{5:.4f}, lr:{6:.6f}'.format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch_time, epoch, num_epochs, val_rmse, loss, lr))

    # --- Write the training log --- #
    with open('./{}/training_log.txt'.format(log_dir), 'a') as f:
        print('Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Val_rmse:{4:.4f}, loss:{5:.4f}, lr:{6:.6f}'.format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch_time, epoch, num_epochs, val_rmse, loss, lr),
            file=f)


# --- read image --- #
def generate_new_seq(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    file_list = []
    for line in lines:
        file_list.append(line.strip())
    f.close()
    return file_list  # [1:10000]


# --- row and high gradient --- #
def gradient(x):
    gradient_h = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])  # 宽度相邻像素之间的梯度
    gradient_v = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])  # 高度相邻像素之间的梯度
    return gradient_h, gradient_v


# --- Normalization --- #

def tensor_normal(tensor):
    """
        normalization: (pixel - min) / (max - min)
    """
    max_v = torch.max(tensor)
    min_v = torch.min(tensor)
    value = (tensor - min_v) / (max_v - min_v)
    return value


def guided_filter_ts4(im, p, r=15, eps=0.1):
    """
        batch guided filtering
        params:
            im: guided image
            p: input image
    """
    gf = GuidedFilter(r, eps).to(im.device)

    return gf(im, p)


if __name__ == "__main__":
    pass
