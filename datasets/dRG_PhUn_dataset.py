import random

import torch
import scipy.io as sio
from options.data_option import args
from torch.utils.data import Dataset


# MyDataset from Dataset
class MyDataset(Dataset):
    def __init__(self, data_txt_dir=None, data_txt_list=None, isTraining=False):
        super().__init__()

        self.in_list = None
        self.gt_list = None

        self.data_txt_dir = data_txt_dir
        self.data_txt_list = data_txt_list
        self.isTraining = isTraining

        self.read_txt_list(self.data_txt_dir, self.data_txt_list, self.isTraining)

    def read_txt_list(self, data_txt_dir=None, data_txt_list=None, isTraining=True):
        self.in_list = generate_new_seq(data_txt_dir + data_txt_list[0])
        self.gt_list = generate_new_seq(data_txt_dir + data_txt_list[1])

        if isTraining:
            # read txt content to list
            pass

    def __getitem__(self, index):
        in_put = load_mat_unit(self.data_txt_dir + self.in_list[index], keys='input')
        gt = load_mat_unit(self.data_txt_dir + self.gt_list[index], keys='gt')
        save_name = self.in_list[index]  # used to testing saved image name

        # --- Transform float numpy to tensor --- #
        in_ts = torch.tensor(in_put, dtype=torch.float32).unsqueeze(0)
        gt_ts = torch.tensor(gt, dtype=torch.float32).unsqueeze(0)

        # -------------- Normalization ---------------- #

        in_ts = (in_ts - args.wp_floor) / (args.wp_ceil - args.wp_floor)
        gt_ts = gt_ts / args.uwp_ceil

        if self.isTraining:
            if args.train_data_aug:
                in_ts, gt_ts = dRG_data_aug(in_ts, gt_ts)
            return in_ts, gt_ts
        else:
            return in_ts, gt_ts, save_name

    def __len__(self):
        return len(self.in_list)


# --- read image --- #
def generate_new_seq(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    file_list = []
    for line in lines:
        file_list.append(line.strip())
    f.close()
    return file_list  # [1:10000]


# --- load .mat file --- #
def load_mat_unit(mat_path, keys):
    """
        .mat要求：文件中保存的是单个结构体
        func: 直接将.mat中的单个结构体转换成numpy
    """
    return sio.loadmat(mat_path)[keys]  # -4表示去除.mat后缀名


# --- data augmentation --- #
def dRG_data_aug(inp, gt):
    """
    data augmentation
    """
    mode = random.randint(0, 5)

    if mode == 0:  # origin
        return inp, gt
    elif mode == 1:  # vertical flip
        return torch.flip(inp, dims=[-1]), torch.flip(gt, dims=[-1])
    elif mode == 2:  # horizontal flip
        return torch.flip(inp, dims=[-2]), torch.flip(gt, dims=[-2])
    elif mode == 3:  # rot 90
        return torch.rot90(inp, k=1, dims=[-2, -1]), torch.rot90(gt, k=1, dims=[-2, -1])
    elif mode == 4:  # rot 180
        return torch.rot90(inp, k=2, dims=[-2, -1]), torch.rot90(gt, k=2, dims=[-2, -1])
    else:  # rot 270
        return torch.rot90(inp, k=-1, dims=[-2, -1]), torch.rot90(gt, k=-1, dims=[-2, -1])