import os
import torch
import time
import torchvision.utils as utils

from torch.utils.data import DataLoader

from options.test_option import args
from options.data_option import args as dataArgs
from datasets.dRG_PhUn_dataset import MyDataset
from models.PUDCNv1 import PUDCN
from Utils.matUtils import save_ts3_to_mat
from Utils.accessUtils import single_rmse_ts3, single_au_ts3, single_ssim_ts3
from Utils.basicUtils import tensor_normal

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---settings --- #
pred_results_dir = args.results_dir
score_txt_dir = pred_results_dir

if not os.path.exists(score_txt_dir):
    os.makedirs(score_txt_dir)

assess_txt = open(score_txt_dir + args.results_dir.split('/')[-2] + ".txt", "a+")

test_txt_dir = args.test_txt_dir
test_input_txt = args.test_input_txt
test_gt_txt = args.test_gt_txt
test_bs = args.test_batch_size

test_txt_list = [test_input_txt, test_gt_txt]

# MyDataset
test_data = MyDataset(data_txt_dir=test_txt_dir, data_txt_list=test_txt_list)

# DataLoader
test_loader = DataLoader(dataset=test_data, batch_size=test_bs, num_workers=args.num_workers)

# 2.创建网络
net = PUDCN()
net = net.to(device)

# load pretrain model
checkpoint = torch.load(args.pretrain_model_dir)  # load .tar file
net.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})

# testing
n = 0

total_ssim = 0.0  # ↑↑
total_rmse = 0.0  # ↓↓
total_au = 0.0  # ↑↑
net.eval()
for batch_id, data in enumerate(test_loader):
    start_time = time.time()

    with torch.no_grad():
        inputs, gt, save_name = data
        inputs = inputs.to(device)
        gt = gt.to(device)

        uwp, _, _ = net(inputs)

        # inv of scaling
        uwp = uwp * dataArgs.uwp_ceil
        gt = gt * dataArgs.uwp_ceil

    for ind in range(test_bs):

        pred_name_split = save_name[ind].split('/')

        n += 1
        print(pred_name_split[-1], n)

        save_dir1 = pred_results_dir + '/uwp/'
        save_dir2 = pred_results_dir + '/bem/'

        if not os.path.exists(save_dir1):
            os.makedirs(save_dir1)
            os.makedirs(save_dir2)

        # performance metrics
        rmse = single_rmse_ts3(uwp[ind], gt[ind])
        ssim = single_ssim_ts3(uwp[ind], gt[ind], data_range=dataArgs.uwp_ceil)
        au, bem = single_au_ts3(uwp[ind], gt[ind], bem_bool=True)

        print("Ph:{}  rmse:{:.4f}  ssim:{:.4f}  au:{:.4f}".format(
            save_name[ind], rmse, ssim, au))
        assess_txt.write("Ph:{}  rmse:{:.4f}  ssim:{:.4f}  au:{:.4f}\n".format(
            save_name[ind], rmse, ssim, au))

        total_ssim += + ssim
        total_rmse += rmse
        total_au += au

        # saving results
        save_ts3_to_mat(uwp[ind], save_dir1, pred_name_split[-1], key='uwp')
        utils.save_image(tensor_normal(bem[ind]), save_dir2 + '{}'.format(
            pred_name_split[-1][:-4] + '.bmp'))

avg_rmse = total_rmse / n
avg_ssim = total_ssim / n
avg_au = total_au / n

print("Total info:\nMean rmse:{:.4f}  ssim:{:.4f}  au:{:.4f}".format(
    avg_rmse, avg_ssim, avg_au))
assess_txt.write("Total info:\nMean rmse:{:.4f}  ssim:{:.4f}  au:{:.4f}\n".format(
    avg_rmse, avg_ssim, avg_au))
assess_txt.close()
