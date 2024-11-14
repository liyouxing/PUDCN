import time
import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from models.PUDCNv1 import PUDCN
from options.train_option import args
from datasets.dRG_PhUn_dataset import MyDataset
from Utils.basicUtils import print_network, find_last_point, print_log
from Utils.initUtils import init_weights
from Utils.accessUtils import validation, batch_rmse_ts4, all_batch_avg_scores
from loss.CharbonnierLoss import CharbonnierLoss, EdgeLoss

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
plt.switch_backend('agg')

# ######################## params settings ########################
# gpu device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hy-params
train_batch_size = args.train_batch_size
val_batch_size = args.val_batch_size
learning_rate = args.lr
num_epochs = args.epoch

print('--- Hyper-parameters for training ---')
print('learning_rate: {}\ntrain_batch_size: {}\nval_batch_size: {}\n'.format(
    learning_rate, train_batch_size, val_batch_size))

# data dir
train_txt_dir = args.train_txt_dir
val_txt_dir = args.val_txt_dir

train_input_txt = args.train_input_txt
train_gt_txt = args.train_gt_txt
train_txt_list = [train_input_txt,
                  train_gt_txt]

val_input_txt = args.val_input_txt
val_gt_txt = args.val_gt_txt
val_txt_list = [val_input_txt,
                val_gt_txt]

# training log
train_save_dir = args.train_save_dir
now_time = datetime.now()  # return time e.g. yyyy-mm-dd xx:xx:xx.xxxxx
time_str = datetime.strftime(now_time, "%m-%d")  # data time format %m-%d

log_dir = os.path.join(train_save_dir, time_str)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir=log_dir)  # tensorboardX writer
# ######################## params settings ########################


# ######################## data loading #########################
# MyDateset
train_data = MyDataset(data_txt_dir=train_txt_dir, data_txt_list=train_txt_list,
                       isTraining=args.is_training)
valid_data = MyDataset(data_txt_dir=val_txt_dir, data_txt_list=val_txt_list)

# dataloader
train_data_loader = DataLoader(dataset=train_data, batch_size=train_batch_size,
                               shuffle=True, num_workers=args.num_workers)
valid_data_loader = DataLoader(dataset=valid_data, batch_size=val_batch_size, shuffle=True,
                               num_workers=args.num_workers // 2 if args.num_workers > 1 else 1)
# ######################## data loading #########################


# ####################### create net ############################
# define net
net = PUDCN()

# parallel
net = nn.DataParallel(net)
net = net.to(device)

# init net
init_weights(net)

# log net
print_network(net, log_dir)  # print and save in log_dir
# ####################### create net ############################


# ############### loss and optimizer #####################
# define loss
char_loss = CharbonnierLoss().to(device)
edge_loss = EdgeLoss().to(device)
# build optimizer
# default: params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.eta_min)

# ############### loss and optimizer #####################


# #################### training iteration ########################
# load the latest log
initial_epoch = find_last_point(save_dir=log_dir)
if initial_epoch > 0:
    print('resuming by loading training epoch %d' % initial_epoch)
    checkpoint = torch.load(os.path.join(
        log_dir, "net_params_%d.tar" % initial_epoch))
    initial_epoch = checkpoint['epoch'] + 1
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    try:
        net.module.load_state_dict({k.replace(
            'module.', ''): v for k, v in checkpoint['model_state_dict'].items()})
    except FileNotFoundError:
        print("FileNotFoundError")
    print('continue training ... start in %d epoch' % (initial_epoch + 1))
    print('lr == %f' % scheduler.get_last_lr()[0])

# net training
for epoch in range(initial_epoch, num_epochs):
    ssim_list = []
    rmse_list = []
    start_time = time.time()

    net.train()
    for batch_id, train_data in enumerate(train_data_loader):

        inputs, gt = train_data
        inputs = inputs.to(device)
        gt = gt.to(device)

        # Zero the params grad
        optimizer.zero_grad()

        # Forward + Backward + Optimize
        uwp, _, _ = net(inputs)

        # cal loss
        loss = char_loss(uwp, gt) + edge_loss(uwp, gt)

        loss.backward()
        optimizer.step()  # update weight

        if not args.speed_training:
            bs_rmse = batch_rmse_ts4(uwp, gt)
            rmse_list.append(bs_rmse)

    # update lr
    scheduler.step(epoch)

    # end training
    one_epoch_time = time.time() - start_time

    # start epoch log
    train_epoch_rmse = all_batch_avg_scores(rmse_list)

    # validation
    net.eval()
    val_rmse = 0.
    if (epoch % args.save_epoch_interval == 0) & (epoch >= num_epochs * args.start_save_percent):
        val_rmse = validation(net, valid_data_loader, device)

    # log in txt
    print_log(log_dir, epoch + 1, num_epochs, one_epoch_time, val_rmse,
              loss, scheduler.get_last_lr()[0])

    # record tensorboard info
    writer.add_scalars('Train_Loss_group', {'train_loss': loss}, epoch)
    writer.add_scalars('learning_rate', {'lr': scheduler.get_last_lr()[0]}, epoch)
    writer.add_scalars('rmse', {'train_rmse': train_epoch_rmse, 'val_rmse': val_rmse}, epoch)

    if (epoch % args.save_epoch_interval == 0) & (epoch >= num_epochs * args.start_save_percent):
        # save net and optim state
        save_path = os.path.join(log_dir, "net_params_" + str(epoch + 1) + ".tar")
        torch.save({'epoch': epoch,
                    'model_state_dict': net.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()}, save_path)
# #################### training iteration ########################

print("Finished Training")
