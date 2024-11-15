import torch
from skimage.metrics import structural_similarity


# -- Binary Error Map and Accuracy of Unwrapping -- #
def single_bem_ts3(pred, gt):
    """ singe bem, a feature map """
    pred = pred.detach()
    gt = gt.detach()

    min_value = 0
    bem = torch.zeros(pred.shape)
    bem[torch.abs(pred - gt) <= ((gt - min_value) * 0.05)] = 1.

    return bem  # (c, h, w)


def batch_bem_ts4(pred, gt):
    """ bem in a batch """
    pred = pred.detach()
    gt = gt.detach()

    min_value = 0  # min(gt)
    bem = torch.zeros(pred.shape)
    bem[torch.abs(pred - gt) <= ((gt - min_value) * 0.05)] = 1.

    return bem  # (b, c, h, w)


def single_au_ts3(pred, gt, bem_bool=False):
    """ average au in a batch ↑↑ """
    bem = single_bem_ts3(pred, gt)
    c, h, w = bem.shape

    au = torch.sum(bem) / (c * h * w)

    if bem_bool:
        return au, bem  # float, (c, h, w)
    else:
        return au


def batch_au_ts4(pred, gt, bem_bool=False):
    """ average au in a batch ↑↑ """
    bem = batch_bem_ts4(pred, gt)
    b, c, h, w = bem.shape

    au = torch.sum(bem) / (b * c * h * w)

    if bem_bool:
        return au, bem  # float, (b, c, h, w)
    else:
        return au


# -- Root Mean Square Error -- #
def single_rmse_ts3(pred, gt):
    """ single rmse ↓↓ """
    pred = pred.detach()
    gt = gt.detach()
    rmse = torch.sqrt(torch.mean((pred - gt) ** 2))

    return rmse


def batch_rmse_ts4(pred, gt):
    """ average rmse in a batch """
    pred = pred.detach()
    gt = gt.detach()
    b, _, _, _ = pred.shape

    rmse = single_rmse_ts3(pred[0], gt[0])
    for i in range(b - 1):
        rmse += single_rmse_ts3(pred[i], gt[i])

    return rmse / b


# -- peak-signal-noise-ratio -- #
def single_psnr_ts3(pred, gt, data_range=1.):
    """
    single psnr ↑↑

    Params
    ----------
    data_range : float
        data max range, e.g. if normalization, data_range = 1.,
        if image data, data_range = 255.,
        if PhUn data, data_range = the ceil of phase data range

    Returns
    ----------
    psnr : float
        Return psnr, range in [0, Inf)
    """
    pred = pred.detach()
    gt = gt.detach()
    mse = torch.mean((pred - gt) ** 2)
    signal_max = data_range

    psnr = 10 * torch.log10((signal_max ** 2) / mse)

    return psnr


def batch_psnr_ts4(pred, gt, data_range=1.):
    """ average psnr in a batch """
    pred = pred.detach()
    gt = gt.detach()
    b, _, _, _ = pred.shape

    psnr = single_psnr_ts3(pred[0], gt[0], data_range)
    for i in range(b - 1):
        psnr += single_psnr_ts3(pred[i], gt[i], data_range)

    return psnr / b


# -- correlation coefficient -- #
def cov_ts3(a, b):
    return torch.mean(torch.mul(
        (a - torch.mean(a)),
        (b - torch.mean(b))))


def single_cc_ts3(pred, gt):
    """
    single cc ↑↑, range in [0, 1],
    cc = cov(x, y) / sqrt(var(pred) * var(gt))
    """
    pred = pred.detach()
    gt = gt.detach()

    # cov(x, y)
    cov_pred_gt = cov_ts3(pred, gt)

    # var(x), or cov(x, x)
    var_pred = cov_ts3(pred, pred)
    var_gt = cov_ts3(gt, gt)

    cc = cov_pred_gt / torch.sqrt(var_pred * var_gt)

    return cc


def batch_cc_ts4(pred, gt):
    """ average cc in a batch """
    pred = pred.detach()
    gt = gt.detach()
    b, _, _, _ = pred.shape

    cc = single_cc_ts3(pred[0], gt[0])
    for i in range(b - 1):
        cc += single_cc_ts3(pred[i], gt[i])

    return cc / b


# --- Structural Similarity --- #
def single_ssim_ts3(pred, gt, data_range=255.0):
    """
    single ssim ↑↑

    Params
    ----------
    data_range : float
        If normalization, data range = 1.,
        If image data, data range = 255.
        If Ph data, data range = the ceil of Ph data range
    Returns
    ----------
    ssim : float
        Return ssim, range in [0, 1]
    """
    pred = pred.permute(1, 2, 0).data.cpu().numpy()
    gt = gt.permute(1, 2, 0).data.cpu().numpy()
    ssim = structural_similarity(pred, gt, data_range=data_range, channel_axis=-1)

    return ssim


def batch_ssim_ts4(pred, gt, data_range=255.):
    """ average ssim in a batch"""
    pred_list = torch.split(pred, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    pred_list_np = [pred_list[ind].permute(1, 2, 0).data.cpu().numpy()
                    for ind in range(len(pred_list))]
    gt_list_np = [gt_list[ind].permute(1, 2, 0).data.cpu().numpy()
                  for ind in range(len(pred_list))]
    ssim_list = [structural_similarity(pred_list_np[ind], gt_list_np[ind], data_range=data_range,
                                       channel_axis=-1) for ind in range(len(pred_list))]
    ssim = 1.0 * sum(ssim_list) / len(ssim_list)

    return ssim


def all_batch_avg_scores(score_list):
    """ average score of all batch """
    if len(score_list) == 0:
        return 0.
    else:
        avg_score = 1.0 * sum(score_list) / len(score_list)

        return avg_score


# --- valid valuate --- #
def validation(net, val_data_loader, device):
    rmse_list = []

    net.eval()

    for batch_id, val_data in enumerate(val_data_loader):
        with torch.no_grad():
            inputs, gt, _ = val_data
            inputs = inputs.to(device)
            gt = gt.to(device)
            pred, _, _ = net(inputs)

        # --- access --- #
        rmse_list.append(batch_rmse_ts4(pred, gt))

    avg_rmse = all_batch_avg_scores(rmse_list)
    return avg_rmse  # avr_ssim


if __name__ == "__main__":
    pass
