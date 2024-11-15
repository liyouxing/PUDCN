import torch
import torch.fft as fft


def FRCLoss(img1, img2, device=None):
    """
        FRC(img1, img2)  ->  1-FRC(img1, img2)  -> return (1-FRC(img1, img2))**2
    :param
        img1: single channel image, dim of (b, 1, w, w)
        img2: single channel image, dim of (b, 1, w, w)
    :return:
        frc loss based on mse loss
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img1 = img1.squeeze(1)  # (b, w, w)
    img2 = img2.squeeze(1)

    bs, nx, ny = img1.shape
    rnyquist = nx // 2
    x = torch.cat((torch.arange(0, nx / 2), torch.arange(-nx / 2, 0))).to(device)
    y = x
    X, Y = torch.meshgrid(x, y)
    map = X ** 2 + Y ** 2  # 2D bin map
    index = torch.round(torch.sqrt(map.float()))
    r = torch.arange(0, rnyquist + 1).to(device)  # 0, 1, 2, ..., w/2
    F1 = fft.fft2(img1).permute(1, 2, 0)  # (b, w, w) -> (w, w, b)
    F2 = fft.fft2(img2).permute(1, 2, 0)

    C_r, C1, C2, C_i = [torch.empty(rnyquist + 1, bs).to(device) for i in range(4)]  # (w/2+1, b)

    for ii in r:  # cal the frc of each radius r in a batch samples
        auxF1 = F1[torch.where(index == ii)]  # (l_r, b)   e.g. (l_1, b) (l_2, b) ... (l_r, b)
        auxF2 = F2[torch.where(index == ii)]
        C_r[ii] = torch.sum(auxF1.real * auxF2.real + auxF1.imag * auxF2.imag, dim=0)
        C_i[ii] = torch.sum(auxF1.imag * auxF2.real - auxF1.real * auxF2.imag, dim=0)
        C1[ii] = torch.sum(auxF1.real ** 2 + auxF1.imag ** 2, dim=0)
        C2[ii] = torch.sum(auxF2.real ** 2 + auxF2.imag ** 2, dim=0)

    FRC = torch.sqrt(C_r ** 2 + C_i ** 2) / torch.sqrt(C1 * C2)  # (w / 2 +1, b)
    FRC[torch.isnan(FRC)] = 1.0  # if nan: -> 1.
    FRCm = 1 - FRC
    My_FRCloss = torch.mean((FRCm) ** 2)
    return My_FRCloss
