import numpy as np
import math
import torch
import torch.nn.functional as F
from torch.fft import fft, fftfreq, ifft, fftshift


def rotate(tensor, degrees, use_GPU=False):
    """
        rotate batch tensor
    params:
        tensor: (b, c, h, w)
        degrees:
    """
    b = tensor.shape[0]
    angle = degrees / 180 * math.pi  # degree to rad

    transform_matrix = torch.tensor([
        [math.cos(angle), math.sin(-angle), 0],
        [math.sin(angle), math.cos(angle), 0]])

    if use_GPU:
        transform_matrix = transform_matrix.cuda()

    transform_matrix = transform_matrix.unsqueeze(0).repeat(b, 1, 1)

    grid = F.affine_grid(transform_matrix, tensor.shape, align_corners=False)
    rotation = F.grid_sample(tensor, grid, align_corners=False)

    return rotation


def radon(tensor, angle_step=180, angle_range=(0, 180)):
    """
        radon in tensor, tensor must be a square
    params:
        tensor: (b, c, w, w)
    return:
        sinogram, (b, c, w, angle_step)
    """
    b, c, h, w = tensor.shape
    res = torch.zeros((b, c, angle_step, w))
    range_low, range_high = angle_range

    for step in range(angle_step):
        rotation = rotate(tensor, -((step * (range_high - range_low) / angle_step) + range_low))
        res[:, :, step, :] = torch.sum(rotation, dim=2)

    return torch.transpose(res, dim0=-1, dim1=-2)


# --- fourier filters by batch tensor --- #
def sinogram_to_square(sinogram):  # (b, c, w, angles)
    _, _, w, _ = sinogram.shape
    diagonal = torch.ceil(torch.sqrt(torch.tensor(2)) * w).type(torch.IntTensor)
    pad = diagonal - w
    old_center = w // 2
    new_center = diagonal // 2
    pad_before = new_center - old_center
    pad_width = (0, 0, pad_before, pad - pad_before)
    return F.pad(sinogram, pad_width, mode='constant', value=0)


def get_fourier_filter(size, filter_name="hamming", use_GPU=False):
    """
        param:
            size: 1D filter length
            filter_name: fourier filter name
        return:
            filter, shape of (1, 1, size, 1)
    """
    n = torch.cat([torch.arange(start=1, end=size / 2 + 1, step=2, dtype=torch.int),
                   torch.arange(start=size / 2 - 1, end=0, step=-2, dtype=torch.int)])
    f = torch.zeros(size)

    if use_GPU:
        f = f.cuda()

    f[0] = 0.25
    f[1::2] = -1 / (math.pi * n) ** 2

    # Computing the ramp filter from the fourier transform of its
    # frequency domain representation lessens artifacts and removes a
    # small bias as explained in [1], Chap 3. Equation 61
    fourier_filter = 2 * torch.real(fft(f))  # ramp filter
    if filter_name == "ramp":
        pass

    elif filter_name == "shepp-logan":
        # Start from first element to avoid divide by zero
        omega = math.pi * fftfreq(size)[1:]
        if use_GPU:
            fourier_filter[1:] *= torch.sin(omega).cuda() / omega
        else:
            fourier_filter[1:] *= torch.sin(omega) / omega

    elif filter_name == "cosine":
        if use_GPU:
            freq = torch.linspace(0, math.pi, size + 1)[:-1].cuda()
        else:
            freq = torch.linspace(0, math.pi, size + 1)[:-1]
        cosine_filter = fftshift(torch.sin(freq))
        fourier_filter *= cosine_filter

    elif filter_name == "hamming":
        if use_GPU:
            fourier_filter *= fftshift(torch.hamming_window(size).cuda())
        else:
            fourier_filter *= fftshift(torch.hamming_window(size))

    elif filter_name == "hann":
        if use_GPU:
            fourier_filter *= fftshift(torch.hann_window(size).cuda())
        else:
            fourier_filter *= fftshift(torch.hann_window(size))
    elif filter_name is None:
        fourier_filter[:] = 1

    return fourier_filter.unsqueeze(-1).unsqueeze(0).unsqueeze(0)  # (1, 1, size, 1)


def fourier_filtered(radon_image, filter_name="ramp", diag_cal=False, use_GPU=False):
    """
        param:
            radon_image: 2D radon data, shape of (b, 1, w, angles)
        return:
            filtered radon data
    """
    filter_types = ('ramp', 'shepp-logan', 'cosine', 'hamming', 'hann', None)
    if filter_name not in filter_types:
        raise ValueError("Unknown filter: %s" % filter_name)

    batch, _, img_shape, _ = radon_image.shape

    if diag_cal:  # 对角投影时是否要扩大正弦的投影
        radon_image = sinogram_to_square(radon_image)
        img_shape = radon_image.shape[-2]

    # Resize image to next power of two (but no less than 64) for
    # Fourier analysis; speeds up Fourier and lessens artifacts
    projection_size_padded = max(64, int(2 ** math.ceil(math.log2(2 * img_shape))))
    pad_width = (0, 0, 0, projection_size_padded - img_shape)
    img = F.pad(radon_image, pad_width, mode='constant', value=0)

    # Apply filter in Fourier domain
    fourier_filter = get_fourier_filter(projection_size_padded, filter_name, use_GPU)
    projection = fft(img, dim=-2) * fourier_filter
    radon_filtered = torch.real(ifft(projection, dim=-2)[:, :, :img_shape, :])

    return radon_filtered  # (b, 1, w, angles)


# --- fbp by batch tensor --- #
def fbp_ts4(sinogram, angle_range=(0, 360), filter_name="hamming", use_GPU=False):  # (b, c, w, angles)

    sinogram_f = fourier_filtered(sinogram, filter_name, use_GPU=use_GPU)  # (b, c, w, angles)

    radon = torch.transpose(sinogram_f, dim0=-2, dim1=-1)  # (b, c, angles, w)
    b, c, angle_step, recon_size = radon.shape  # ((b, c, angle_step, w))
    iradon = torch.zeros((b, c, recon_size, recon_size))  # (b, c, w, w)
    if use_GPU:
        iradon = iradon.cuda()

    low, high = angle_range

    for step in range(angle_step):
        projection_expand_dim = radon[:, :, step, :].unsqueeze(2)  # (b, c, 1, w)

        projection_value_repeat = projection_expand_dim.repeat(1, 1, recon_size, 1)  # (b, c, w, w)
        iradon += rotate(projection_value_repeat, (
                (step * (high - low) / angle_step) + low), use_GPU)

    radius = recon_size // 2
    xprt, yprt = torch.meshgrid(torch.arange(end=recon_size), torch.arange(end=recon_size), indexing='ij')
    xprt = xprt - radius
    yprt = yprt - radius
    out_reconstruction_circle = (xprt ** 2 + yprt ** 2) > radius ** 2
    iradon[:, :, out_reconstruction_circle] = 0.

    return iradon


if __name__ == "__main__":
    pass
