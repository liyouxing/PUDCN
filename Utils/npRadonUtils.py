import numpy as np
from skimage.transform import radon, iradon, warp
from numpy.fft import fft, fftfreq, fftshift, ifft


def rotation_skimage(mat_data, theta=60.):
    padded_image = mat_data
    center = padded_image.shape[0] // 2

    rad_angle = np.deg2rad(theta)
    cos_a, sin_a = np.cos(rad_angle), np.sin(rad_angle)
    R = np.array([[cos_a, sin_a, -center * (cos_a + sin_a - 1)],
                  [-sin_a, cos_a, -center * (cos_a - sin_a - 1)],
                  [0, 0, 1]])
    rotated = warp(padded_image, R, clip=False)
    return rotated


def mat_radon_skimage(mat_data, angle_step=180, angle_range=(0, 180)):
    """
        use skimage.transform.radon for Radon transform on .mat data
        param:
            mat_data: 2-D numpy.float32 matrix
            angle_step: projection angle numbers
            angle_range: pro_angle range
        return:
            a sinogram, dim of (n, angle_step)
    """
    range_low, range_high = angle_range
    theta = np.linspace(range_low, range_high, num=angle_step, endpoint=False)
    res = radon(mat_data, theta, circle=False)
    return res


def mat_radon_adaptive(mat_data, angle_step=180, angle_range=(0, 180)):
    width = mat_data.shape[0]
    res = np.zeros((angle_step, width)) * 1.

    range_low, range_high = angle_range

    for step in range(angle_step):
        rotation = rotation_skimage(mat_data, ((step * (
                range_high - range_low) / angle_step) + range_low))

        res[step, :] = sum(rotation)  # sum of pixels in the first dimension

    return res.transpose()


def mat_radon_fixed(mat_data, angle_step=180, angle_range=(0, 180), high_value=None):
    high, width = mat_data.shape[0], mat_data.shape[1]

    if high_value is not None:
        high = high_value

    angle_bin = high / angle_step
    res = np.zeros((high, width)) * 1.
    range_low, range_high = angle_range

    for step in range(angle_step):
        rotation = rotation_skimage(mat_data, ((step * (
                range_high - range_low) / angle_step) + range_low))

        for index in range(int(step * angle_bin), int((step + 1) * angle_bin)):
            res[index, :] = sum(rotation)  # sum of pixels in the first dimension
    return res.transpose()


# -------------------------------
# use skimage pkg for radon and iradon
# -------------------------------
def img_radon_skimage(image, angle_step=180, angle_range=(0, 180)):
    """
        use skimage.transform.radon for Radon transform
    """
    range_low, range_high = angle_range
    theta = np.linspace(range_low, range_high, num=angle_step, endpoint=False)
    res = radon(image, theta, circle=True)

    return res


def fbp_skimage(radon_image, angle_range=(0, 360), filter_name='Ramp', interpolation="linear"):
    angle_step = len(radon_image[0])
    range_low, range_high = angle_range
    theta = np.linspace(range_low, range_high, num=angle_step, endpoint=False)
    res = iradon(radon_image, theta, filter_name=filter_name, interpolation=interpolation)
    return res


# --- fourier filters --- #
def sinogram_to_square(sinogram):
    diagonal = int(np.ceil(np.sqrt(2) * sinogram.shape[0]))
    pad = diagonal - sinogram.shape[0]
    old_center = sinogram.shape[0] // 2
    new_center = diagonal // 2
    pad_before = new_center - old_center
    pad_width = ((pad_before, pad - pad_before), (0, 0))
    return np.pad(sinogram, pad_width, mode='constant', constant_values=0)


def get_fourier_filter(size, filter_name):
    n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=int),
                        np.arange(size / 2 - 1, 0, -2, dtype=int)))
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2

    # Computing the ramp filter from the fourier transform of its
    # frequency domain representation lessens artifacts and removes a
    # small bias as explained in [1], Chap 3. Equation 61
    fourier_filter = 2 * np.real(fft(f))  # ramp filter
    if filter_name == "ramp":
        pass
    elif filter_name == "shepp-logan":
        # Start from first element to avoid divide by zero
        omega = np.pi * fftfreq(size)[1:]
        fourier_filter[1:] *= np.sin(omega) / omega
    elif filter_name == "cosine":
        freq = np.linspace(0, np.pi, size, endpoint=False)
        cosine_filter = fftshift(np.sin(freq))
        fourier_filter *= cosine_filter
    elif filter_name == "hamming":
        fourier_filter *= fftshift(np.hamming(size))
    elif filter_name == "hann":
        fourier_filter *= fftshift(np.hanning(size))
    elif filter_name is None:
        fourier_filter[:] = 1

    return fourier_filter[:, np.newaxis]


def fourier_filtered(radon_image, filter_name="ramp", diag_cal=False):
    filter_types = ('ramp', 'shepp-logan', 'cosine', 'hamming', 'hann', None)
    if filter_name not in filter_types:
        raise ValueError("Unknown filter: %s" % filter_name)

    img_shape = radon_image.shape[0]

    if diag_cal:  # 对角投影时是否要扩大正弦的投影
        radon_image = sinogram_to_square(radon_image)
        img_shape = radon_image.shape[0]

    # Resize image to next power of two (but no less than 64) for
    # Fourier analysis; speeds up Fourier and lessens artifacts
    projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * img_shape))))
    pad_width = ((0, projection_size_padded - img_shape), (0, 0))
    img = np.pad(radon_image, pad_width, mode='constant', constant_values=0)

    # Apply filter in Fourier domain
    fourier_filter = get_fourier_filter(projection_size_padded, filter_name)
    projection = fft(img, axis=0) * fourier_filter
    radon_filtered = np.real(ifft(projection, axis=0)[:img_shape, :])

    return radon_filtered


# --- fbp by numpy --- #
def fbp(radon_image, angle_range=(0, 360), filter_name='ramp'):
    """
        param:
            radon_image: shape of (width, angles)
            angle_range: rotation range
            filter_name: None, 'ramp', 'shepp-logan', 'cosine', 'hamming', 'hann'
        return: fbp result, shape of (width, width)
    """
    output_size = radon_image.shape[0]
    angle_step = radon_image.shape[1]

    radon_image = fourier_filtered(radon_image, filter_name)

    radon_image = np.transpose(radon_image)
    origin = np.zeros((angle_step, output_size, output_size))
    low, high = angle_range

    for step in range(angle_step):
        projection_value = radon_image[step, :]

        projection_value_expand_dim = np.expand_dims(projection_value, axis=0)
        projection_value_repeat = projection_value_expand_dim.repeat(output_size, axis=0)
        origin[step] = rotation_skimage(projection_value_repeat, -((step * (high - low) / angle_step) + low))

    iradon = np.sum(origin, axis=0)

    radius = output_size // 2
    xpr, ypr = np.mgrid[:output_size, :output_size] - radius
    out_reconstruction_circle = (xpr ** 2 + ypr ** 2) > radius ** 2
    iradon[out_reconstruction_circle] = 0.

    return iradon
