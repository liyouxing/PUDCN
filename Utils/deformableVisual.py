from models.PUDCN import PUDCN
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms as T


def scale_transformation(img, scale_factor=1.0, borderValue=0):
    img_h, img_w = img.shape[0:2]

    cx = img_w // 2
    cy = img_h // 2

    tx = cx - scale_factor * cx
    ty = cy - scale_factor * cy

    # scale matrix
    sm = np.float32([[scale_factor, 0, tx],
                     [0, scale_factor, ty]])  # [1, 0, tx], [1, 0, ty]

    img = cv2.warpAffine(img, sm, (img_w, img_h), borderValue=borderValue)
    return img


def rotation_transformation(img, angle=3., borderValue=0):
    img_h, img_w = img.shape[0:2]
    rm = cv2.getRotationMatrix2D((img_w // 2, img_h // 2), angle=angle, scale=1.0)  # rotation matrix
    img = cv2.warpAffine(img, rm, (img_w, img_h), flags=cv2.INTER_LINEAR, borderValue=borderValue)
    return img


def random_rotation(img, scale_factor=1.0, borderValue=0):
    img_h, img_w = img.shape[0:2]

    cx = img_w // 2
    cy = img_h // 2

    tx = cx - scale_factor * cx
    ty = cy - scale_factor * cy

    # scale matrix
    sm = np.float32([[scale_factor, 0, tx],
                     [0, scale_factor, ty]])  # [1, 0, tx], [1, 0, ty]

    img = cv2.warpAffine(img, sm, (img_w, img_h), borderValue=borderValue)
    return img


def plot_offsets(img, save_output, roi_x, roi_y):
    cv2.circle(img, center=(roi_x, roi_y), color=(0, 255, 0), radius=1, thickness=-1)
    input_img_h, input_img_w = img.shape[:2]
    for offsets in save_output.outputs:
        offset_tensor_h, offset_tensor_w = offsets.shape[2:]
        resize_factor_h, resize_factor_w = input_img_h / offset_tensor_h, input_img_w / offset_tensor_w

        offsets_y = offsets[:, ::2]
        offsets_x = offsets[:, 1::2]

        grid_y = np.arange(0, offset_tensor_h)
        grid_x = np.arange(0, offset_tensor_w)

        grid_x, grid_y = np.meshgrid(grid_x, grid_y)

        sampling_y = grid_y + offsets_y.detach().cpu().numpy()
        sampling_x = grid_x + offsets_x.detach().cpu().numpy()

        sampling_y *= resize_factor_h
        sampling_x *= resize_factor_w

        sampling_y = sampling_y[0]  # remove batch axis
        sampling_x = sampling_x[0]  # remove batch axis

        sampling_y = sampling_y.transpose(1, 2, 0)  # c, h, w -> h, w, c
        sampling_x = sampling_x.transpose(1, 2, 0)  # c, h, w -> h, w, c

        sampling_y = np.clip(sampling_y, 0, input_img_h)
        sampling_x = np.clip(sampling_x, 0, input_img_w)

        sampling_y = cv2.resize(sampling_y, dsize=None, fx=resize_factor_w, fy=resize_factor_h)
        sampling_x = cv2.resize(sampling_x, dsize=None, fx=resize_factor_w, fy=resize_factor_h)

        sampling_y = sampling_y[roi_y, roi_x]
        sampling_x = sampling_x[roi_y, roi_x]

        for y, x in zip(sampling_y, sampling_x):
            y = round(y)
            x = round(x)
            cv2.circle(img, center=(x, y), color=(0, 0, 255), radius=1, thickness=-1)


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = PUDCN()
model = model.to(device)

# checkpoint = torch.load("../log_train/RME22000/12-02/net_params_119.tar")  # load .tar file
# model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})

save_output = SaveOutput()

for name, layer in model.named_modules():
    # print(name)
    if "dec2.3.branch1.3.offset_conv" in name and isinstance(layer, nn.Conv2d):
        layer.register_forward_hook(save_output)

image = cv2.imread("../results/RME12000/bem/000011.bmp", flags=cv2.IMREAD_GRAYSCALE)
input_img_h, input_img_w = image.shape

with torch.no_grad():
    # image = scale_transformation(image, scale_factor=scale_factors[scale_idx_factor])
    # image = rotation_transformation(image, angle=rotation_factors[rotation_idx_factor])
    # scale_idx_factor = (scale_idx_factor + 1) % len(scale_factors)
    # rotation_idx_factor = (rotation_idx_factor + 1) % len(rotation_factors)

    image_tensor = torch.from_numpy(image) / 255.
    image_tensor = image_tensor.view(1, 1, input_img_h, input_img_w)
    # image_tensor = T.Normalize((0.1307,), (0.3081,))(image_tensor)
    image_tensor = (image_tensor + 3.1416) / (3.1416 * 2)
    image_tensor = image_tensor.to(device)
    model.eval()
    out = model(image_tensor)

    image = np.repeat(image[..., np.newaxis], 3, axis=-1)
    roi_y, roi_x = input_img_h // 2, input_img_w // 2
    plot_offsets(image, save_output, roi_x=roi_x, roi_y=roi_y)

    save_output.clear()
    image = cv2.resize(image, dsize=(512, 512))
    cv2.imshow("image", image)
    key = cv2.waitKey(100000)

