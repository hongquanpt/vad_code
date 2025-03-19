import numpy as np
from PIL import Image
import cv2
import random

import random
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
#rng = np.random.RandomState(2020)

def convert_0_255_to_minus1_1(img):
    # convert numpy.ndarray (img) from [0, 255] to [-1, 1]
    img = (img / 127.5) - 1.0
    return img

# https://stackoverflow.com/questions/50966204/convert-images-from-1-1-to-0-255
def convert_minus1_1_to_0_255(img):
    # convert numpy.ndarray (img) from [-1, 1] to [0, 255]
    norm_image = cv2.normalize(img, None, alpha = 0, beta = 255, 
                               norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    return norm_image
    
def convert_tensor_minus1_1_to_0_255(img_tensor):
    # convert tensor (img) from [-1, 1] to [0, 255]
    # 1. convert tensor to numpy.ndarray
    img = img_tensor.numpy()
    
    norm_image = cv2.normalize(img, None, alpha = 0, beta = 255, 
                               norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    return norm_image

def np_load_frame(filename, resize_height, resize_width):
    """
    Load image path and convert it to numpy.ndarray. 
    Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    param filename: the full path of image
    param resize_height: resized height
    param resize_width: resized width
    :return: numpy.ndarray
    """
    image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized

def open_image_PIL(image_path):
    img = Image.open(image_path)
    return img, img.height, img.width

def open_image_cv2(image_path):
    img = cv2.imread(image_path)
    return img, img.shape[0], img.shape[1]


def pseudoanomaly_patch_cifar_smooth(img, cifar_img, max_size, h, w, dataset, max_move=0):
    assert 0 <= max_size <= 1

    pil_img = transforms.ToPILImage()(cifar_img)
    pil_img = transforms.Grayscale(num_output_channels=1)(pil_img)
    cifar_img = transforms.ToTensor()(pil_img)

    cifar_img = transforms.Normalize(mean=[0.5], std=[0.5])(cifar_img)

    cifar_patch = F.interpolate(cifar_img.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=True)

    x_mu, y_mu = random.randint(0, w), random.randint(0, h)
    x_sigma = max(10, int(np.random.uniform(high=max_size) * w))
    y_sigma = max(10, int(np.random.uniform(high=max_size) * h))
    if max_move == 0:
        mask = torch.tensor(_get_gaussian_mask(x_mu, y_mu, x_sigma, y_sigma, h, w)).to(img.device).float()
        img = mask * cifar_patch.to(img.device) + (1-mask) * img
    else:
        mask = []
        for frame_idx in range(img.size(1)):
            delta_x = np.random.randint(-max_move, max_move + 1)
            delta_y = np.random.randint(-max_move, max_move + 1)
            mask_ = torch.tensor(_get_gaussian_mask(x_mu, y_mu, x_sigma, y_sigma, h, w)).to(img.device).float()

            img[:, frame_idx] = mask_ * cifar_patch.to(img.device) + (1-mask_) * img[:, frame_idx]
            mask.append(mask_)

            x_mu = min(max(0, x_mu + delta_x), w)
            y_mu = min(max(0, y_mu + delta_y), h)

        mask = torch.stack(mask, dim=0)

    return img, mask

def pseudoanomaly_patch_cifar_smoothborder(img, cifar_img, max_size, h, w, dataset, max_move=0):
    assert 0 <= max_size <= 1

    pil_img = transforms.ToPILImage()(cifar_img)
    pil_img = transforms.Grayscale(num_output_channels=1)(pil_img)
    cifar_img = transforms.ToTensor()(pil_img)

    cifar_img = transforms.Normalize(mean=[0.5], std=[0.5])(cifar_img)
    cifar_patch = F.interpolate(cifar_img.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=True)
    cx, cy = np.random.randint(w), np.random.randint(h)

    cut_w= max(10, int(np.random.uniform(high=max_size) * w))
    cut_h = max(10, int(np.random.uniform(high=max_size) * h))
    if max_move == 0:
        mask = torch.tensor(_get_smoothborder_mask(cx, cy, cut_h, cut_w, h, w)).to(img.device).float()
        img = mask * cifar_patch.to(img.device) + (1-mask) * img

    else:
        mask = []
        for frame_idx in range(img.size(1)):
            delta_x = np.random.randint(-max_move, max_move + 1)
            delta_y = np.random.randint(-max_move, max_move + 1)
            mask_ = torch.tensor(_get_smoothborder_mask(cx, cy, cut_h, cut_w, h, w)).to(img.device).float()

            img[:, frame_idx] = mask_ * cifar_patch.to(img.device) + (1 - mask_) * img[:, frame_idx]
            mask.append(mask_)

            cx = min(max(0, cx + delta_x), w)
            cy = min(max(0, cy + delta_y), h)

        mask = torch.stack(mask, dim=0)

    return img, mask

def _get_gaussian_mask(x_mu, y_mu, x_sigma, y_sigma, h, w):
    x, y = np.arange(w), np.arange(h)

    # x_mu, y_mu = random.randint(0, w), random.randint(0, h)
    # x_sigma = max(10, int(np.random.uniform(high=max_size) * w))
    # y_sigma = max(10, int(np.random.uniform(high=max_size) * h))

    gx = np.exp(-(x - x_mu) ** 2 / (2 * x_sigma ** 2))
    gy = np.exp(-(y - y_mu) ** 2 / (2 * y_sigma ** 2))
    g = np.outer(gx, gy)
    # g /= np.sum(g)  # normalize, if you want that

    # sum_g = np.sum(g)
    # lam = sum_g / (w * h)
    # print(lam)

    # plt.imshow(g, interpolation="nearest", origin="lower")
    # plt.show()
    # g = np.dstack([g, g, g])

    return g

def pseudoanomaly_fusion(img, seq, mask = 0.5):

    img = mask * seq.to(img.device) + (1-mask) * img
    return img

# mean = 0.
# std = 0.01; 0.05; 0.1; 0.5; 1
def pseudoanomaly_noise(img, mean = 0., std = 1.):
    
    img = img + torch.randn(img.size()) * std + mean
    return img

def _get_smoothborder_mask(cx, cy, Cut_h, Cut_w, h, w):
    lam = np.random.beta(1, 1)
    percentage = 0.1
    cut_rat = np.sqrt(1. - lam)

    # Cut_w = min(np.int(max_size*w), max(2, np.int(w * cut_rat)))
    # Cut_h = min(np.int(max_size*h), max(2, np.int(h * cut_rat)))

    # cx, cy = np.random.randint(w), np.random.randint(h)

    bbx1 = np.clip(cx - Cut_w // 2, 0, w)  # top left x
    bby1 = np.clip(cy - Cut_h // 2, 0, h)  # top left y
    bbx2 = np.clip(cx + Cut_w // 2, 0, w)  # bottom right x
    bby2 = np.clip(cy + Cut_h // 2, 0, h)  # bottom right y

    img = np.zeros((w, h))
    img2, img3 = np.ones_like(img), np.ones_like(img)
    img[bbx1:bbx2, bby1:bby2] = img2[bbx1:bbx2, bby1:bby2]

    lo = bbx1 - (Cut_w // 2) * percentage  # left side: beginning linear from 0
    li = bbx1  # + (Cut_w // 2) * percentage  # left side: end of linear at 1
    ri = bbx2  # - (Cut_w // 2) * percentage  # right : start linear from 1
    ro = bbx2 + (Cut_w // 2) * percentage  # right: end linear at 0

    to = bby1 - (Cut_h // 2) * percentage  # top: start linear from 0
    ti = bby1  # + (Cut_h // 2) * percentage  # top: end linear at 1
    bi = bby2  # - (Cut_h // 2) * percentage  # bottom: start linear from 1
    bo = bby2 + (Cut_h // 2) * percentage  # bottom: end linear at 0

    # glx = lambda x: ((x - lo) / (li - lo))
    # grx = lambda x: (-(x - ro) / (ro - ri))
    # gtx = lambda x: ((x - to) / (ti - to))
    # gbx = lambda x: (-(x - bo) / (bo - bi))

    for i in range(w):
        for j in range(h):
            if i < cx:
                img2[j][i] = ((i - lo) / (li - lo))  # linear going up
            else:
                img2[j][i] = (-(i - ro) / (ro - ri))  # linear going down
            if j < cy:
                img3[j][i] = ((j - to) / (ti - to))
            else:
                img3[j][i] = (-(j - bo) / (bo - bi))

    img2[img2 < 0] = 0
    img2[img2 > 1] = 1

    img3[img3 < 0] = 0
    img3[img3 > 1] = 1

    # plt.figure()
    # plt.subplot(131)
    # plt.imshow(img2)
    # # plt.show()
    # plt.subplot(132)
    # plt.imshow(img3)
    # # plt.show()
    img4 = np.multiply(img2, img3)
    # sum_img4 = np.sum(img4)
    # lam = sum_img4 / (w * h)

    # plt.subplot(133)
    # plt.imshow(img4)
    # plt.show()
    return img4  #, lam

# a = _get_smoothborder_mask(0.5, 256, 256)