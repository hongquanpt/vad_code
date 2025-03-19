import numpy as np
from random import shuffle

# https://docs.python.org/3/library/random.html#random.uniform
import random
import bisect
import collections

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
# rng = np.random.RandomState(2020)

# https://pypi.org/project/opensimplex/
# https://github.com/lmas/opensimplex
from .noise import Simplex_CLASS


# ================================================================
# TMT, SRT
# ================================================================
def get_pos_patch(width=256, height=256, patch_size=90, h_cut_size=30):
    """
    Get a position of patch
        
    param W: width of image
    param H: height of image
    param patch_size: size of patch
    param h_cut_size: height of upper and lower discarded image blocks
    """
    minW = 0
    maxW = width - patch_size  # 360-90=270
    posW = random.randrange(minW, maxW)  # (0, 270)

    minH = h_cut_size
    maxH = height - patch_size - h_cut_size  # 240-90-30 = 120
    posH = random.randrange(minH, maxH)  # (30, 120)
    # print(f'(posW, posH) = {posW}, {posH}\n')
    return posW, posH


def get_clip_TMT(clip, patch_size_isRandom, patch_size_range, patch_size, width, height, h_cut_size):
    """
    Apply patches on clip with TMT transform (skip the last frame)
        
    param clip: a list of tensor images
    """
    if patch_size_isRandom:
        start = patch_size_range[0]
        stop = patch_size_range[1]
        step = patch_size_range[2]
        patch_size = random.randrange(start, stop, step)

    posW, posH = get_pos_patch(width, height, patch_size, h_cut_size)

    # list of image ids, skip the last frame in clip (for prediction)
    shuffled_ids = list(range(len(clip) - 1))
    shuffle(shuffled_ids)

    clip_TMT = []
    num_frames = len(clip)

    # skip the last frame in clip
    for i in range(num_frames - 1):
        # ground truth image: torch.Size([3, 256, 256]) 
        gt_image = clip[i].clone()

        # shuffled image and crop a patch of image
        shuffled_image = clip[shuffled_ids[i]].clone()
        ano_patch = shuffled_image[:, posH:(posH + patch_size), posW:(posW + patch_size)]

        # paste the patch to original image
        gt_image[:, posH:(posH + patch_size), posW:(posW + patch_size)] = ano_patch
        clip_TMT.append(gt_image)

    # add the ground truth last frame
    clip_TMT.append(clip[num_frames - 1])
    return clip_TMT


def get_clip_SRT(clip, patch_size_isRandom, patch_size_range, patch_size, width, height, h_cut_size):
    """
    Apply patches on clip with SRT transform (skip the last frame)
        
    param clip: a list of frames
    """
    if (patch_size_isRandom):
        start = patch_size_range[0]
        stop = patch_size_range[1]
        step = patch_size_range[2]
        patch_size = random.randrange(start, stop, step)

    posW, posH = get_pos_patch(width, height, patch_size, h_cut_size)

    clip_SRT = []
    num_frames = len(clip)

    # skip the last frame in clip
    for i in range(num_frames - 1):
        # ground truth image: torch.Size([3, 256, 256]) 
        gt_image = clip[i].clone()

        # crop and rotate a patch
        ano_patch = gt_image[:, posH:(posH + patch_size), posW:(posW + patch_size)]
        ano_patch = torch.rot90(input=ano_patch, k=random.randrange(0, 4), dims=(1, 2))

        # paste the patch to original image
        gt_image[:, posH:(posH + patch_size), posW:(posW + patch_size)] = ano_patch
        clip_SRT.append(gt_image)

    # add the ground truth last frame
    clip_SRT.append(clip[num_frames - 1])  # the ground truth last frame
    return clip_SRT


def get_clip_patch_cifar(clip, cifar_img, patch_size, width, height, h_cut_size):
    """
   Apply a cifar patch on clip (skip the last frame)
        
    param clip: a list of frames
    """
    posW, posH = get_pos_patch(width, height,
                               patch_size, h_cut_size)
    clip_patch_cifar = []
    num_frames = len(clip)

    # skip the last frame in clip
    for i in range(num_frames - 1):
        # ground truth image: torch.Size([3, 256, 256]) 
        gt_image = clip[i].clone()

        # cifar_img: torch.Size([3, 32, 32])
        ano_patch = cifar_img.squeeze(0)

        # paste the patch to original image
        gt_image[:, posH:(posH + ano_patch.shape[1]), posW:(posW + ano_patch.shape[2])] = ano_patch
        clip_patch_cifar.append(gt_image)

    # add the ground truth last frame
    clip_patch_cifar.append(clip[num_frames - 1])
    return clip_patch_cifar


def get_clip_gaussian_noise(clip, noise_isAdded, mean=0, std=0.1):
    """
    Apply Gaussian noise on clip (skip the last frame)
        
    param clip: a list of frames
    """
    if noise_isAdded is not True:
        return clip
    clip_noise = []
    num_frames = len(clip)

    # skip the last frame in clip
    for i in range(num_frames - 1):
        # ground truth image: torch.Size([3, 256, 256]) 
        gt_image = clip[i].clone()

        # tensor filled with random numbers from a standard_normal_distribution 
        std_norm_image = torch.randn(gt_image.size())
        std_norm_image = std_norm_image.to(gt_image.device)

        # add the standard_normal_distribution image to original image with (std, mean)
        gt_image = (gt_image + std_norm_image * std + mean)
        clip_noise.append(gt_image)

    # add the ground truth last frame
    clip_noise.append(clip[num_frames - 1])
    return clip_noise


def get_clip_simplex_noise(clip, noise_isAdded, patch_size, width, height, h_cut_size):
    """
    Apply Simplex noise on clip (skip the last frame)
        
    param clip: a list of frames
    """
    if noise_isAdded is not True:
        return clip
    print('add simplex_noise')
    posW, posH = get_pos_patch(width, height, patch_size, h_cut_size)
    clip_noise = []
    num_frames = len(clip)

    # skip the last frame in clip
    for i in range(num_frames - 1):
        # ground truth image: torch.Size([3, 256, 256]) 
        gt_image = clip[i].clone()

        # add the simplex noise to original image
        simplex_noise = Simplex_CLASS()
        simplex_noise_patch = simplex_noise.rand_3d_octaves((3, patch_size, patch_size), 6, 0.6)  # shape=(3,H,W)

        gt_image[:, posH:(posH + patch_size), posW:(posW + patch_size)] = torch.from_numpy(simplex_noise_patch)
        clip_noise.append(gt_image)

    # add the ground truth last frame
    clip_noise.append(clip[num_frames - 1])
    return clip_noise


# ================================================================
# Gaussian noise, Simplex noise
# ================================================================
def add_gaussian_noise_cv2(img, mean=0, var=0.1):
    H, W, C = img.shape
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (H, W, C))
    gauss = gauss.reshape(H, W, C)
    noisy = img + gauss
    return noisy


def add_poisson_noise_cv2(img, mean=0, var=0.1):
    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(img * vals) / float(vals)
    return noisy


def apply_gaussian_noise(img_tensor, mean=0., std=1.):
    """
    Add Gaussian noise to img_tensor
    
    param mean: 0.0
    param std: 0.01; 0.05; 0.1; 0.5; 1
    """
    # torch.randn: generate random numbers from standard normal distribution
    std_norm_img = torch.randn(img_tensor.size())
    std_norm_img = std_norm_img.to(img_tensor.device)
    img_noise = (img_tensor + std_norm_img * std + mean)
    return img_noise


def apply_fusion(img_tensor_1, img_tensor_2):
    img_fusion = (img_tensor_1 + img_tensor_2) / 2
    return img_fusion


def add_simplex_noise(img_tensor, H=256, W=256):
    simplex_noise = Simplex_CLASS()

    h_noise = np.random.randint(10, int(H // 8))  # random integers from 10 to 32
    w_noise = np.random.randint(10, int(W // 8))
    start_h_noise = np.random.randint(1, H - h_noise)
    start_w_noise = np.random.randint(1, W - w_noise)
    noise_size = (h_noise, w_noise)
    '''
    simplex_noise = simplex_noise.rand_3d_octaves((3, *noise_size), 6, 0.6)
    init_zero = np.zeros((H,W,3))
    init_zero[start_h_noise: start_h_noise + h_noise, start_w_noise: start_w_noise+w_noise, :] = 0.2 * simplex_noise.transpose(1,2,0) # (H,W,3)
    img_noise = img_tensor + torch.from_numpy(init_zero).permute(2,0,1) #(3,H,W)
    '''
    simplex_noise = simplex_noise.rand_3d_octaves((3, *noise_size), 6, 0.6)
    init_zero = np.zeros(shape=(H, W, 3))
    init_zero[start_h_noise: start_h_noise + h_noise, start_w_noise: start_w_noise + w_noise,
    :] = 0.2 * simplex_noise.transpose(1, 2, 0)  # (H,W,3)
    img_noise = img_tensor + torch.from_numpy(init_zero).permute(2, 0, 1)  # (3,H,W)
    return img_noise


# ================================================================
# ===========================================================
# Functions for selecting elements in list with different probability
# https://stackoverflow.com/questions/4113307/pythonic-way-to-select-list-elements-with-different-probability
# ===========================================================
# cdf (Cumulative distribution function)
def cdf(weights):
    total = sum(weights)
    result = []
    cumsum = 0
    for w in weights:
        cumsum += w
        result.append(cumsum / total)
    return result


def choice(population, weights):
    assert len(population) == len(weights)
    cdf_vals = cdf(weights)
    x = random.random()
    idx = bisect.bisect(cdf_vals, x)
    return population[idx]


# population is a list of elements (with any size)
def get_lists_dict_with_prob(population=None, weights=None, num_elements=1000):
    if weights is None:
        weights = [0.01, 0.99]
    if population is None:
        population = ['anomaly', 'normal']
    assert len(population) == len(weights)
    assert sum(weights) == 1.0
    lists_dict = collections.defaultdict(list)

    for i in range(0, num_elements):
        lists_dict[choice(population, weights)].append(i)

    # print(lists_dict)
    # anomaly_ids_list = lists_dict.get('anomaly')
    return lists_dict


# ===========================================================
# ================================================================
def patch_cifar_smooth(img, cifar_img, max_size=0.2, h=256, w=256, max_move=0):
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
        img = mask * cifar_patch.to(img.device) + (1 - mask) * img
    else:
        mask = []
        for frame_idx in range(img.size(1)):
            delta_x = np.random.randint(-max_move, max_move + 1)
            delta_y = np.random.randint(-max_move, max_move + 1)
            mask_ = torch.tensor(_get_gaussian_mask(x_mu, y_mu, x_sigma, y_sigma, h, w)).to(img.device).float()

            img[:, frame_idx] = mask_ * cifar_patch.to(img.device) + (1 - mask_) * img[:, frame_idx]
            mask.append(mask_)

            x_mu = min(max(0, x_mu + delta_x), w)
            y_mu = min(max(0, y_mu + delta_y), h)

        mask = torch.stack(mask, dim=0)

    return img, mask


# 'pseudo anomaly using a patch (SmoothMixS)'
def patch_SmoothMixS(img, cifar_img, max_size=0.2, c=3, h=256, w=256, max_move=0):
    """
    param img: is a tensor with 4D shape, for example, (1, 15, 256, 256)
    param cifar_img: is a tensor with 3D shape, for example, (3, 32, 32)
    """
    assert 0 <= max_size <= 1

    pil_img = transforms.ToPILImage()(cifar_img)
    pil_img = transforms.Grayscale(num_output_channels=1)(pil_img)
    cifar_img = transforms.ToTensor()(pil_img)

    cifar_img = transforms.Normalize(mean=[0.5], std=[0.5])(cifar_img)
    cifar_patch = F.interpolate(cifar_img.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=True)
    cx, cy = np.random.randint(w), np.random.randint(h)

    cut_w = max(10, int(np.random.uniform(high=max_size) * w))
    cut_h = max(10, int(np.random.uniform(high=max_size) * h))
    if max_move == 0:
        mask = torch.tensor(_get_smoothborder_mask(cx, cy, cut_h, cut_w, h, w)).to(img.device).float()
        img = mask * cifar_patch.to(img.device) + (1 - mask) * img
    else:
        mask = []
        # print('img.size(1) = ', img.size(1)) # 15; and c = 3
        for frame_idx in range(0, img.size(1), c):
            delta_x = np.random.randint(-max_move, max_move + 1)
            delta_y = np.random.randint(-max_move, max_move + 1)
            mask_ = torch.tensor(_get_smoothborder_mask(cx, cy, cut_h, cut_w, h, w)).to(img.device).float()

            # torch.Size([1, 3, 256, 256])
            # print('img[:, frame_idx:frame_idx+c].shape =', img[:, frame_idx:frame_idx+c].shape)
            img[:, frame_idx:frame_idx + c] = mask_ * cifar_patch.to(img.device) + (1 - mask_) * img[:, frame_idx:frame_idx + c]
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
    return img4  # , lam


# a = _get_smoothborder_mask(0.5, 256, 256)
def apply_camera_shake(clip, c=3, rotation_range=(1, 5), translation_range=(0.01, 0.05)):
    """
    param clip: is a tensor with 3D shape, for example, (15, 256, 256)
    param rotation_range: range of degrees to select from
    param translation_range: maximum absolute fraction for horizontal and vertical translations.
    """
    print('rotation_range = ', rotation_range)
    print('translation_range = ', translation_range)
    shaken_clip = []
    # print('clip.size(0) = ', clip.size(0)) # 15; and c = 3
    for frame_idx in range(0, clip.size(0), c):
        tensor_frame = clip[frame_idx:frame_idx + c]
        # Define the PyTorch transformation for random translation and rotation
        shake = transforms.Compose([
            transforms.RandomAffine(degrees=rotation_range, translate=translation_range),
        ])
        shaken_clip.append(shake(tensor_frame))

    result = torch.cat(shaken_clip, dim=0)
    return result
