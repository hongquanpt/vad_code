import os
import torch
import random
import bisect
import collections
import torch.nn as nn


# ===========================================================
# Get 5 frames for training: the 4 first frames for model,  the last frame is called predicted frame.
# ===========================================================
# for models: MNAD (channel=3)
def decode_clip_cat(data, num_frames=5, channel=3):
    """
    Decode a clip which is concatenated by num_frames frames
    param data: a tensor (i.e, torch.Size([bz, 15, 256, 256]))
    param num_frames: the number of frames in a clip
    param channel: the channel color
    """
    # if model_type == 'prediction':
    seq_len = (num_frames - 1) * channel  # i.e, (5-1)*3 = 12
    inputs = data[:, :seq_len]  # torch.Size([bz, 12, 256, 256])
    target = data[:, seq_len:]  # torch.Size([bz, 3, 256, 256])

    # Delete data to free up memory
    del data
    return inputs, target


def decode_clip_cat_last_input(data, num_frames=5, channel=3):
    """
    Decode a clip which is concatenated by num_frames frames
    param data: a tensor (i.e, torch.Size([bz, 15, 256, 256]))
    param num_frames: the number of frames in a clip
    param channel: the channel color
    """
    a = (num_frames - 2) * channel  # i.e, (5-2)*3 = 9
    b = (num_frames - 1) * channel  # i.e, (5-1)*3 = 12
    input_last = data[:, a:b]
    # if model_type == 'prediction':
    inputs = data[:, :b]  # torch.Size([bz, 12, 256, 256])
    target = data[:, b:]  # torch.Size([bz, 3, 256, 256])

    return inputs, target, input_last


# for models: FastAno, channel=1(grayscale)
def decode_clip_stack(data, num_frames=6):
    """
    Decode a clip which is stacked by num_frames frames
    param data: a tensor (i.e, torch.Size([bz, 6, 1, 256, 256]))
    param num_frames: the number of frames in a clip
    """
    # if model_type == 'prediction':
    seq_len = (num_frames - 1)  # i.e, (6-1) = 5
    inputs = data[:, :seq_len]  # torch.Size([bz, 5, 1, 256, 256])
    target = data[:, seq_len:]  # torch.Size([bz, 1, 1, 256, 256])

    return inputs, target


# for models: ASTNet
def decode_clip_list(data):
    """
    Decode a clip which is a list of num_frames frames
    param data: is a list (i.e, len(data) = 5; data[0].shape = torch.Size([1, 3, 256, 256])
    """
    # if model_type == 'prediction':
    inputs = data[:-1]  # a list of the first (num_frames-1)^th elements
    target = data[-1]  # a list of the last element

    return inputs, target


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
def point_score(output, imgs):
    loss_func_mse = nn.MSELoss(reduction='none')
    error = loss_func_mse((output[0] + 1) / 2, (imgs[0] + 1) / 2)
    normal = (1 - torch.exp(-error))
    score = (torch.sum(normal * loss_func_mse((output[0] + 1) / 2, (imgs[0] + 1) / 2)) / torch.sum(normal)).item()
    return score


# ===========================================================
def make_dir_path(root_dir, dir_name):
    dir_path = os.path.join(root_dir, dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        print(f'=> Creating a new directory: {dir_path}')
    else:
        print(f"- The directory [{dir_path}] existed!!!")

    if not os.path.exists(dir_path):
        raise Exception('Something wrong in creating dir_path: {}'.format(dir_path))
    return dir_path
