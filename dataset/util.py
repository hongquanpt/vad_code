import glob
import random
from collections import OrderedDict
import matplotlib.pyplot as plt
import natsort
import numpy as np
import torchvision.transforms as transforms
import sys
import os

from PIL import Image
import torch
from .image_reader import CV2_imread, CV2_load_image

sys.path.append('../')
# from os_lib import *  # for test when call from another python file
# from .os_lib import *  # for implement

''' 
1) PIL lib:
https://github.com/uploadcare/pillow-simd
How to install pillow-simd (replace pillow) for speed up image processing
    $ pip uninstall pillow
    $ CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

2) transforms. store collections of data:
- We can use: List, Tuple, Set, and Dictionary.
    mylist = ["apple", "banana", "cherry"]
    mytuple = ("apple", "banana", "cherry")
    myset = {"apple", "banana", "cherry"}
    thisdict = {
                "brand": "Ford",
                "model": "Mustang",
                "year": 1964
                }
'''


def is_windows_path(path):
    return "\\" in path


def is_linux_path(path):
    return "/" in path


def make_power_2(n, base=32.0):
    """
    Xuất ra kích thước ảnh luôn là bội số của 2
    """
    return int(round(n / base) * base)


# be used in ASTNet
def get_transform(size,
                  method=transforms.InterpolationMode.BICUBIC,
                  channel=1,  # 1 (grayscale) or 3 (RGB)
                  is_toTensor=True,
                  is_normalized=True):
    """
    Returns a transform 
    - is_normalized=True: convert (PIL image) to [-1,1] (ASTNet, MNAD)
    - is_normalized=False: convert (PIL image) to [0,1]
    PyTorch uses channels-first by default (Channel, Height, Width)
    """
    w, h = size
    # new_size = [make_power_2(w), make_power_2(h)]
    new_size = [w, h]

    transform_list = [transforms.Resize(new_size, method)]
    if channel == 1:
        transform_list += [transforms.Grayscale()]
    if is_toTensor:  # Scales data into [0,1]
        transform_list += [transforms.ToTensor()]
    if is_normalized:  # Scale between [-1, 1]
        # Normalize a tensor image with mean and standard deviation.
        # This transform does not support PIL Image. 
        # mean: Sequence of means for each channel.
        # std: Sequence of standard deviations for each channel.
        # This transform will normalize each channel of the input torch.*Tensor 
        # i.e., output[channel] = (input[channel] - mean[channel]) / std[channel]
        if channel == 1:
            transform_list += [transforms.Normalize(mean=0.5,
                                                    std=0.5)]
        else:
            transform_list += [transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                    std=(0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __scale_image(img, size, method=transforms.InterpolationMode.BICUBIC):
    w, h = size
    return img.resize((w, h), method)


# ===========================================================
def is_exists_dir(directory_path):
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        print(f"The directory '{directory_path}' exists and is a directory.")
        return True
    else:
        print(f"The directory '{directory_path}' does not exist or is not a directory.")
        return False


def is_exists_file(file_path):
    if os.path.exists(file_path) and os.path.isfile(file_path):
        print(f"The file '{file_path}' exists.")
        return True
    else:
        print(f"The file '{file_path}' does not exist.")
        return False


def delete_file(file_path):
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"The file '{file_path}' has been deleted.")
    except OSError as e:
        print(f"Error deleting the file '{file_path}': {str(e)}")


def delete_files_in_dir(directory_path):  # delete all files in a directory
    # Check if the directory exists
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        # List all files in the directory
        file_list = os.listdir(directory_path)

        # Iterate over the files and delete them
        for file_name in file_list:
            file_path = os.path.join(directory_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                else:
                    print(f"Skipped: {file_path} (not a file)")
            except Exception as e:
                print(f"Error deleting {file_path}: {str(e)}")
    else:
        print(f"The directory '{directory_path}' does not exist or is not a directory.")


# ===========================================================
def get_videos(root):
    """
    Return:
    - videos: is a list of files by video in a dataset (ped2, avenue, shanghaitech)
    - total_frames: total of frames in dataset
    For example:
        videos = [['/home/dataset/ped2/training/01/000.jpg', 
                    '/home/dataset/ped2/training/01/001.jpg',..,
                    '/home/dataset/ped2/training/01/119.jpg'
                  ], 
                  [..], 
                  [..]]
    """
    is_exists_dir(root)

    include_ext = [".png", ".jpg", "jpeg", ".bmp"]
    # https://www.geeksforgeeks.org/os-walk-python/
    # os.walk() generates the file names in a directory tree by walking the tree either top-down or bottom-up.

    # dirs = [x[0] for x in os.walk(root, followlinks=True)] # 2023-12-06
    # dirs = [os.path.join(root, x) for x in os.listdir(root)]

    # 01. dirs: a list containing the names of the entries in the directory
    dirs = []
    for name in os.listdir(root):
        fdir = os.path.join(root, name)
        if os.path.isdir(fdir) and not name.startswith('.'):
            dirs.append(fdir)

    dirs = natsort.natsorted(dirs)  # sort a list
    # print('dirs = ', dirs)
    print('len(dirs) =', len(dirs))
    '''
        dirs is a list of directory in training set
        dirs = ['/home/dataset/ped2/training', 
                '/home/dataset/ped2/training/01',..,
                '/home/dataset/ped2/training/16']
    '''

    # 02. dataset is a list of files and directory in training set
    videos = []
    total_frames = 0
    for fdir in dirs:  # fdir = video_path
        video = []
        # os.listdir(): get the list of all files and directories in the directory.
        for name in natsort.natsorted(os.listdir(fdir)):
            # if it is a file with include_ext = [".png", ".jpg", "jpeg", ".bmp"]
            if os.path.isfile(os.path.join(fdir, name)) \
                    and not name.startswith('.') \
                    and any([name.endswith(ext) for ext in include_ext]):
                video.append(os.path.join(fdir, name))
        if len(video) > 0:
            total_frames += len(video)
            videos.append(video)

    '''            
    datasets = [[os.path.join(fdir, name) for name in natsort.natsorted(os.listdir(fdir))
             if os.path.isfile(os.path.join(fdir, name))
             and not name.startswith('.')
             and any([name.endswith(ext) for ext in include_ext])]
            for fdir in dirs
        ]
    # because the 1st item in datasets is empty
    return [name for name in datasets if name] 
    '''
    # print('videos=', videos)
    return videos, total_frames


def extract_numeric_value_from_filename(file_name):
    # Split the file_name (001.jpg, 0001.jpg) into two parts using the dot as the separator
    parts = file_name.split(".")

    # Extract the first part and convert it to a number
    try:
        numeric_value = int(parts[0])
        return numeric_value
    except ValueError:
        print(f"Error: Unable to convert '{parts[0]}' to an integer.")
        return None


def get_videos_kf(root):
    """
    get_videos_kf() loai bo 1 trong 2 frames lien tiep trong dataset_kf (neu co)
    Return:
    - videos: is a list of files by video in a dataset (ped2, avenue, shanghaitech)
    - total_frames: total of frames in dataset
    For example:
        videos = [['/home/dataset/ped2/training/01/000.jpg',
                    '/home/dataset/ped2/training/01/001.jpg',..,
                    '/home/dataset/ped2/training/01/119.jpg'
                  ],
                  [..],
                  [..]]
    """
    is_exists_dir(root)

    include_ext = [".png", ".jpg", "jpeg", ".bmp"]
    # https://www.geeksforgeeks.org/os-walk-python/
    # os.walk() generates the file names in a directory tree by walking the tree either top-down or bottom-up.

    # dirs = [x[0] for x in os.walk(root, followlinks=True)] # 2023-12-06
    # dirs = [os.path.join(root, x) for x in os.listdir(root)]

    # 01. dirs: a list containing the names of the entries in the directory
    dirs = []
    for name in os.listdir(root):
        fdir = os.path.join(root, name)
        if os.path.isdir(fdir) and not name.startswith('.'):
            dirs.append(fdir)

    dirs = natsort.natsorted(dirs)  # sort a list
    # print('dirs = ', dirs)
    # print('len(dirs) =', len(dirs))
    '''
        dirs is a list of directory in training set
        dirs = ['/home/dataset/ped2/training', 
                '/home/dataset/ped2/training/01',..,
                '/home/dataset/ped2/training/16']
    '''

    # 02. dataset is a list of files and directory in training set
    videos = []
    total_frames = 0
    lower_d = 2  # can duoi khoang cach d giua 2 frames trong KF
    upper_d = 5  # can tren khoang cach d giua 2 frames trong KF
    for fdir in dirs:  # fdir = video_path
        video = []
        pre_number = -2
        # os.listdir(): get the list of all files and directories in the directory.
        for name in natsort.natsorted(os.listdir(fdir)):
            # if it is a file with include_ext = [".png", ".jpg", "jpeg", ".bmp"]
            if os.path.isfile(os.path.join(fdir, name)) \
                    and not name.startswith('.') \
                    and any([name.endswith(ext) for ext in include_ext]):
                # name: is image file_name (000.jpg, 001.jpg, 0002.jpg)
                cur_number = extract_numeric_value_from_filename(name)
                if cur_number is not None:
                    # print(f"Numeric value extracted from the file_name: {cur_number}")
                    if pre_number + lower_d <= cur_number <= pre_number + upper_d:
                        video.append(os.path.join(fdir, name))
                    elif cur_number == pre_number + 1:  # neu 2 frames lien tiep
                        continue
                    elif cur_number > pre_number + upper_d:  # neu 2 frames cach nhau hon 5 frames
                        if len(video) >= 5:
                            videos.append(video)
                        video = [os.path.join(fdir, name)]
                    pre_number = cur_number

        if len(video) > 5:
            total_frames += len(video)
            videos.append(video)

    '''            
    datasets = [[os.path.join(fdir, name) for name in natsort.natsorted(os.listdir(fdir))
             if os.path.isfile(os.path.join(fdir, name))
             and not name.startswith('.')
             and any([name.endswith(ext) for ext in include_ext])]
            for fdir in dirs
        ]
    # because the 1st item in datasets is empty
    return [name for name in datasets if name] 
    '''
    # print('videos=', videos)
    return videos, total_frames


def get_clips_by_video(videos, num_frames=5, frame_steps=1):
    """
    Return a list of clips collected by video
    """
    clips_by_video = []
    for video_id in range(0, len(videos)):  # for each video
        frames = videos[video_id]  # list of path of all frames in video_id
        clips = []
        for frame_id in range(0, len(frames), frame_steps):  # for each frame
            if frame_id > len(frames) - num_frames:
                continue
            clip = []  # a clip is a list of frames
            for i in range(0, num_frames):
                clip.append(frames[frame_id + i])
            clips.append(clip)
        clips_by_video.append(clips)
    return clips_by_video


def get_clips(videos, num_frames=5, frame_steps=1):
    """
    Return clips (a list of clips) with frame_mode = 'successive'
    """
    clips = []
    for video_id in range(0, len(videos)):  # for each video
        frames = videos[video_id]  # list of path of all frames in video_id
        for frame_id in range(0, len(frames), frame_steps):  # for each frame
            if frame_id > len(frames) - num_frames:
                continue
            clip = []  # a clip is a list of frames
            for i in range(0, num_frames):
                clip.append(frames[frame_id + i])
            clips.append(clip)
    return clips


def get_clips_sf(videos, num_frames=5, skip_frames=None):
    T = num_frames
    if skip_frames is None:
        skip_frames = [2, 3, 4, 5]
    clips = []
    for video_id in range(0, len(videos)):  # for each video
        frames = videos[video_id]  # list of path of all frames in video_id
        L = len(frames)
        for n in range(L):
            retry = 0
            s = random.choice(skip_frames)

            # try to get a sequence of frames in a range of len(frames)
            while (n + (T - 1) * s) > L - 1 and retry < 10:
                s = random.choice(skip_frames)
                retry += 1
            if retry == 10:
                continue

            clip = []
            for t in range(0, T):
                j = min(n + t * s, L - 1)
                clip.append(frames[j])
            if len(clip) == T:
                clips.append(clip)
    return clips


def get_clips_rf(videos, num_frames=5, repeat_frames=None):
    T = num_frames
    if repeat_frames is None:
        repeat_frames = [2, 3]
    clips = []
    for video_id in range(0, len(videos)):  # for each video
        frames = videos[video_id]  # list of path of all frames in video_id
        L = len(frames)
        for n in range(L):
            retry = 0
            r = random.choice(repeat_frames)

            # try to get a sequence of frames in a range of len(frames)
            while (n + (T - 1) // r) > L - 1 and retry < 10:
                r = random.choice(repeat_frames)
                retry += 1
            if retry == 10:
                continue

            clip = []
            for t in range(0, T):
                j = min(n + t // r, L - 1)
                clip.append(frames[j])
            if len(clip) == T:
                clips.append(clip)
    return clips


# =============================================================
# =============================================================
def show_images(dataset_source, max_samples=20, num_cols=4):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(15, 15))
    for i, img in enumerate(dataset_source):
        if i == max_samples:
            break
        plt.subplot(int(max_samples / num_cols) + 1, num_cols, i + 1)
        plt.imshow(img[0])


def get_transform2(w=64, h=64):
    """
    Returns a transform that convert a PIL image to a tensor
    """
    transform = [
        transforms.Resize((w, h)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Scales data into [0,1]
        # transforms.Lambda(lambda t: (t * 2) - 1) # [0,1] to [-1, 1]
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))
    ]
    return transforms.Compose(transform)


def get_reverse_transform2():
    """
    Returns a transform that convert a tensor to a PIL image
    """
    transform = [
        transforms.Lambda(lambda t: (t + 1) / 2),  # tensor [-1, 1] to [0, 1]
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),  # tensor [0, 1] to tensor [0.0,255.0]
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),  # tensor[0.0,255.0] to numpy [0,255]
        transforms.ToPILImage(),  # numpy to PIL image object
    ]
    return transforms.Compose(transform)


def show_tensor_image(image):
    reverse_transform = get_reverse_transform2()
    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transform(image))


# =============================================================
# Code tham khao (MNAD)
# =============================================================
def get_videos_dict(dataset_dir_path):
    videos_dir_path = glob.glob(os.path.join(dataset_dir_path, '*'))
    videos_dict = OrderedDict()
    for video_dir_path in sorted(videos_dir_path):
        video_name = video_dir_path.split('/')[-1]
        videos_dict[video_name] = {}
        videos_dict[video_name]['path'] = video_dir_path
        videos_dict[video_name]['frames'] = glob.glob(os.path.join(video_dir_path, '*.jpg'))
        videos_dict[video_name]['frames'].sort()
        videos_dict[video_name]['length'] = len(videos_dict[video_name]['frames'])
    return videos_dict


def get_frames_list(dataset_dir_path, videos_dict, clip_step):
    # prediction: clip_step=4; reconstruction: clip_step=1
    if clip_step == 1:  # reconstruction
        clip_step = 0
    videos_dir_path = glob.glob(os.path.join(dataset_dir_path, '*'))
    frames_list = []
    for video_dir_path in sorted(videos_dir_path):
        video_name = video_dir_path.split('/')[-1]
        video_length = len(videos_dict[video_name]['frames'])
        for i in range(video_length - clip_step):  # i = 0,1,2,..,(len-step)-1
            frames_list.append(videos_dict[video_name]['frames'][i])

    return frames_list


def get_clip_tensor(batch_size, clip, data_clip):
    """
    param batch_size: batch size
    param clip: is a list of frames (PATHS)
    param data_clip: is a dictionary of a clip
    :return clip_tensor: is a list of frames (TENSORS)
    """
    clip_tensors = []
    for bz in range(batch_size):
        clip_tensor = []
        if data_clip['read_format'] == 'PIL':
            for i in range(len(clip)):
                raw_frame = Image.open(clip[i][bz]).convert('RGB')  # a PIL object image
                clip_tensor.append(data_clip['transform'](raw_frame))  # append a tensor (a transformed image)
        elif data_clip['read_format'] == 'CV2':
            for i in range(len(clip)):
                raw_frame = CV2_load_image(clip[i][bz], data_clip['channel'],
                                           data_clip['width'], data_clip['height'])
                clip_tensor.append(data_clip['transform'](raw_frame))  # append a tensor (a transformed image)

        if data_clip['clip_mode'] == 'cat':  # for models: MNAD
            clip_tensor = torch.cat(clip_tensor, dim=data_clip['clip_dim'])
        elif data_clip['clip_mode'] == 'stack':  # for models: STEAL
            clip_tensor = torch.stack(clip_tensor, dim=data_clip['clip_dim'])
        # clip_tensor = clip_tensor.unsqueeze(0)  # torch.Size([1, 15, 256, 256])

        clip_tensors.append(clip_tensor)
    clip_tensors = torch.stack(clip_tensors, dim=0)
    return clip_tensors


def get_sample_sf(dataloader_sf, iter_sf, data_clip):
    if dataloader_sf is not None:
        # is_get_clip_tensor = False
        try:
            # Samples the batch
            data_sf = next(iter_sf)
            # if is_get_clip_tensor:
            #     data_sf = get_clip_tensor(1, data_sf, data_clip)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            iter_sf = iter(dataloader_sf)
            data_sf = next(iter_sf)
            # if is_get_clip_tensor:
            #     data_sf = get_clip_tensor(1, data_sf, data_clip)
        return data_sf
    else:
        print("Warning: self.dataloader_sf is None")
        return None


def get_sample_kf(dataloader_kf, iter_kf, data_clip):
    if dataloader_kf is not None:
        # is_get_clip_tensor = False
        try:
            # Samples the batch
            data_kf = next(iter_kf)
            # if is_get_clip_tensor:
            #     data_kf = get_clip_tensor(1, data_kf, data_clip)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            iter_kf = iter(dataloader_kf)
            data_kf = next(iter_kf)
            # if is_get_clip_tensor:
            #     data_kf = get_clip_tensor(1, data_kf, data_clip)
        return data_kf
    else:
        print("Warning: self.dataloader_kf is None")
        return None


def get_sample_obj(dataloader_obj, iter_obj, data_clip):
    if dataloader_obj is not None:
        # is_get_clip_tensor = False
        try:
            # Samples the batch
            data_obj = next(iter_obj)
            # if is_get_clip_tensor:
            #     data_obj = get_clip_tensor(1, data_obj, data_clip)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            iter_obj = iter(dataloader_obj)
            data_obj = next(iter_obj)
            # if is_get_clip_tensor:
            #     data_obj = get_clip_tensor(1, data_obj, data_clip)
        return data_obj
    else:
        print("Warning: self.dataloader_kf is None")
        return None


def get_sample_patch(dataloader_patch, iter_patch):
    if dataloader_patch is not None:
        try:
            # Samples the batch
            data_patch, _ = next(iter_patch)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            iter_patch = iter(dataloader_patch)
            data_patch, _ = next(iter_patch)
        return data_patch, _
    else:
        print("Warning: self.dataloader_patch is None")
        return None

