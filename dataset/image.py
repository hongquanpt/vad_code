import os
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as T
import numpy as np
import random

from .util import get_transform, get_filelist_by_video


class TrainImageDataset(data.Dataset):
    """
    Returns a image TrainSet (stanfordCars)
    """

    def __init__(self, cfg):
        super(TrainImageDataset, self).__init__()

        self.new_size = [cfg.DATASET.width, cfg.DATASET.height]
        num_frames = cfg.DATASET.num_frames  # 5

        root = os.path.join(os.getcwd(), 'dataset')
        dataset_name = cfg.DATASET.name
        train_set = cfg.DATASET.train.train_set
        lower_bound = cfg.DATASET.lower_bound

        # self.dir = 'dataset/ped2/training/'
        self.dir = os.path.join(root, dataset_name, train_set)
        assert (os.path.exists(self.dir))

        # =================================
        # Danh sách các đường dẫn đến ảnh được nhóm theo từng video
        # =================================
        # =================================
        # train_dataloader:  Danh sách các clips 
        # (mỗi clip gồm các đường dẫn đến 5 ảnh liên tiếp)
        # =================================
        videos = get_filelist_by_video(self.dir)
        self.clips = []
        for video_id in range(0, len(videos)):
            video = videos[video_id]
            for frame_id in range(len(video)):
                if frame_id > len(video) - num_frames: continue
                clip = []
                for i in range(0, num_frames):
                    j = frame_id + i
                    clip.append(video[j])

                self.clips.append(clip)

        # print('self.clips = ', self.clips)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        # self.clips[index]: is a clip which includes num_frames(=5) frames
        # raw_frames: a list of num_frames(=5) RBG converted images
        frames = self.clips[index]
        raw_frames = [Image.open(f).convert('RGB') for f in frames]

        # clip: a list of num_frames(=5) tensors (transformed images)
        clip = []
        for f in raw_frames:
            transform = get_transform(self.new_size)
            f = transform(f)
            clip.append(f)
            # print('f.shape = ', f.shape) # f.shape =  torch.Size([3, 256, 256])

        # Cách 1: dùng np.concatenate(): Join a sequence of arrays along an existing axis.
        stack_clip = np.concatenate(clip, axis=0)  # return a numpy array
        # print('stack_clip.shape =', stack_clip.shape) #  (15, 256, 256)
        return stack_clip  # return an array includes 15 elements (tensors) có kích thước 256x256

        # Cách 2: 
        # return clip # a list of num_frames(=5) tensors (transformed images)

    # ===========================================================================


class TestImageDataset(data.Dataset):
    """
    Returns a image test_set (stanfordCars)
    """

    def __init__(self, cfg):
        super(TestImageDataset, self).__init__()
        self.new_size = [cfg.DATASET.width, cfg.DATASET.height]
        root = os.path.join(os.getcwd(), 'dataset')
        dataset_name = cfg.DATASET.name
        test_set = cfg.DATASET.test.test_set
        self.dir = os.path.join(root, dataset_name, test_set)
        assert (os.path.exists(self.dir))

        # =================================
        # Danh sách các video, mỗi video gồm nhiều ảnh
        # =================================
        self.videos = get_filelist_by_video(self.dir)

    def __len__(self):
        return len(self.videos)

    # Get a video which include many frames
    def __getitem__(self, index):
        frames = self.videos[index]

        video = []
        transform = get_transform(self.new_size)
        for frame in frames:
            raw_frame = Image.open(frame).convert('RGB')
            raw_frame = transform(raw_frame)
            video.append(raw_frame)

        return video  # return a list includes frames (tensors) in one video
