from .utils import np_load_frame

import torch.utils.data as data
import torchvision.transforms as transforms

import os
import glob
import numpy as np
import random  # anhle: for CustomDatset_SkipFrames
from collections import OrderedDict


class CustomDataset(data.Dataset):
    def __init__(self, dataset_dir_path: str,
                 transform: transforms.Compose,
                 resize_height: int, resize_width: int,
                 clip_step: int, num_pred: int):

        # dataset_dir_path = training/testing dir_path
        self.dataset_dir_path = dataset_dir_path
        self.transform = transform

        self.resize_height = resize_height
        self.resize_width = resize_width

        self.clip_step = clip_step
        self.num_pred = num_pred

        self.videos_dict = self.get_videos_dict(dataset_dir_path)
        self.frames_list = self.get_frames_list(dataset_dir_path, self.videos_dict)

    def get_videos_dict(self, dataset_dir_path):
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

    def get_frames_list(self, dataset_dir_path, videos_dict):
        frames_list = []
        videos_dir_path = glob.glob(os.path.join(dataset_dir_path, '*'))
        for video_dir_path in sorted(videos_dir_path):
            video_name = video_dir_path.split('/')[-1]
            for i in range(len(videos_dict[video_name]['frames'])):
                frames_list.append(videos_dict[video_name]['frames'][i])

        return frames_list

    def __getitem__(self, index):
        video_name = self.frames_list[index].split('/')[-2]
        frame_name = int(self.frames_list[index].split('/')[-1].split('.')[-2])

        clip = []
        for i in range(self.clip_step + self.num_pred):
            # min(a, b) để tránh tràn ra ngoài danh sách
            image = np_load_frame(self.videos_dict[video_name]['frames'][
                                      min(frame_name + i, len(self.videos_dict[video_name]['frames']) - 1)],
                                  self.resize_height,
                                  self.resize_width)
            if self.transform is not None:
                clip.append(self.transform(image))

        return np.concatenate(clip, axis=0)

    def __len__(self):
        return len(self.frames_list)


class CustomDataset_SkipFrames(CustomDataset):
    def __init__(self, dataset_dir_path: str,
                 transform: transforms.Compose,
                 resize_height: int, resize_width: int,
                 clip_step: int, num_pred: int, skip_frames=None):

        # dataset_dir_path = training/testing dir_path
        if skip_frames is None:
            skip_frames = [2]
        self.dataset_dir_path = dataset_dir_path
        self.transform = transform

        self.resize_height = resize_height
        self.resize_width = resize_width

        self.clip_step = clip_step
        self.num_pred = num_pred

        self.skip_frames = skip_frames

        self.videos_dict = self.get_videos_dict(dataset_dir_path)
        self.frames_list = self.get_frames_list(dataset_dir_path, self.videos_dict)

    def __getitem__(self, index):
        video_name = self.frames_list[index].split('/')[-2]
        frame_name = int(self.frames_list[index].split('/')[-1].split('.')[-2])

        clips_list = []
        skip_frames = random.choice(self.skip_frames)
        # print('skip_frames = ', skip_frames)

        retry = 0
        while len(self.videos_dict[video_name]['frames']) < frame_name + (
                self.clip_step - 1) * skip_frames and retry < 10:
            # reselect the frame_name
            frame_name = np.random.randint(len(self.videos_dict[video_name]['frames']))
            retry += 1
            # print('retry = ', retry)

        for i in range(self.clip_step + self.num_pred):
            print('\n id_skip_frames = ', frame_name + i * skip_frames)
            image = np_load_frame(self.videos_dict[video_name]['frames'][min(frame_name + i * skip_frames, len(
                self.videos_dict[video_name]['frames']) - 1)],
                                  self.resize_height,
                                  self.resize_width)
            if self.transform is not None:
                clips_list.append(self.transform(image))

        return np.concatenate(clips_list, axis=0)
