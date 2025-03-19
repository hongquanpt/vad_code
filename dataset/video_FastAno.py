from .video import VideoDataset
import torch
from .data_types import dict_clip, dict_patch_ST
from .aug import get_clip_TMT, get_clip_SRT
from .aug import get_clip_gaussian_noise, get_clip_simplex_noise
from .aug import get_clip_patch_cifar
from .aug import get_lists_dict_with_prob

from .samples import get_dataset_cifar100
from PIL import Image

# https://docs.python.org/3/library/random.html#random.uniform
import random


class VideoDataset_FastAno(VideoDataset):
    """
    Input: Train dataset or Test dataset
    Return: self.clips là danh sách các clip,
            Trong đó, self.clips[i]: clip thứ i gồm các đường dẫn đến 5 frames liên tiếp
            sau khi Data Augmentation: TMT, SRT
    Paper: 2022- FastAno_Fast_Anomaly_Detection_via_Spatio-Temporal_Patch_Transformation
    """

    def __init__(self, data_clip: dict_clip, data_patch: dict_patch_ST, dataset_path: str):
        """
        param augmode='normal_only', 'random', 'probability'
        param list_augtype=['TMT', 'SRT']
        """
        super().__init__(data_clip, dataset_path)
        print('VideoDataset_FastAno')
        self.data_patch = data_patch
        # [0.5, 0.5]
        self.lists_dict = get_lists_dict_with_prob(population=self.data_patch['list_augtype'],
                                                   weights=self.data_patch['list_prob_weights'],
                                                   num_elements=len(self.clips))
        self.augtype = 'normal_only'

    def get_clip_aug(self, clip, augtype='TMT'):  # TMT or SRT
        # print(f'self.augtype = {self.augtype}\n')
        if augtype == 'TMT':
            clip_TMT = get_clip_TMT(clip,
                                    self.data_patch['patch_size_isRandom'],
                                    self.data_patch['patch_size_range'],
                                    self.data_patch['patch_size'],
                                    self.data_patch['width'],
                                    self.data_patch['height'],
                                    self.data_patch['h_cut_size']
                                    )

            std = random.uniform(self.data_patch['mean'], self.data_patch['std'])
            clip = get_clip_gaussian_noise(clip_TMT, self.data_patch['noise_isAdded'],
                                           self.data_patch['mean'], std)
        elif augtype == 'SRT':
            clip_SRT = get_clip_SRT(clip,
                                    self.data_patch['patch_size_isRandom'],
                                    self.data_patch['patch_size_range'],
                                    self.data_patch['patch_size'],
                                    self.data_patch['width'],
                                    self.data_patch['height'],
                                    self.data_patch['h_cut_size']
                                    )
            std = random.uniform(self.data_patch['mean'], self.data_patch['std'])
            clip = get_clip_gaussian_noise(clip_SRT,
                                           self.data_patch['noise_isAdded'],
                                           self.data_patch['mean'], std)
        return clip

    def __getitem__(self, index):
        """
        Get a clip which includes num_frames(=5) frames
        """
        # print(f'TrainVideoDataset.index = {index}')
        clip = []
        # 1. self.clips[index]: is a clip which includes num_frames(=5) frames
        for frame in self.clips[index]:  # for each frame in a clip
            raw_frame = Image.open(frame).convert('RGB')  # a PIL object image
            clip.append(self.data_clip['transform'](raw_frame))  # append a tensor (a transformed image)

        if self.data_patch['augmode'] == 'random':
            self.augtype = random.choice(self.data_patch['list_augtype'])
        elif self.data_patch['augmode'] == 'probability':
            for i in range(len(self.data_patch['list_augtype'])):
                augtype = self.data_patch['list_augtype'][i]
                if (self.lists_dict.get(augtype) is not None) and (index in self.lists_dict.get(augtype)):
                    self.augtype = augtype
                    break
        elif self.data_patch['augmode'] == 'normal_only':
            self.augtype = 'normal_only'

        clip = self.get_clip_aug(clip, self.augtype)

        # 3. clip_mode
        if self.data_clip['clip_mode'] == 'cat':  # for models: MNAD
            # clip[i].shape = tensor.size(3,256,256)
            # => cat => clip.shape = tensor.size(15,256,256)
            clip = torch.cat(clip, dim=0)
        elif self.data_clip['clip_mode'] == 'stack':  # for models: ConvLSTM_AE
            # clip[i].shape = tensor.size(3,256,256)
            # => stack => clip.shape = tensor.size(10,3,256,256)
            clip = torch.stack(clip, dim=0)
        elif self.data_clip['clip_mode'] == 'list':  # for models: ASTNet
            return clip
        return clip


# ===========================================================================
class VideoDataset_aug(VideoDataset_FastAno):
    """
    Input: Train dataset or Test dataset
    Return: self.clips là danh sách các clip,
            Trong đó, self.clips[i]: clip thứ i gồm các đường dẫn đến 5 frames liên tiếp
            sau khi Data Augmentation: TMT, SRT, patch_cifar, gaussian_noise, simplex_noise
    """

    def __init__(self, data_clip: dict_clip, data_patch: dict_patch_ST, dataset_path: str):
        """
        param list_augtype=['normal_only', 'TMT', 'SRT', 'gaussian_noise', 'simplex_noise']
        """
        super().__init__(data_clip, data_patch, dataset_path)
        print('VideoDataset_aug')

        # [0.0, 0.5, 0.5, 0.0, 0.0]
        self.lists_dict = get_lists_dict_with_prob(population=self.data_patch['list_augtype'],
                                                   weights=self.data_patch['list_prob_weights'],
                                                   num_elements=len(self.clips))
        '''
        For example:
        self.lists_dict = [
                            "normal_only": [0, 4, 6],
                            "TMT": [1, 5, 7],
                            "SRT": [2, 3, 8],...
                            ]
        '''
        # print for showing frame ids of each list_dict_augtype
        '''
        for i in range(len(self.list_augtype)):
            aug_type = self.list_augtype[i]
            list_dict_augtype = self.lists_dict.get(aug_type)
            print(f'list_dict_augtype[{aug_type}] = {list_dict_augtype}')
        '''

        # 001-0.01-2023-10-22-20-52:
        # no initialize_weights
        # list_prob_weights = [0.2, 0.2, 0.2, 0.2, 0.2] => AUC max = 92.29%

        self.cifar100_dataset, self.cifar100_dataloader = get_dataset_cifar100()
        self.cifar_iter = iter(self.cifar100_dataloader)
        self.aug_type = 'normal_only'

    # ===========================================================
    # ===========================================================
    def get_clip_aug(self, clip, aug_type='normal_only'):
        # print(f'self.aug_type = {self.aug_type}\n')
        if aug_type == 'TMT':
            return get_clip_TMT(clip,
                                self.data_patch['patch_size_isRandom'],
                                self.data_patch['patch_size_range'],
                                self.data_patch['patch_size'],
                                self.data_patch['width'],
                                self.data_patch['height'],
                                self.data_patch['h_cut_size']
                                )
        elif aug_type == 'SRT':
            return get_clip_SRT(clip,
                                self.data_patch['patch_size_isRandom'],
                                self.data_patch['patch_size_range'],
                                self.data_patch['patch_size'],
                                self.data_patch['width'],
                                self.data_patch['height'],
                                self.data_patch['h_cut_size']
                                )
        elif aug_type == 'Patch_Cifar100':
            try:
                # Samples the batch
                cifar_img, _ = next(self.cifar_iter)
            except StopIteration:
                # restart the generator if the previous generator is exhausted.
                self.cifar_iter = iter(self.cifar100_dataloader)
                cifar_img, _ = next(self.cifar_iter)
            return get_clip_patch_cifar(clip, cifar_img,
                                        self.data_patch['patch_size'],
                                        self.data_patch['width'],
                                        self.data_patch['height'],
                                        self.data_patch['h_cut_size']
                                        )
        elif aug_type == 'gaussian_noise':
            return get_clip_gaussian_noise(clip, self.data_patch['noise_isAdded'], mean=0, std=0.1)
        elif aug_type == 'simplex_noise':
            return get_clip_simplex_noise(clip,
                                          self.data_patch['noise_isAdded'],
                                          self.data_patch['patch_size'],
                                          self.data_patch['width'],
                                          self.data_patch['height'],
                                          self.data_patch['h_cut_size']
                                          )
        elif aug_type == 'normal_only':
            return clip
