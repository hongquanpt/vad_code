import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from .image_reader import load_image
from .util import get_videos, get_videos_kf
from .util import get_clips, get_clips_sf, get_clips_rf
from .data_types import dict_clip


# PIL Image
# https://github.com/vt-le/astnet
class Clips(data.Dataset):
    """
    Input: Train dataset or Test dataset
    Return: self.clips là danh sách các clip,
            Trong đó, self.clips[i]: clip thứ i gồm các đường dẫn đến 5 frames liên tiếp.
    """

    def __init__(self, data_clip: dict_clip, dataset_path: str):
        """
        param data_clip: dict_clip
        param dataset_path: fullpath to train_set or test_set
        """
        super(Clips, self).__init__()
        print("call: super(Clips, self).__init__()")
        self.data_clip = data_clip
        # ============================================
        # self.videos là list các videos
        # self.videos[i] là list các đường dẫn đến các frames của video thứ i.
        self.videos, self.frame_count = get_videos(dataset_path)
        print('len(self.videos) = ', len(self.videos))
        # ============================================
        # self.clips là list các clips (được gộp lại từ các videos)
        # self.clips[i]: clip thứ i gồm các đường dẫn đến 5 frames liên tiếp.
        self.clips = get_clips(self.videos, self.data_clip['num_frames'], self.data_clip['frame_steps'])
        print('len(self.clips) = ', len(self.clips))
        print("self.data_clip['read_format'] = ", self.data_clip['read_format'])

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index: int):
        """
        Return a clip: (TENSOR)
        - a tensor concatenated by num_frames(=5) frames
        - or a tensor stacked by num_frames(=5) frames
        - or a list of num_frames(=5) frames (tensors)
        """
        frames = self.clips[index]  # self.clips[index] is a clip which includes num_frames(=5) frames
        clip = []
        if isinstance(self.data_clip['transform'], transforms.Compose):
            # print("The transform is a torchvision.transforms.Compose object.")
            for frame in frames:
                raw_frame = load_image(path=frame, channel=self.data_clip['channel'],
                                       resize_width=self.data_clip['width'], resize_height=self.data_clip['height'],
                                       format=self.data_clip['read_format'])
                clip.append(self.data_clip['transform'](raw_frame))
                del raw_frame  # Release memory after using raw_frame
            if self.data_clip['clip_mode'] == 'cat':  # for models: MNAD
                # Concatenates the given sequence of seq tensors in the given dimension.
                # return an array includes 15 elements (tensors) with size of 256x256
                # clip[i].shape = tensor.size(3,256,256) => cat => clip.shape = tensor.size(15,256,256)
                clip = torch.cat(clip, dim=self.data_clip['clip_dim'])
                # print('clip.shape =', clip.shape) #  (15,256,256)
            elif self.data_clip['clip_mode'] == 'stack':  # for models: ConvLSTM_AE
                # Concatenates a sequence of tensors along a new dimension.
                # return a tensor of num_frames(=10) tensors (transformed images)
                # clip[i].shape = tensor.size(3,256,256) => stack => clip.shape = tensor.size(10,3,256,256)
                clip = torch.stack(clip, dim=self.data_clip['clip_dim'])
                # print('clip.shape =', clip.shape) # (10,3,256,256)
        else:
            print("The transform is NOT a torchvision.transforms.Compose object.")
            for frame in frames:
                raw_frame = load_image(path=frame, channel=self.data_clip['channel'],
                                       resize_width=self.data_clip['width'], resize_height=self.data_clip['height'],
                                       format="CV2")
                clip.append(raw_frame)
                del raw_frame  # Release memory after using raw_frame
        return clip


class Clips_sf(Clips):
    """
    VideoDataset_sf: Clips with skip frames
    Input: Train dataset or Test dataset
    Return: self.clips là danh sách các clip, 
            Trong đó, self.clips[i]: clip thứ i gồm các đường dẫn đến 5 frames.
    """

    def __init__(self, data_clip: dict_clip, dataset_path: str, skip_frames: list):
        super().__init__(data_clip, dataset_path)

        # Replace self.clips with clips obtained from get_clips_sf()
        if skip_frames is None:
            skip_frames = [2, 3, 4, 5]
        self.clips = get_clips_sf(self.videos, self.data_clip['num_frames'], skip_frames)
        print('len(self.clips) = ', len(self.clips))
        print("self.data_clip['read_format'] = ", self.data_clip['read_format'])


# ===============================================================
class Clips_rf(Clips):
    """
    VideoDataset_rf: Clips with repeat frames
    Input: Train dataset or Test dataset
    Return: self.clips là danh sách các clip,
            Trong đó, self.clips[i]: clip thứ i gồm các đường dẫn đến 5 frames.
    """

    def __init__(self, data_clip: dict_clip, dataset_path: str, repeat_frames: list):
        super().__init__(data_clip, dataset_path)

        # Replace self.clips with clips obtained from get_clips_rf()
        if repeat_frames is None:
            repeat_frames = [2, 3]
        self.clips = get_clips_rf(self.videos, self.data_clip['num_frames'], repeat_frames)
        print('len(self.clips) = ', len(self.clips))
        print("self.data_clip['read_format'] = ", self.data_clip['read_format'])


# ===========================================================================
class Clips_kf(Clips):
    """
    VideoDataset_rf: Clips with repeat frames
    Input: Train dataset or Test dataset
    Return: self.clips là danh sách các clip,
            Trong đó, self.clips[i]: clip thứ i gồm các đường dẫn đến 5 frames.
    """

    def __init__(self, data_clip: dict_clip, dataset_path: str):
        # Call the __init__ method of the parent class
        super().__init__(data_clip, dataset_path)

        # Replace the videos with those from get_video_kf()
        # get_videos_kf() remove one of two successive frames in dataset_kf if they exist
        self.videos, _ = get_videos_kf(dataset_path)
        print('len(self.videos) = ', len(self.videos))
        # ============================================
        # Update the clips using the new videos
        self.clips = get_clips(self.videos, self.data_clip['num_frames'], self.data_clip['frame_steps'])
        print('len(self.clips) = ', len(self.clips))
