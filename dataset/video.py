from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from .image_reader import load_image
from .util import get_videos, get_videos_kf
from .util import get_clips, get_clips_by_video, get_clips_sf, get_clips_rf
from .data_types import dict_clip


# PIL Image
# https://github.com/vt-le/astnet
class VideoDataset(data.Dataset):
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
        super(VideoDataset, self).__init__()
        self.data_clip = data_clip
        # ============================================
        # self.videos is a list of videos
        # self.videos[i] is a list of paths to frames of the i-th video
        self.videos, self.frame_count = get_videos(dataset_path)
        print('len(self.videos) = ', len(self.videos))
        # ============================================
        # self.clips is a list of clips (concatenated from multiple videos).
        # self.clips[i]: the i-th clip consists of paths to 5 consecutive frames.
        self.clips = get_clips(self.videos, self.data_clip['num_frames'], self.data_clip['frame_steps'])
        print('len(self.clips) = ', len(self.clips))
        # ============================================
        # self.clips_by_video is a list of clips (grouped by individual videos).
        # self.clips_by_video[i] is a list of clips of the i-th video.
        # self.clips_by_video[i][j] is the j-th clip of the i-th video.
        self.clips_by_video = get_clips_by_video(self.videos, data_clip['num_frames'], data_clip['frame_steps'])
        print('len(self.clips_by_video) = ', len(self.clips_by_video))
        print("self.data_clip['read_format'] = ", self.data_clip['read_format'])

    def __len__(self):
        if self.data_clip['sample_type'] == 'videos':
            return len(self.videos)
        if self.data_clip['sample_type'] == 'clips':
            return len(self.clips)
        elif self.data_clip['sample_type'] == 'clips_by_video':
            return len(self.clips_by_video)

    def __getitem__(self, index: int):
        if self.data_clip['sample_type'] == 'videos':
            return self.get_sample_from_videos(index)
        elif self.data_clip['sample_type'] == 'clips':
            return self.get_sample_from_clips(index)
        elif self.data_clip['sample_type'] == 'clips_by_video':
            return self.get_sample_from_clips_by_video(index)

    def get_sample_from_videos(self, index: int):
        """
        Return a video: is a list of frames in a video (TENSOR)
        """
        frames = self.videos[index]  # self.videos[index] is a list of frames in a video
        video = []
        if isinstance(self.data_clip['transform'], transforms.Compose):
            # print("The transform is a torchvision.transforms.Compose object.")
            for frame in frames:
                raw_frame = load_image(path=frame, channel=self.data_clip['channel'],
                                       resize_width=self.data_clip['width'], resize_height=self.data_clip['height'],
                                       format=self.data_clip['read_format'])
                video.append(self.data_clip['transform'](raw_frame))
        else:
            print("The transform is NOT a torchvision.transforms.Compose object.")
            for frame in frames:
                raw_frame = load_image(path=frame, channel=self.data_clip['channel'],
                                       resize_width=self.data_clip['width'], resize_height=self.data_clip['height'],
                                       format="CV2")
                video.append(raw_frame)

        return video

    def get_sample_from_clips(self, index: int):
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

    def get_sample_from_clips_by_video(self, index: int):
        """
        Return a video: is a list of clips (TENSOR)
        => may cause "Not enough memory" error in testing
        RuntimeError: [enforce fail at C:\cb\pytorch_1000000000000\work\c10\core\impl\alloc_cpu.cpp:72]
        data. DefaultCPUAllocator: not enough memory: you tried to allocate 3932160 bytes.
        at command line: clip = torch.cat(clip, dim=self.data_clip['clip_dim'])
        """

        video = []
        for i in range(len(self.clips_by_video[index])):  # for each clip in a video index
            clip = []
            frames = self.clips_by_video[index][i]
            if isinstance(self.data_clip['transform'], transforms.Compose):
                # print("The transform is a torchvision.transforms.Compose object.")
                for frame in frames:  # for each frame in a clip
                    raw_frame = load_image(path=frame, channel=self.data_clip['channel'],
                                           resize_width=self.data_clip['width'], resize_height=self.data_clip['height'],
                                           format=self.data_clip['read_format'])
                    clip.append(self.data_clip['transform'](raw_frame))
                    del raw_frame  # Release memory after using raw_frame
                if self.data_clip['clip_mode'] == 'cat':  # for models: MNAD
                    clip = torch.cat(clip, dim=self.data_clip['clip_dim'])
                elif self.data_clip['clip_mode'] == 'stack':  # for models: ConvLSTM_AE
                    clip = torch.stack(clip, dim=self.data_clip['clip_dim'])
            else:
                print("The transform is NOT a torchvision.transforms.Compose object.")
                for frame in frames:  # for each frame in a clip
                    raw_frame = load_image(path=frame, channel=self.data_clip['channel'],
                                           resize_width=self.data_clip['width'], resize_height=self.data_clip['height'],
                                           format="CV2")
                    clip.append(raw_frame)
            video.append(clip)
        return video
    # ===============================================================


class VideoDataset_sf(VideoDataset):
    """
    VideoDataset_sf: VideoDataset with skip frames
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
class VideoDataset_rf(VideoDataset):
    """
    VideoDataset_rf: VideoDataset with repeat frames
    Input: Train dataset or Test dataset
    Return: self.clips là danh sách các clip,
            Trong đó, self.clips[i]: clip thứ i gồm các đường dẫn đến 5 frames.
    """

    def __init__(self, data_clip: dict_clip, dataset_path: str, repeat_frames: list):
        super().__init__(data_clip, dataset_path)
        if repeat_frames is None:
            repeat_frames = [2, 3]
        self.clips = get_clips_rf(self.videos, self.data_clip['num_frames'], repeat_frames)
        print('len(self.clips) = ', len(self.clips))
        print("self.data_clip['read_format'] = ", self.data_clip['read_format'])


# ===========================================================================
class VideoDataset_kf(VideoDataset):
    """
    VideoDataset_rf: VideoDataset with repeat frames
    Input: Train dataset or Test dataset
    Return: self.clips là danh sách các clip,
            Trong đó, self.clips[i]: clip thứ i gồm các đường dẫn đến 5 frames.
    """

    def __init__(self, data_clip: dict_clip, dataset_path: str):
        super().__init__(data_clip, dataset_path)

        # Replace the videos with those from get_video_kf()
        # get_videos_kf() remove one of two successive frames in dataset_kf if they exist
        self.videos, _ = get_videos_kf(dataset_path)
        print('len(self.videos) = ', len(self.videos))
        # ============================================
        # Update the clips using the new videos
        self.clips = get_clips(self.videos, self.data_clip['num_frames'], self.data_clip['frame_steps'])
        print('len(self.clips) = ', len(self.clips))
