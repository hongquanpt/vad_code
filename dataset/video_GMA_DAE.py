from .video import VideoDataset
from PIL import Image
import torchvision.transforms as transforms


class VideoDataset_GMM_DAE(VideoDataset):
    """
    Input: Img dataset or Dimg dataset
    Return: self.clips là danh sách các clip,
            Trong đó, self.clips[i]: clip thứ i gồm đường dẫn đến 1 frame
    Paper: 2020- GMM_DAE
    """

    def __init__(self, data_clip, dataset_path):
        """
        param dataset_path: fullpath to train_set or test_set
        param transform: a transform applied on a frame
        """
        super().__init__(data_clip, dataset_path)
        print('VideoDataset_GAMM_DAE')

    def __getitem__(self, index):
        clip = []
        # 1. self.clips[index]: is a clip which includes num_frames(=5) frames
        for img in self.clips[index]:  # for each frame in a clip
            img = Image.open(img).convert('L')  # a PIL object image
            temp = int((img.size[1] - img.size[0]) / 2)
            arround = 0
            pad = (temp + arround, 0 + arround) if temp >= 0 else (0 + arround, arround - temp)
            w = 64
            # Image.ANTIALIAS was removed in Pillow 10.0.0
            # Now you need to use: PIL.Image.LANCZOS or PIL.Image.Resampling.LANCZOS.
            self.transform = transforms.Compose([transforms.Pad(pad, fill=0),
                                                 transforms.Resize((w, w), Image.LANCZOS),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=0.5, std=0.5)])
            clip.append(self.transform(img))  # append a tensor (a transformed image)
        return clip
