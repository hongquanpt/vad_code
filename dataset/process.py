from .video import VideoDataset, VideoDataset_sf, VideoDataset_rf, VideoDataset_kf
from .video_FastAno import VideoDataset_FastAno, VideoDataset_aug
from .video_GMA_DAE import VideoDataset_GMM_DAE
from .util import get_transform
from .data_types import dict_clip, dict_patch_ST
from .data_types import dict_sf, dict_rf, dict_patch, dict_kf, dict_obj
from .data_types import dict_fusion, dict_noise, dict_shake
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
import os
import torch


def get_train_dataset(cfg):
    # fullpath of train_set
    root = os.path.join(os.getcwd(), 'dataset')  # '/home/dataset'
    dataset_path = os.path.join(root, cfg.DATASET.name, cfg.DATASET.train.train_set)
    assert (os.path.exists(dataset_path))

    data_clip = dict(dict_clip)
    data_clip['sample_type'] = cfg.DATASET.train.sample_type
    data_clip['num_frames'] = cfg.DATASET.num_frames
    data_clip['frame_steps'] = cfg.DATASET.frame_steps
    data_clip['clip_mode'] = cfg.DATASET.clip_mode
    data_clip['clip_dim'] = cfg.DATASET.clip_dim
    data_clip['read_format'] = cfg.DATASET.read_format
    data_clip['width'] = cfg.DATASET.width
    data_clip['height'] = cfg.DATASET.height
    data_clip['channel'] = cfg.DATASET.channel
    if data_clip['read_format'] == 'PIL':
        # is_normalize=True: convert (PIL image) to [-1,1]
        # is_normalize=False: convert (PIL image) to [0,1]
        data_clip['transform'] = get_transform(size=[cfg.DATASET.width, cfg.DATASET.height],
                                               channel=cfg.DATASET.channel,
                                               is_toTensor=True,
                                               is_normalized=cfg.DATASET.normalized)
    elif data_clip['read_format'] == "CV2":
        data_clip['transform'] = transforms.Compose([transforms.ToTensor()])

    dataset = VideoDataset(data_clip, dataset_path)  # a clip = 5 frames (256,256,3)
    num_gpus_available = torch.cuda.device_count()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.DATASET.train.batch_size_per_gpu,  # 4
        shuffle=cfg.DATASET.train.shuffle,  # cfg.DATASET.train.shuffle
        num_workers=cfg.DATASET.train.num_workers,  # cfg.DATASET.train.num_workers,  # 4 or 8; 4 * num_gpus_available,
        pin_memory=False,  # False (to reduce RAM consumption): gpustat (VRAM), htop
        drop_last=cfg.DATASET.train.drop_last
    )
    # Get the number of samples
    num_samples = len(dataset)
    print(f"Number of samples in TRAIN dataset: {num_samples}")

    return dataloader, data_clip


def get_train_dataset_DAE(cfg, img_type='Img'):
    """
    param cfg: is a config
    param img_type: 'Img' or 'Dimg'
    Note: batch_size_per_gpu_DAE
    """
    dataset_selector = {"Img": cfg.DATASET.train.train_set_Img,  # 'training_Crop_Img'
                        "Dimg": cfg.DATASET.train.train_set_Dimg}  # 'training_Crop_Dimg'
    root = os.path.join(os.getcwd(), 'dataset')  # '/home/dataset'
    dataset_path = os.path.join(root, cfg.DATASET.name, dataset_selector[img_type])
    assert (os.path.exists(dataset_path))

    data_clip = dict(dict_clip)
    data_clip['sample_type'] = cfg.DATASET.train.sample_type
    data_clip['num_frames'] = cfg.DATASET.num_frames
    data_clip['frame_steps'] = cfg.DATASET.frame_steps
    data_clip['clip_mode'] = cfg.DATASET.clip_mode
    data_clip['clip_dim'] = cfg.DATASET.clip_dim
    data_clip['transform'] = None
    data_clip['read_format'] = cfg.DATASET.read_format
    dataset = VideoDataset_GMM_DAE(data_clip, dataset_path)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.DATASET.train.batch_size_per_gpu_DAE,
        shuffle=cfg.DATASET.train.shuffle,
        num_workers=cfg.DATASET.train.num_workers,
        pin_memory=False,
        drop_last=cfg.DATASET.train.drop_last
    )
    # Get the number of samples
    num_samples = len(dataset)
    print(f"Number of samples in TRAIN dataset DAE: {num_samples}")
    return dataset, dataloader


def get_train_dataset_GMM(cfg, img_type='Img'):
    """
    param img_type: 'Img' or 'Dimg'
    Note: batch_size_per_gpu_GMM
    """
    dataset_selector = {"Img": cfg.DATASET.train.train_set_Img,  # 'training_Crop_Img'
                        "Dimg": cfg.DATASET.train.train_set_Dimg}  # 'training_Crop_Dimg'
    root = os.path.join(os.getcwd(), 'dataset')  # '/home/dataset'
    dataset_path = os.path.join(root, cfg.DATASET.name, dataset_selector[img_type])
    assert (os.path.exists(dataset_path))

    data_clip = dict(dict_clip)
    data_clip['sample_type'] = cfg.DATASET.train.sample_type
    data_clip['num_frames'] = cfg.DATASET.num_frames
    data_clip['frame_steps'] = cfg.DATASET.frame_steps
    data_clip['clip_mode'] = cfg.DATASET.clip_mode
    data_clip['clip_dim'] = cfg.DATASET.clip_dim
    data_clip['read_format'] = cfg.DATASET.read_format
    data_clip['transform'] = None
    dataset = VideoDataset_GMM_DAE(data_clip, dataset_path)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.DATASET.train.batch_size_per_gpu_GMM,
        shuffle=cfg.DATASET.train.shuffle,
        num_workers=cfg.DATASET.train.num_workers,
        pin_memory=False,
        drop_last=cfg.DATASET.train.drop_last
    )
    # Get the number of samples
    num_samples = len(dataset)
    print(f"Number of samples in TRAIN dataset GMM: {num_samples}")
    return dataset, dataloader


def get_train_dataset_aug(cfg):
    # fullpath of train_set
    root = os.path.join(os.getcwd(), 'dataset')  # '/home/dataset'
    dataset_path = os.path.join(root, cfg.DATASET.name, cfg.DATASET.train.train_set)
    assert (os.path.exists(dataset_path))

    data_clip = dict(dict_clip)
    data_clip['sample_type'] = cfg.DATASET.train.sample_type
    data_clip['num_frames'] = cfg.DATASET.num_frames
    data_clip['frame_steps'] = cfg.DATASET.frame_steps
    data_clip['clip_mode'] = cfg.DATASET.clip_mode
    data_clip['clip_dim'] = cfg.DATASET.clip_dim
    data_clip['read_format'] = cfg.DATASET.read_format
    data_clip['width'] = cfg.DATASET.width
    data_clip['height'] = cfg.DATASET.height
    data_clip['channel'] = cfg.DATASET.channel
    if data_clip['read_format'] == 'PIL':
        # is_normalize=True: convert (PIL image) to [-1,1]
        # is_normalize=False: convert (PIL image) to [0,1]
        data_clip['transform'] = get_transform(size=[cfg.DATASET.width, cfg.DATASET.height],
                                               channel=cfg.DATASET.channel,
                                               is_toTensor=True,
                                               is_normalized=cfg.DATASET.normalized)
    elif data_clip['read_format'] == "CV2":
        data_clip['transform'] = transforms.Compose([transforms.ToTensor()])

    data_patch = dict(dict_patch_ST)
    data_patch['augmode'] = cfg.DATASET.augmode
    data_patch['list_augtype'] = cfg.DATASET.list_augtype
    data_patch['list_prob_weights'] = cfg.DATASET.list_prob_weights
    data_patch['patch_size_isRandom'] = cfg.DATASET.patch_size_isRandom
    data_patch['patch_size_range'] = cfg.DATASET.patch_size_range
    data_patch['patch_size'] = cfg.DATASET.patch_size

    data_patch['width'] = cfg.DATASET.width
    data_patch['height'] = cfg.DATASET.height
    data_patch['H_cut_size'] = cfg.DATASET.H_cut_size
    data_patch['noise_isAdded'] = cfg.DATASET.noise_isAdded
    data_patch['mean'] = cfg.DATASET.mean
    data_patch['std'] = cfg.DATASET.std
    dataset = VideoDataset_aug(data_clip, data_patch, dataset_path)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.DATASET.train.batch_size_per_gpu,
        shuffle=cfg.DATASET.train.shuffle,
        num_workers=cfg.DATASET.train.num_workers,
        pin_memory=False,
        drop_last=cfg.DATASET.train.drop_last
    )
    # Get the number of samples
    num_samples = len(dataset)
    print(f"Number of samples in TRAIN dataset aug: {num_samples}")
    return dataset, dataloader


def get_data_sf(cfg):
    data_sf = dict(dict_sf)
    data_sf['p'] = cfg.DATASET.dict_sf.p
    data_sf['s'] = cfg.DATASET.dict_sf.s
    data_sf['dataloader'] = None

    # fullpath of train_set
    root = os.path.join(os.getcwd(), 'dataset')  # '/home/dataset'
    dataset_path = os.path.join(root, cfg.DATASET.name, cfg.DATASET.train.train_set)
    assert (os.path.exists(dataset_path))

    data_clip = dict(dict_clip)
    data_clip['sample_type'] = cfg.DATASET.train.sample_type
    data_clip['num_frames'] = cfg.DATASET.num_frames
    data_clip['frame_steps'] = cfg.DATASET.frame_steps
    data_clip['clip_mode'] = cfg.DATASET.clip_mode
    data_clip['clip_dim'] = cfg.DATASET.clip_dim
    data_clip['read_format'] = cfg.DATASET.read_format
    data_clip['width'] = cfg.DATASET.width
    data_clip['height'] = cfg.DATASET.height
    data_clip['channel'] = cfg.DATASET.channel
    if data_clip['read_format'] == 'PIL':
        # is_normalize=True: convert (PIL image) to [-1,1]
        # is_normalize=False: convert (PIL image) to [0,1]
        data_clip['transform'] = get_transform(size=[cfg.DATASET.width, cfg.DATASET.height],
                                               channel=cfg.DATASET.channel,
                                               is_toTensor=True,
                                               is_normalized=cfg.DATASET.normalized)
    elif data_clip['read_format'] == "CV2":
        data_clip['transform'] = transforms.Compose([transforms.ToTensor()])

    if cfg.DATASET.dict_sf.p > 0:
        dataset = VideoDataset_sf(data_clip, dataset_path, data_sf['s'])
        data_sf['dataloader'] = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=cfg.DATASET.train.shuffle,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )
        # Get the number of samples
        num_samples = len(dataset)
        print(f"Number of samples in TRAIN dataset SKIP_FRAMES: {num_samples}")
    return data_sf


def get_data_kf(cfg):
    data_kf = dict(dict_kf)
    data_kf['p'] = cfg.DATASET.dict_kf.p
    data_kf['dataloader'] = None

    # fullpath of train_set
    root = os.path.join(os.getcwd(), 'dataset')  # '/home/dataset'
    dataset_path = os.path.join(root, cfg.DATASET.name, cfg.DATASET.train.keyframes_dir)
    print('dataset_path =', dataset_path)
    if not os.path.exists(dataset_path):
        print(f'{dataset_path} is NOT exists !!!')
        return data_kf
    assert (os.path.exists(dataset_path))

    data_clip = dict(dict_clip)
    data_clip['sample_type'] = cfg.DATASET.train.sample_type
    data_clip['num_frames'] = cfg.DATASET.num_frames
    data_clip['frame_steps'] = cfg.DATASET.frame_steps
    data_clip['clip_mode'] = cfg.DATASET.clip_mode
    data_clip['clip_dim'] = cfg.DATASET.clip_dim
    data_clip['read_format'] = cfg.DATASET.read_format
    data_clip['width'] = cfg.DATASET.width
    data_clip['height'] = cfg.DATASET.height
    data_clip['channel'] = cfg.DATASET.channel
    if data_clip['read_format'] == 'PIL':
        # is_normalize=True: convert (PIL image) to [-1,1]
        # is_normalize=False: convert (PIL image) to [0,1]
        data_clip['transform'] = get_transform(size=[cfg.DATASET.width, cfg.DATASET.height],
                                               channel=cfg.DATASET.channel,
                                               is_toTensor=True,
                                               is_normalized=cfg.DATASET.normalized)
    elif data_clip['read_format'] == "CV2":
        data_clip['transform'] = transforms.Compose([transforms.ToTensor()])

    if cfg.DATASET.dict_kf.p > 0:
        dataset = VideoDataset_kf(data_clip, dataset_path)  # VideoDataset(data_clip, dataset_path)  #
        data_kf['dataloader'] = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=cfg.DATASET.train.shuffle,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )
        # Get the number of samples
        num_samples = len(dataset)
        print(f"Number of samples in TRAIN dataset KEY_FRAMES: {num_samples}")

    return data_kf


def get_data_obj(cfg):
    data_obj = dict(dict_obj)
    data_obj['p'] = cfg.DATASET.dict_obj.p
    data_obj['dataloader'] = None

    # fullpath of train_set
    root = os.path.join(os.getcwd(), 'dataset')  # '/home/dataset'
    dataset_path = os.path.join(root, cfg.DATASET.name, cfg.DATASET.train.obj_dir)
    print('dataset_path =', dataset_path)
    if not os.path.exists(dataset_path):
        print(f'{dataset_path} is NOT exists !!!')
        return data_obj
    assert (os.path.exists(dataset_path))

    data_clip = dict(dict_clip)
    data_clip['sample_type'] = cfg.DATASET.train.sample_type
    data_clip['num_frames'] = cfg.DATASET.num_frames
    data_clip['frame_steps'] = cfg.DATASET.frame_steps
    data_clip['clip_mode'] = cfg.DATASET.clip_mode
    data_clip['clip_dim'] = cfg.DATASET.clip_dim
    data_clip['read_format'] = cfg.DATASET.read_format
    data_clip['width'] = cfg.DATASET.width
    data_clip['height'] = cfg.DATASET.height
    data_clip['channel'] = cfg.DATASET.channel
    if data_clip['read_format'] == 'PIL':
        # is_normalize=True: convert (PIL image) to [-1,1]
        # is_normalize=False: convert (PIL image) to [0,1]
        data_clip['transform'] = get_transform(size=[cfg.DATASET.width, cfg.DATASET.height],
                                               channel=cfg.DATASET.channel,
                                               is_toTensor=True,
                                               is_normalized=cfg.DATASET.normalized)
    elif data_clip['read_format'] == "CV2":
        data_clip['transform'] = transforms.Compose([transforms.ToTensor()])

    if cfg.DATASET.dict_obj.p > 0:
        dataset = VideoDataset(data_clip, dataset_path)  # VideoDataset(data_clip, dataset_path)  #
        data_obj['dataloader'] = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=cfg.DATASET.train.shuffle,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )
        # Get the number of samples
        num_samples = len(dataset)
        print(f"Number of samples in TRAIN dataset OBJECT: {num_samples}")

    return data_obj


def get_data_rf(cfg):
    data_rf = dict(dict_rf)
    data_rf['p'] = cfg.DATASET.dict_rf.p
    data_rf['r'] = cfg.DATASET.dict_rf.r
    data_rf['dataloader'] = None

    # fullpath of train_set
    root = os.path.join(os.getcwd(), 'dataset')  # '/home/dataset'
    dataset_path = os.path.join(root, cfg.DATASET.name, cfg.DATASET.train.train_set)
    assert (os.path.exists(dataset_path))

    data_clip = dict(dict_clip)
    data_clip['sample_type'] = cfg.DATASET.train.sample_type
    data_clip['num_frames'] = cfg.DATASET.num_frames
    data_clip['frame_steps'] = cfg.DATASET.frame_steps
    data_clip['clip_mode'] = cfg.DATASET.clip_mode
    data_clip['clip_dim'] = cfg.DATASET.clip_dim
    data_clip['read_format'] = cfg.DATASET.read_format
    data_clip['width'] = cfg.DATASET.width
    data_clip['height'] = cfg.DATASET.height
    data_clip['channel'] = cfg.DATASET.channel
    if data_clip['read_format'] == 'PIL':
        # is_normalize=True: convert (PIL image) to [-1,1]
        # is_normalize=False: convert (PIL image) to [0,1]
        data_clip['transform'] = get_transform(size=[cfg.DATASET.width, cfg.DATASET.height],
                                               channel=cfg.DATASET.channel,
                                               is_toTensor=True,
                                               is_normalized=cfg.DATASET.normalized)
    elif data_clip['read_format'] == "CV2":
        data_clip['transform'] = transforms.Compose([transforms.ToTensor()])

    if cfg.DATASET.dict_rf.p > 0:
        dataset = VideoDataset_rf(data_clip, dataset_path, data_rf['r'])
        data_rf['dataloader'] = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=cfg.DATASET.train.shuffle,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )
        # Get the number of samples
        num_samples = len(dataset)
        print(f"Number of samples in TRAIN dataset REPEAT: {num_samples}")

    return data_rf


def get_data_patch(cfg):
    data_patch = dict(dict_patch)
    data_patch['p'] = cfg.DATASET.dict_patch.p
    data_patch['alpha'] = cfg.DATASET.dict_patch.alpha
    data_patch['beta'] = cfg.DATASET.dict_patch.beta
    data_patch['technique'] = cfg.DATASET.dict_patch.technique
    data_patch['intruder'] = cfg.DATASET.dict_patch.intruder
    data_patch['dataloader'] = None

    if cfg.DATASET.dict_patch.p > 0 and data_patch['intruder'] == 'CIFAR-100':
        path = os.path.join(os.getcwd(), 'dataset', 'CIFAR100')
        dataset = CIFAR100(path,
                           train=True,
                           download=True,
                           transform=transforms.ToTensor())
        cifar100_dataloader = torch.utils.data.DataLoader(dataset,
                                                          batch_size=1,
                                                          shuffle=cfg.DATASET.train.shuffle,
                                                          num_workers=0,
                                                          pin_memory=False,
                                                          drop_last=True)

        data_patch['dataloader'] = cifar100_dataloader
        # Get the number of samples
        num_samples = len(dataset)
        print(f"Number of samples in TRAIN dataset CIFAR100: {num_samples}")

    return data_patch


def get_data_fusion(cfg):
    data_fusion = dict(dict_fusion)
    data_fusion['p'] = cfg.DATASET.dict_fusion.p
    return data_fusion


def get_data_noise(cfg):
    data_noise = dict(dict_noise)
    data_noise['p'] = cfg.DATASET.dict_noise.p
    data_noise['sigma'] = cfg.DATASET.dict_noise.sigma
    return data_noise


def get_data_shake(cfg):
    data_shake = dict(dict_shake)
    data_shake['p'] = cfg.DATASET.dict_shake.p
    data_shake['rot'] = cfg.DATASET.dict_shake.rot
    data_shake['trans'] = cfg.DATASET.dict_shake.trans
    return data_shake


def get_train_dataset_FastAno(cfg):
    # fullpath of train_set
    root = os.path.join(os.getcwd(), 'dataset')  # '/home/dataset'
    dataset_path = os.path.join(root, cfg.DATASET.name, cfg.DATASET.train.train_set)
    assert (os.path.exists(dataset_path))

    data_clip = dict(dict_clip)
    data_clip['sample_type'] = cfg.DATASET.train.sample_type
    data_clip['num_frames'] = cfg.DATASET.num_frames
    data_clip['frame_steps'] = cfg.DATASET.frame_steps
    data_clip['clip_mode'] = cfg.DATASET.clip_mode
    data_clip['clip_dim'] = cfg.DATASET.clip_dim
    data_clip['read_format'] = cfg.DATASET.read_format
    data_clip['width'] = cfg.DATASET.width
    data_clip['height'] = cfg.DATASET.height
    data_clip['channel'] = cfg.DATASET.channel
    if data_clip['read_format'] == 'PIL':
        # is_normalize=True: convert (PIL image) to [-1,1]
        # is_normalize=False: convert (PIL image) to [0,1]
        data_clip['transform'] = get_transform(size=[cfg.DATASET.width, cfg.DATASET.height],
                                               channel=cfg.DATASET.channel,
                                               is_toTensor=True,
                                               is_normalized=cfg.DATASET.normalized)
    elif data_clip['read_format'] == "CV2":
        data_clip['transform'] = transforms.Compose([transforms.ToTensor()])

    data_patch = dict(dict_patch_ST)
    data_patch['augmode'] = cfg.DATASET.augmode
    data_patch['list_augtype'] = cfg.DATASET.list_augtype
    data_patch['list_prob_weights'] = cfg.DATASET.list_prob_weights
    data_patch['patch_size_isRandom'] = cfg.DATASET.patch_size_isRandom
    data_patch['patch_size_range'] = cfg.DATASET.patch_size_range
    data_patch['patch_size'] = cfg.DATASET.patch_size

    data_patch['width'] = cfg.DATASET.width
    data_patch['height'] = cfg.DATASET.height
    data_patch['H_cut_size'] = cfg.DATASET.H_cut_size
    data_patch['noise_isAdded'] = cfg.DATASET.noise_isAdded
    data_patch['mean'] = cfg.DATASET.mean
    data_patch['std'] = cfg.DATASET.std
    dataset = VideoDataset_FastAno(data_clip, data_patch, dataset_path)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.DATASET.train.batch_size_per_gpu,
        shuffle=cfg.DATASET.train.shuffle,
        num_workers=cfg.DATASET.train.num_workers,
        pin_memory=False,
        drop_last=cfg.DATASET.train.drop_last
    )
    # Get the number of samples
    num_samples = len(dataset)
    print(f"Number of samples in TRAIN dataset FastAno: {num_samples}")
    return dataloader


# ============================================================
# ============================================================
def get_test_dataset(cfg):
    """
    param: sample_type = 'video' (MNAD, MNAD, ASTNet)
            sample_type = clips_by_video' (FastAno, STEAL, PAMAE4)
    """
    # fullpath of test_set
    root = os.path.join(os.getcwd(), 'dataset')  # '/home/dataset'
    dataset_path = os.path.join(root, cfg.DATASET.name, cfg.DATASET.test.test_set)
    print('dataset_path = ', dataset_path)
    assert (os.path.exists(dataset_path))

    data_clip = dict(dict_clip)
    data_clip['sample_type'] = cfg.DATASET.test.sample_type
    data_clip['num_frames'] = cfg.DATASET.num_frames
    data_clip['frame_steps'] = cfg.DATASET.frame_steps
    data_clip['clip_mode'] = cfg.DATASET.clip_mode
    data_clip['clip_dim'] = cfg.DATASET.clip_dim
    data_clip['read_format'] = cfg.DATASET.read_format
    data_clip['width'] = cfg.DATASET.width
    data_clip['height'] = cfg.DATASET.height
    data_clip['channel'] = cfg.DATASET.channel
    if data_clip['read_format'] == 'PIL':
        # is_normalize=True: convert (PIL image) to [-1,1]
        # is_normalize=False: convert (PIL image) to [0,1]
        data_clip['transform'] = get_transform(size=[cfg.DATASET.width, cfg.DATASET.height],
                                               channel=cfg.DATASET.channel,
                                               is_toTensor=True,
                                               is_normalized=cfg.DATASET.normalized)
    elif data_clip['read_format'] == "CV2":
        data_clip['transform'] = transforms.Compose([transforms.ToTensor()])
    dataset = VideoDataset(data_clip, dataset_path)
    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )
    # Get the number of samples
    num_samples = len(dataset)
    print(f"Number of samples in TEST dataset: {num_samples}")
    return test_dataloader, data_clip


# ============================================================
# ============================================================
def get_dataset_to_generate_OpticalFlow(cfg, dataset_path):
    data_clip = dict(dict_clip)
    data_clip['sample_type'] = cfg.DATASET.train.sample_type
    data_clip['num_frames'] = cfg.DATASET.num_frames
    data_clip['frame_steps'] = cfg.DATASET.frame_steps
    data_clip['clip_mode'] = cfg.DATASET.clip_mode
    data_clip['clip_dim'] = cfg.DATASET.clip_dim
    data_clip['read_format'] = cfg.DATASET.read_format
    data_clip['width'] = cfg.DATASET.width
    data_clip['height'] = cfg.DATASET.height
    data_clip['channel'] = cfg.DATASET.channel
    if data_clip['read_format'] == 'PIL':
        # is_normalize=True: convert (PIL image) to [-1,1]
        # is_normalize=False: convert (PIL image) to [0,1]
        data_clip['transform'] = get_transform(size=[cfg.DATASET.width, cfg.DATASET.height],
                                               channel=cfg.DATASET.channel,
                                               is_toTensor=True,
                                               is_normalized=cfg.DATASET.normalized)
    elif data_clip['read_format'] == "CV2":
        data_clip['transform'] = transforms.Compose([transforms.ToTensor()])
    dataset = VideoDataset(data_clip, dataset_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    return dataset, dataloader
