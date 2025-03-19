import torchvision.transforms as transforms

# Define the default values for the data structure
dict_clip = {
    'sample_type': 'videos',  # 'videos', 'clips', 'clips_by_video'
    'num_frames': 5,
    'frame_steps': 1,
    'clip_mode': 'cat',  # 'cat', 'stack', 'list'
    'clip_dim': 0,  # 0, 1
    'transform': transforms.Compose([transforms.ToTensor()]),
    'read_format': "PIL",
    'width': 256,
    'height': 256,
    'channel': 3
}

dict_patch_ST = {
    'augmode': 'random',  # 'normal_only', 'random', 'probability'
    'list_augtype': ['TMT', 'SRT'],
    'list_prob_weights': [0.5, 0.5],
    'patch_size_isRandom': False,  # 'cat', 'stack', 'list'
    'patch_size_range': [30, 90, 10],  # 0, 1
    'patch_size': 60,
    'width': 256,
    'height': 256,
    'h_cut_size': 30,
    'noise_isAdded': True,
    'mean': 0,
    'std': 0.03
}

dict_sf = {
    # skip frames dictionary
    'p': 0.0,  # p is probability
    's': [2, 3, 4, 5],  # s is skip frames
    'dataloader': None
}

dict_kf = {
    # key frames dictionary
    'p': 0.0,  # p is probability
    'dataloader': None
}

dict_obj = {
    # object-level dictionary
    'p': 0.0,  # p is probability
    'dataloader': None
}

dict_rf = {
    # repeat frames dictionary
    'p': 0.0,  # p is probability
    'r': [2, 3],  # r is repeat frames
    'dataloader': None
}

dict_patch = {
    'p': 0.0,  # p is probability
    'alpha': 0.5,  # alpha is maximum size of the patch relative to the frame size.
    'beta': 25,  # beta is to adjust the maximum movement of the patch in terms of pixels
    'technique': 'SmoothMixS',
    'intruder': 'CIFAR-100',
    'dataloader': None
}

dict_fusion = {
    'p': 0.0  # p is probability
}

dict_noise = {
    'p': 0.0,  # p is probability
    'sigma': 0.05
}

dict_shake = {
    'p': 0.0,  # p is probability
    'rot': (1, 3),  # range of degrees
    'trans': (0.01, 0.02)  # fraction for horizontal and vertical translations.
}
