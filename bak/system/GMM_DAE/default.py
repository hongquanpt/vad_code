from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN


cfg = CN()


# https://github.com/vt-le/astnet
def update_config(cfg_file, opts):
    global cfg
    cfg.defrost()
    cfg.merge_from_file(cfg_file)
    cfg.merge_from_list(opts)
    cfg.freeze()


# ==========================================
# 1. PROJECT related params
# configure the system related matters, such as gpus, cudnn and so on
# ===================
cfg.SYSTEM = CN()
cfg.SYSTEM.phase = 'train'
cfg.SYSTEM.loops = [1, 2]
cfg.SYSTEM.device = "cuda:1"  # "cuda:0"; "cuda:1"; "cpu"

# ===================
cfg.SYSTEM.cudnn = CN()
cfg.SYSTEM.cudnn.benchmark = True
cfg.SYSTEM.cudnn.deterministic = False
cfg.SYSTEM.cudnn.enable = True

# ===================
cfg.SYSTEM.output = CN()
cfg.SYSTEM.output.log_dir = 'log'
cfg.SYSTEM.output.tesorboard_dir = 'tensorboard'
cfg.SYSTEM.output.checkpoint_dir = 'checkpoint'
cfg.SYSTEM.output.visualization_dir = 'visualization'
# ==========================================
# 2. DATASET related params
# ===================
cfg.DATASET = CN()
cfg.DATASET.name = 'ped2'

cfg.DATASET.read_format = 'PIL'  # opencv
cfg.DATASET.image_format = 'jpg'

cfg.DATASET.width = 256
cfg.DATASET.height = 256
cfg.DATASET.channel = 3
cfg.DATASET.normalized = True

# prediction: num_frames=5, frame_steps=1
# recontruction: num_frames=1, frame_steps=1
cfg.DATASET.num_frames = 5
cfg.DATASET.frame_steps = 1
cfg.DATASET.lower_bound = 100
cfg.DATASET.clip_mode = 'cat'  # cat, stack, list
cfg.DATASET.clip_dim = 0
cfg.DATASET.weights_yolov5 = 'libs/pytorch_yolov5/weights/yolov5s.pt'

cfg.DATASET.augmode = 'normal_only'  # ['normal_only', 'random', 'probability']
cfg.DATASET.list_augtype = ['normal_only', 'TMT', 'SRT', 'gaussian_noise', 'simplex_noise']
cfg.DATASET.list_prob_weights = [0.0, 0.5, 0.5, 0.0, 0.0]

cfg.DATASET.patch_size_isRandom = True
cfg.DATASET.patch_size = 60
cfg.DATASET.patch_size_range = [30, 90, 10]  # = [start,stop,step]
cfg.DATASET.H_cut_size = 30

cfg.DATASET.noise_isAdded = True
cfg.DATASET.mean = 0
cfg.DATASET.std = 0.03
# ===================
cfg.DATASET.train = CN()
cfg.DATASET.train.train_set = 'training'
cfg.DATASET.train.batch_size_per_gpu_DAE = 4
cfg.DATASET.train.batch_size_per_gpu_GMM = 4
cfg.DATASET.train.shuffle = True
cfg.DATASET.train.num_workers = 0
cfg.DATASET.train.pin_memory = True
cfg.DATASET.train.drop_last = True
cfg.DATASET.train.sample_type = 'clips'  # 'videos', 'clips', 'clips_by_video'
cfg.DATASET.train.train_set_Img = 'training_Crop_Img'
cfg.DATASET.train.train_set_Dimg = 'training_Crop_Dimg'
# ===================
cfg.DATASET.test = CN()
cfg.DATASET.test.test_set = 'testing'
cfg.DATASET.test.batch_size_per_gpu = 1
cfg.DATASET.test.shuffle = False
cfg.DATASET.test.num_workers = 4
cfg.DATASET.test.pin_memory = True
cfg.DATASET.test.drop_last = False
cfg.DATASET.test.gt_filename = 'ped2.mat'
cfg.DATASET.test.sample_type = 'clips_by_video'  # videos', 'clips_by_video'
cfg.DATASET.test.label_filename = 'frame_labels_ped2.npy'

# ==========================================
# 3. MODEL related params
# ===============================

cfg.MODEL = CN()
cfg.MODEL.name = 'GMM_DAE'
cfg.MODEL.type = 'prediction'  # reconstruction

# ==========================================
# 4. TRAIN related params
# ===============================
cfg.TRAIN = CN()
cfg.TRAIN.name = 'pamae_trainer'

# EPOCH
cfg.TRAIN.begin_epoch = 0
cfg.TRAIN.end_epoch = 60
cfg.TRAIN.save_freq = 10
cfg.TRAIN.resume = False
cfg.TRAIN.checkpoint_filepath = ''

# LOSSES
cfg.TRAIN.losses = ['MSELoss', 1]

# OPTIMIZER
cfg.TRAIN.optimizer = CN()
cfg.TRAIN.optimizer.name = 'Adam'  # SGD, Adam, RMSprop,
cfg.TRAIN.optimizer.lr = 2e-4
cfg.TRAIN.optimizer.betas = [0.9, 0.999]
cfg.TRAIN.optimizer.eps = 1e-08
cfg.TRAIN.optimizer.weight_decay = 0

cfg.TRAIN.optimizer.momentum = 0.0  # SGD, RMSprop
cfg.TRAIN.optimizer.weight_decay = 0.0  # Adam, SGD, RMSprop
cfg.TRAIN.optimizer.nesterov = False  # SGD
cfg.TRAIN.optimizer.rmsprop_alpha = 0.99  # RMSprop
cfg.TRAIN.optimizer.rmsprop_centered = False  # RMSprop

# SCHEDULER
# https://machinelearningmastery.com/using-learning-rate-schedule-in-pytorch-training/
# https://towardsdatascience.com/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863
cfg.TRAIN.lr_scheduler = CN()
cfg.TRAIN.lr_scheduler.use = True
# LinearLR, LambdaLR, StepLR, MultiSetpLR, CosineAnnealingLR, ConstantLR, ExponentialLR, PolynomialLR
cfg.TRAIN.lr_scheduler.name = 'CosineAnnealingLR'
cfg.TRAIN.lr_scheduler.step_size = 20  # for StepLR
cfg.TRAIN.lr_scheduler.milestones = [20, 40]  # for MultiStepLR
cfg.TRAIN.lr_scheduler.gamma = 0.1  # for StepLR, MultiStepLR
cfg.TRAIN.lr_scheduler.eta_min = 0.0001  # Minimum learning rate for 'CosineAnnealingLR'

cfg.TRAIN.other = CN()
cfg.TRAIN.other.gmm_components = 15

# ===============================
# TESTING
# ===============================
cfg.TEST = CN()
cfg.TEST.name = 'pamae_tester'
cfg.TEST.checkpoint_DAE_Img = 'final_checkpoint_1.pth'
cfg.TEST.checkpoint_DAE_Dimg = 'final_checkpoint_1.pth'
cfg.TEST.checkpoint_GMM_Img = 'final_checkpoint_1.pth'
cfg.TEST.checkpoint_GMM_Dimg = 'final_checkpoint_1.pth'
