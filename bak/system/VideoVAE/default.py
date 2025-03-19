from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN
# https://github.com/vt-le/astnet
def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfgfile)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
cfg = CN()
#==========================================
# 1. PROJECT related params
# configure the system related matters, such as gpus, cudnn and so on
#===================
cfg.SYSTEM = CN()
cfg.SYSTEM.phase = 'train'
cfg.SYSTEM.loop_index = 1
cfg.SYSTEM.device = "cuda:1" # "cuda:0"; "cuda:1"; "cpu"

#===================
cfg.SYSTEM.cudnn = CN()
cfg.SYSTEM.cudnn.benchmark = True
cfg.SYSTEM.cudnn.deterministic = False
cfg.SYSTEM.cudnn.enable = True

#===================
cfg.SYSTEM.output = CN()
cfg.SYSTEM.output.log_dir = 'log'
cfg.SYSTEM.output.tesorboard_dir = 'tensorboard'
cfg.SYSTEM.output.checkpoint_dir = 'checkpoint'
cfg.SYSTEM.output.visualization_dir = 'visualization'
#==========================================
# 2. DATASET related params
#===================
cfg.DATASET = CN()
cfg.DATASET.name = 'ped2'

cfg.DATASET.read_format = 'PIL' # opencv
cfg.DATASET.image_format = 'jpg'

cfg.DATASET.width = 256
cfg.DATASET.height = 256
cfg.DATASET.channel = 3

# prediction: num_frames=5, frame_steps=1
# recontruction: num_frames=1, frame_steps=1
cfg.DATASET.num_frames = 5
cfg.DATASET.frame_steps = 1
cfg.DATASET.lower_bound = 100

cfg.DATASET.z_dim = 512
cfg.DATASET.h_dim = 512
#===================
cfg.DATASET.train = CN()
cfg.DATASET.train.trainset = 'training'
cfg.DATASET.train.batch_size_per_gpu = 4
cfg.DATASET.train.shuffle = True
cfg.DATASET.train.num_workers = 4
cfg.DATASET.train.pin_memory = True
cfg.DATASET.train.drop_last = True
#===================
cfg.DATASET.test = CN()
cfg.DATASET.test.testset = 'testing'
cfg.DATASET.test.batch_size_per_gpu = 1
cfg.DATASET.test.shuffle = False
cfg.DATASET.test.num_workers = 4
cfg.DATASET.test.pin_memory = True
cfg.DATASET.test.drop_last = False
cfg.DATASET.test.gt_filename = 'ped2.mat'

#==========================================
# 3. MODEL related params
#===============================

cfg.MODEL = CN()
cfg.MODEL.name = 'MNAD'
cfg.MODEL.type = 'prediction' # reconstruction

#==========================================
# 4. TRAIN related params
#===============================
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
cfg.TRAIN.optimizer.name = 'Adam' # SGD, Adam, RMSprop, 
cfg.TRAIN.optimizer.learning_rate = 2e-4

cfg.TRAIN.optimizer.momentum = 0.9 # SGD, RMSprop
cfg.TRAIN.optimizer.weight_decay = 0.1 # SGD, RMSprop
cfg.TRAIN.optimizer.nesterov = False # SGD
cfg.TRAIN.optimizer.rmsprop_alpha = 0.1 # RMSprop
cfg.TRAIN.optimizer.rmsprop_centered = 0.1 # RMSprop

# SCHEDULER
# https://machinelearningmastery.com/using-learning-rate-schedule-in-pytorch-training/
# https://towardsdatascience.com/a-visual-guide-to-learning-rate-schedulers-in-pytorch-24bbb262c863
cfg.TRAIN.lr_scheduler = CN()
cfg.TRAIN.lr_scheduler.use = True
# StepLR, MultiSetpLR, ConstantLR, LinearLR, ExponentialLR, PolynomialLR, CosineAnnealingLR
cfg.TRAIN.lr_scheduler.name = 'CosineAnnealingLR'
cfg.TRAIN.lr_scheduler.step_size = 1000 # for StepLR
cfg.TRAIN.lr_scheduler.steps = [63800, 127600] # for MultiStepLR
cfg.TRAIN.lr_scheduler.gamma = 0.1 # for StepLR, MultiStepLR

# OTHER_PARAMS
cfg.TRAIN.other_params = CN()
cfg.TRAIN.other_params.feature_dim = 512
cfg.TRAIN.other_params.memory_dim = 512
cfg.TRAIN.other_params.memory_size = 10
cfg.TRAIN.other_params.skip_frames = [2, 3, 4, 5]
cfg.TRAIN.other_params.skip_frames_prob = 0.01
cfg.TRAIN.other_params.patch_prob = 0.01
cfg.TRAIN.other_params.noise_prob = 0.01
cfg.TRAIN.other_params.fusion_prob = 0.01
cfg.TRAIN.other_params.loss_compact = 0.01
cfg.TRAIN.other_params.loss_separate = 0.01

#===============================
# TESTING
#===============================
cfg.TEST = CN()
cfg.TEST.name = 'pamae_tester'
cfg.TEST.checkpoint_filepaths =  ['final_checkpoint_1.pth', 'final_checkpoint_2.pth']

cfg.TEST.other_params = CN()
cfg.TEST.other_params.ano_score_alpha = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
cfg.TEST.other_params.update_threshold = 0.01




