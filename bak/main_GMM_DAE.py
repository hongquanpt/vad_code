import argparse
from system.output import make_output_dir, make_logger, make_writer, make_dir_path
import torch.backends.cudnn as cudnn

from dataset.motion import generate_MotionImages, generate_OpticalFlow_Flownet2
from dataset.detect import detect_Objects_yolov5_training
from dataset.process import get_train_dataset_DAE, get_train_dataset_GMM
from model.GMM_DAE.DAEModel import DAEModel
from model.GMM_DAE.GMMModel import GMMModel
from model.GMM_DAE.evaluate import evaluate

import os
import torch
import yaml
from yacs.config import CfgNode

# ===========================================================
# ===========================================================
def preprocess_training(cfg):
    root = os.path.join(os.getcwd(), 'dataset')  # '/home/dataset'
    dataset_path = os.path.join(root, cfg.DATASET.name, cfg.DATASET.train.train_set)
    assert (os.path.exists(dataset_path))

    # 1. Create DI directory and generate Dynamic Images
    generate_MotionImages(dataset_path=dataset_path, motion_type='DI')

    # 2. Detect Objects
    detect_Objects_yolov5_training(dataset_path=dataset_path,
                                   detect_file='libs/pytorch_yolov5/detect_training.py',
                                   weights=cfg.DATASET.weights_yolov5)
    return 1


def preprocess_testing(cfg):
    root = os.path.join(os.getcwd(), 'dataset')  # '/home/dataset'
    dataset_path = os.path.join(root, cfg.DATASET.name, cfg.DATASET.test.testset)
    assert (os.path.exists(dataset_path))

    # 1. Create DI directory and generate Dynamic Images
    generate_MotionImages(dataset_path=dataset_path, motion_type='DI')
    return 1


# ===========================================================
# ===========================================================
def train_DAE(cfg, device, logger, writer, checkpoint_dirpath):
    logger.info('............>>> TRAIN_DAE PROCESS............>>>')
    # 1. Training DAE_Img
    train_dataset_Img, train_dataloader_Img = get_train_dataset_DAE(cfg, img_type='Img')
    logger.info(f'len(train_dataset_Img) = {len(train_dataset_Img)}')
    logger.info(f'len(train_dataloader_Img) = {len(train_dataloader_Img)}')
    DAE_Img = DAEModel(cfg, device, logger, writer, img_type='Img')
    DAE_Img.train(train_dataloader_Img, checkpoint_dirpath)

    # 2. Training DAE_Dimg
    train_dataset_Dimg, train_dataloader_Dimg = get_train_dataset_DAE(cfg, img_type='Dimg')
    logger.info(f'len(train_dataset_Dimg) = {len(train_dataset_Dimg)}')
    logger.info(f'len(train_dataloader_Dimg) = {len(train_dataloader_Dimg)}')
    DAE_Dimg = DAEModel(cfg, device, logger, writer, img_type='Dimg')
    DAE_Dimg.train(train_dataloader_Dimg, checkpoint_dirpath)
    return 1


def train_GMM(cfg, device, logger, writer, checkpoint_dirpath):
    logger.info('............>>> TRAIN_GMM PROCESS............>>>')
    # 1. Training on latent features with pretrained DAE_Img
    train_dataset_Img, train_dataloader_Img = get_train_dataset_GMM(cfg, img_type='Img')
    logger.info(f'len(train_dataset_Img) = {len(train_dataset_Img)}')
    logger.info(f'len(train_dataloader_Img) = {len(train_dataloader_Img)}')
    GMM_Img = GMMModel(cfg, device, logger, writer, img_type='Img')
    GMM_Img.train(train_dataloader_Img, checkpoint_dirpath)

    # 2. Training on latent features with pretrained DAE_Dimg
    train_dataset_Dimg, train_dataloader_Dimg = get_train_dataset_GMM(cfg, img_type='Dimg')
    logger.info(f'len(train_dataset_Dimg) = {len(train_dataset_Dimg)}')
    logger.info(f'len(train_dataloader_Dimg) = {len(train_dataloader_Dimg)}')
    GMM_Dimg = GMMModel(cfg, device, logger, writer, img_type='Dimg')
    GMM_Dimg.train(train_dataloader_Dimg, checkpoint_dirpath)
    return 1


# ===========================================================
# args, device
# ===========================================================
# https://github.com/vt-le/astnet
def parse_args(net: str):
    parser = argparse.ArgumentParser(description='VAD')

    parser.add_argument("--phase", help='train or test',
                        default='train', type=str)
    parser.add_argument("--dataset", help='ped2, avenue, or shanghaitech',
                        default='ped2', type=str)

    '''
    https://github.com/rbgirshick/yacs
    Now override from a list (opts could come from the command line)
    '''
    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # Read args from command line
    args = parser.parse_args()

    # Merge "cfg" from "cfg_file" and 'args.opts"
    cfg_file = os.path.join('system', net, f'{args.dataset}.yaml')
    print(f'======>>> cfg_file = {cfg_file}')

    # Create a new CfgNode with modifications
    modified_cfg = CfgNode()
    modified_cfg.SYSTEM = CfgNode()
    modified_cfg.SYSTEM.phase = args.phase
    modified_cfg.DATASET = CfgNode()
    modified_cfg.DATASET.name = args.dataset

    # Load the configuration file
    with open(cfg_file, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    # Create a new CfgNode: cfg is the original configuration
    cfg = CfgNode(config)

    # Merge the modified CfgNode with the original configuration
    cfg.merge_from_other_cfg(modified_cfg)

    # Merge other config options with the original configuration
    cfg.merge_from_list(args.opts)
    # cfg.freeze()
    return cfg


def setup_device(cfg, logger):
    num_gpus_available = torch.cuda.device_count()
    if num_gpus_available >= 1:
        logger.info(f"The number of GPUs available = {num_gpus_available}")
        cudnn.benchmark = cfg.SYSTEM.cudnn.benchmark  # true
        cudnn.determinstic = cfg.SYSTEM.cudnn.deterministic  # false
        cudnn.enabled = cfg.SYSTEM.cudnn.enable  # true

        device = torch.device(cfg.SYSTEM.device)  # "cuda:0"; "cuda:1"; "cpu"
        logger.info(f'device = {device}')
    else:
        logger.info("GPU unavailable")
        device = torch.device("cpu")
    return device


# ===========================================================
def generate_OpticalFlow(cfg):
    root = os.path.join(os.getcwd(), 'dataset')  # '/home/dataset'
    dataset_path = os.path.join(root, cfg.DATASET.name, cfg.DATASET.train.train_set)
    assert (os.path.exists(dataset_path))

    model_path = 'libs/flownet_networks/weights/FlowNet2_checkpoint.pth.tar'
    generate_OpticalFlow_Flownet2(dataset_path, model_path)
    return 1


# ===========================================================
def main(net: str):
    # ====================================
    # SYSTEM: cfg (config), output directory, logger, tensorboard, device ...
    # ====================================
    cfg = parse_args(net)
    output_dir, time_str = make_output_dir(cfg)
    logger, log_file = make_logger(output_dir, cfg.SYSTEM.output.log_dir)
    writer = make_writer(output_dir, cfg.SYSTEM.output.tesorboard_dir)
    device = setup_device(cfg, logger)
    logger.info(f'phase = {cfg.SYSTEM.phase}')

    if cfg.SYSTEM.phase == 'preprocess_training':
        preprocess_training(cfg)
    elif cfg.SYSTEM.phase == 'preprocess_testing':
        preprocess_testing(cfg)
    elif cfg.SYSTEM.phase == 'train_DAE':
        checkpoint_dir_path = make_dir_path(output_dir, cfg.SYSTEM.output.checkpoint_dir)
        train_DAE(cfg, device, logger, writer, checkpoint_dir_path)
    elif cfg.SYSTEM.phase == 'train_GMM':
        checkpoint_dir_path = make_dir_path(output_dir, cfg.SYSTEM.output.checkpoint_dir)
        train_GMM(cfg, device, logger, writer, checkpoint_dir_path)
    elif cfg.SYSTEM.phase == 'evaluate':
        visualization_dir_path = make_dir_path(output_dir, cfg.SYSTEM.output.visualization_dir)
        evaluate(cfg, device, logger, visualization_dir_path)
    elif cfg.SYSTEM.phase == 'generate_OpticalFlow':
        generate_OpticalFlow(cfg)


if __name__ == '__main__':
    main(net='GMM_DAE')
