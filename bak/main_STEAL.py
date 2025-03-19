import argparse
from system.output import make_output_dir, make_logger, make_writer, make_dir_path
import torch.backends.cudnn as cudnn

from dataset.process import get_train_dataset, get_skipframes_dataset, get_test_dataset
from dataset.label import LabelVideoDataset
from model.STEAL.STEALModel import STEALModel

import os
import torch
import yaml
from yacs.config import CfgNode


def train(cfg, device, logger, writer, checkpoint_dir_path):
    logger.info('TRAINING PROCESS............>>>')
    # ====================================
    # DATASET: Dataset, DataLoader
    # ====================================
    # 1. dataset, dataloader (successive frames)
    train_dataset, train_dataloader = get_train_dataset(cfg)
    logger.info(f'len(train_dataset) = {len(train_dataset)}')
    logger.info(f'len(train_dataloader) =  len(train_dataset) / batch_size = {len(train_dataloader)}')

    # 2. dataset, dataloader (skip_frames)
    train_dataset_sf, train_dataloader_sf = get_skipframes_dataset(cfg, len(train_dataset))

    logger.info(f'len(train_dataset_sf) = {len(train_dataset_sf)}')
    logger.info(
        f'len(train_dataloader_sf) =  len(train_dataset_sf) / batch_size = {len(train_dataloader_sf)}')

    # ====================================
    # MODEL: model, memory_items, loss_fn, optimizer, scheduler
    # ====================================
    # 4. load model
    steal = STEALModel(cfg, device, logger, writer)
    checkpoint_filepath = os.path.join('output', 'train',
                                       cfg.MODEL.name, cfg.DATASET.name,
                                       cfg.TEST.dir_name,
                                       'checkpoint', f'epoch_{cfg.TRAIN.begin_epoch}.pth')
    logger.info(f'checkpoint_filepath = {checkpoint_filepath}')
    steal.train(train_dataloader, train_dataloader_sf, checkpoint_filepath, checkpoint_dir_path)
    return 1


def test(cfg, device, logger, writer, visualization_dir_path):
    logger.info('TESTING PROCESS............>>>')
    # ====================================
    # DATASET: Dataset, DataLoader
    # ====================================
    test_dataset, test_dataloader = get_test_dataset(cfg)
    logger.info(f'len(test_dataset) = {len(test_dataset)}')
    logger.info(f'len(test_dataloader) = len(test_dataset) / batch_size = {len(test_dataloader)}')

    # Danh sách label tương ứng với các video trong test_tset
    label_list = LabelVideoDataset(cfg.DATASET.name, cfg.DATASET.test.test_set)()
    # print('label = ', label)

    # ====================================
    # MODEL: model, memory_items, loss_fn, optimizer, scheduler
    # ====================================
    steal = STEALModel(cfg, device, logger, writer)
    checkpoint_filepath = os.path.join('output', 'train',
                                       cfg.MODEL.name, cfg.DATASET.name,
                                       cfg.TEST.dir_name,
                                       'checkpoint', f'epoch_{cfg.TRAIN.end_epoch}.pth')
    logger.info(f'checkpoint_filepath = {checkpoint_filepath}')
    sub_visualization_dir_path = make_dir_path(visualization_dir_path, cfg.TEST.dir_name)
    steal.test(test_dataloader, label_list, checkpoint_filepath, sub_visualization_dir_path)
    writer.close()
    return 0


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

    # ====================================
    # Training and Testing process
    # ====================================
    logger.info(f'phase = {cfg.SYSTEM.phase}')
    if cfg.SYSTEM.phase == 'train':
        checkpoint_dir_path = make_dir_path(output_dir, cfg.SYSTEM.output.checkpoint_dir)
        train(cfg, device, logger, writer, checkpoint_dir_path)
    elif cfg.SYSTEM.phase == 'test':
        visualization_dir_path = make_dir_path(output_dir, cfg.SYSTEM.output.visualization_dir)
        test(cfg, device, logger, writer, visualization_dir_path)


if __name__ == '__main__':
    """
    python main_STEAL.py --phase 'train' --dataset 'ped2' SYSTEM.device "cuda:0"
    """
    main(net='STEAL')
