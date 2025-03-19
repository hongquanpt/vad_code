import argparse
from system.Diffusion2.default import update_config, cfg # for Diffusion2
from system.output import * # logger, make_output_dir
import torch.backends.cudnn as cudnn

from dataset.video import VideoDataset, TestVideoDataset
from dataset.label import LabelVideoDataset
from dataset.process import *

from model.Diffusion2.DiffusionModel2 import DiffusionModel2

import os
import torch
import torchvision

def train(cfg, gpus, logger, writer, checkpoint_dirpath):
    logger.info('TRAINING PROCESS............>>>')
    logger.info(f'cfg.DATASET.name = {cfg.DATASET.name}')
    #====================================
    # DATASET: Dataset, DataLoader
    #====================================
  
    if cfg.DATASET.name == 'ped2':
        train_dataset = VideoDataset(cfg, dataset_type = 'train', frame_mode = 'successive')
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = cfg.DATASET.train.batch_size_per_gpu * len(gpus),
            shuffle = cfg.DATASET.train.shuffle,
            num_workers = cfg.DATASET.train.num_workers,
            pin_memory = cfg.DATASET.train.pin_memory,
            drop_last = cfg.DATASET.train.drop_last
        )
    elif cfg.DATASET.name == 'cifar10-64':
        '''
        transform = get_transform2(w=cfg.DATASET.width, h=cfg.DATASET.height)
        train_dataset = get_cifar10_dataset(transform=transform)
        train_dataloader = get_cifar10_dataloader(cfg, train_dataset, gpus)
        '''
        transform = get_transform2(w=cfg.DATASET.width, h=cfg.DATASET.height)
        train_dataset = get_cifar10_dataset_ImageFolder(cfg, transform)
        train_dataloader = get_cifar10_dataloader(cfg, train_dataset, gpus)
    
    logger.info(f'len(train_dataset) = {len(train_dataset)}')
    logger.info(f'len(train_dataloader) =  len(train_dataset) / (batch_size * len(gpus)) = {len(train_dataloader)}')
    #====================================
    # MODEL: model, loss_fn, optimizer, scheduler
    #====================================
    if cfg.MODEL.name == 'Diffusion2':
        dm = DiffusionModel2(cfg, gpus, logger, writer)
        dm.train(train_dataloader, checkpoint_dirpath)
    return 1

def test(cfg, gpus, logger, writer, visualization_dirpath):
    logger.info('TESTING PROCESS............>>>')
    #====================================
    # MODEL: model, loss_fn, optimizer, scheduler
    #====================================
    if cfg.MODEL.name == 'Diffusion2':
        dm = DiffusionModel2(cfg, gpus, logger, writer)
        for i in range(0, len(cfg.TEST.checkpoint_filepaths)):
            logger.info(f'checkpoint_filepath[{i}] = {cfg.TEST.checkpoint_filepaths[i]}')
            checkpoint_filepath = cfg.TEST.checkpoint_filepaths[i]
            train_session_name = checkpoint_filepath.split('/')[-3]
            epoch_session_name = checkpoint_filepath.split('/')[-1].split('.')[0]
            logger.info(f'train_session_name = {train_session_name}')
            logger.info(f'epoch_session_name = {epoch_session_name}')
            sub_visualization_dirpath = make_dir_path(visualization_dirpath, 
                                                      train_session_name + '/' + epoch_session_name)
            dm.test(checkpoint_filepath, sub_visualization_dirpath)
    return 0

#===========================================================
# args, gpus
#===========================================================
# https://github.com/vt-le/astnet
def parse_args():

    parser = argparse.ArgumentParser(description='VAD')
    cfg_filename = 'system/Diffusion2/ddpm_unconditional.yaml'
    parser.add_argument('--cfgfile', 
                        help='experiment configuration filename',
                        default=cfg_filename, type=str)

    '''
    https://github.com/rbgirshick/yacs
    Now override from a list (opts could come from the command line)
    '''
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(cfg, args)
    return cfg, args

def get_gpus(cfg, logger):
    num_gpus_available = torch.cuda.device_count()
    if num_gpus_available > 1:
        logger.info(f"The number of GPUs available = {num_gpus_available}")
        cudnn.benchmark = cfg.SYSTEM.cudnn.benchmark
        cudnn.determinstic = cfg.SYSTEM.cudnn.deterministic
        cudnn.enabled = cfg.SYSTEM.cudnn.enable
        gpus = list(cfg.SYSTEM.gpus)
        
        logger.info(f"The number of GPUS is used = {len(gpus)}; gpus = {gpus}")
    else:
        logger.info("Single GPU or no GPU available")
    return gpus

#===========================================================
#===========================================================
def main():
    #====================================
    # SYSTEM: cfg (config), output directory, logger, tensorboard, gpus ...
    #====================================
    cfg, args = parse_args()
    output_dir, time_str = make_output_dir(cfg)
    logger, log_file = make_logger(output_dir, cfg.SYSTEM.output.log_dir)
    writer = make_writer(output_dir, cfg.SYSTEM.output.tesorboard_dir)
    checkpoint_dirpath = make_dir_path(output_dir, cfg.SYSTEM.output.checkpoint_dir)
    visualization_dirpath = make_dir_path(output_dir, cfg.SYSTEM.output.visualization_dir)
    
    logger.info(f'phase = {cfg.SYSTEM.phase}')
    logger.info(f'loop_index = {cfg.SYSTEM.loop_index}')
    
    gpus = get_gpus(cfg, logger)
    logger.info(f'gpus = {gpus}')
    if (cfg.SYSTEM.phase == 'train'):
        train(cfg, gpus, logger, writer, checkpoint_dirpath)
    else:
        test(cfg, gpus, logger, writer, visualization_dirpath)
        
if __name__ == '__main__':
    main()
