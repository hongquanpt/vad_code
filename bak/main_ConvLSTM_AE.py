import argparse
from system.ConvLSTM_AE.default import update_config, cfg # for ConvLSTM_AE
from system.output import * # logger, make_output_dir
import torch.backends.cudnn as cudnn

from dataset.process import *

from model.ConvLSTM_AE.ConvLSTM_AEModel import ConvLSTM_AEModel

import os
import torch

def train(cfg, device, logger, writer, checkpoint_dirpath):
    logger.info('TRAINING PROCESS............>>>')
    #====================================
    # DATASET: Dataset, DataLoader
    #====================================
    train_dataset, train_dataloader = get_train_dataset(cfg)
    logger.info(f'len(train_dataset) = {len(train_dataset)}')
    logger.info(f'len(train_dataloader) =  len(train_dataset) / batch_size = {len(train_dataloader)}')
    
    #====================================
    # MODEL: model, memory_items, loss_fn, optimizer, scheduler
    #====================================
    ConvLSTM_AE = ConvLSTM_AEModel(cfg, gpus, logger, writer)
    ConvLSTM_AE.train(train_dataloader, checkpoint_dirpath)
   
    return 1

def test(cfg, device, logger, writer, visualization_dirpath):
    logger.info('TESTING PROCESS............>>>')
    #====================================
    # DATASET: Dataset, DataLoader
    #====================================
    test_dataset, test_dataloader = get_test_dataset(cfg)
    logger.info(f'len(test_dataset) = {len(test_dataset)}')
    logger.info(f'len(test_dataloader) = len(test_dataset) / batch_size  = {len(test_dataloader)}')
    label_list = LabelVideoDataset(cfg)() # Danh sách label tương ứng với các video trong testset
    #label_list = label_list()
    #print('label_list = ', label_list)
    print(f'len(label_list) = {len(label_list)}')
    #====================================
    # MODEL: model, memory_items, loss_fn, optimizer, scheduler
    #====================================
    if cfg.MODEL.name == 'ConvLSTM_AE':
        ConvLSTM_AE = ConvLSTM_AEModel(cfg, device, logger, writer)
        for i in range(0, len(cfg.TEST.checkpoint_filepaths)):
            logger.info(f'checkpoint_filepath[{i}] = {cfg.TEST.checkpoint_filepaths[i]}')
            checkpoint_filepath = cfg.TEST.checkpoint_filepaths[i]
            train_session_name = checkpoint_filepath.split('/')[-3]
            epoch_session_name = checkpoint_filepath.split('/')[-1].split('.')[0]
            logger.info(f'train_session_name = {train_session_name}')
            logger.info(f'epoch_session_name = {epoch_session_name}')
            sub_visualization_dirpath = make_dir_path(visualization_dirpath, 
                                                      train_session_name + '/' + epoch_session_name)
            ConvLSTM_AE.test(test_dataloader, label_list, checkpoint_filepath, sub_visualization_dirpath)
    return 0

#===========================================================
# args, device
#===========================================================
# https://github.com/vt-le/astnet
def parse_args():

    parser = argparse.ArgumentParser(description='VAD')
    cfg_filename = 'system/ConvLSTM_AE/ped2.yaml'
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

def setup_device(cfg, logger):
    num_gpus_available = torch.cuda.device_count()
    if num_gpus_available > 1:
        logger.info(f"The number of GPUs available = {num_gpus_available}")
        cudnn.benchmark = cfg.SYSTEM.cudnn.benchmark # true
        cudnn.determinstic = cfg.SYSTEM.cudnn.deterministic # false
        cudnn.enabled = cfg.SYSTEM.cudnn.enable # true
        
        device = torch.device(cfg.SYSTEM.device) # "cuda:0"; "cuda:1"; "cpu"
        logger.info(f'device = {device}')
    else:
        logger.info("Single GPU or no GPU available")
    return device

#===========================================================
#===========================================================    
def main():
    #====================================
    # SYSTEM: cfg (config), output directory, logger, tensorboard, device ...
    #====================================
    cfg, args = parse_args()
    output_dir, time_str = make_output_dir(cfg)
    logger, log_file = make_logger(output_dir, cfg.SYSTEM.output.log_dir)
    writer = make_writer(output_dir, cfg.SYSTEM.output.tesorboard_dir)
    checkpoint_dirpath = make_dir_path(output_dir, cfg.SYSTEM.output.checkpoint_dir)
    visualization_dirpath = make_dir_path(output_dir, cfg.SYSTEM.output.visualization_dir)
    
    logger.info(f'phase = {cfg.SYSTEM.phase}')
    logger.info(f'loop_index = {cfg.SYSTEM.loop_index}')
    
    device = setup_device(cfg, logger)
    if (cfg.SYSTEM.phase == 'train'):
        train(cfg, device, logger, writer, checkpoint_dirpath)
    else:
        test(cfg, device, logger, writer, visualization_dirpath)
        
if __name__ == '__main__':
    main()
