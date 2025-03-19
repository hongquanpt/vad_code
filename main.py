import os
import time
import torch
import yaml
from yacs.config import CfgNode
import argparse
from system.output import make_logger, make_dir_path
import torch.backends.cudnn as cudnn

from dataset.process import get_train_dataset, get_test_dataset
from dataset.process import get_train_dataset_FastAno
from dataset.process import get_data_sf, get_data_rf, get_data_patch, get_data_kf
from dataset.process import get_data_fusion, get_data_noise, get_data_shake
from dataset.label import LabelVideoDataset

from model.MNAD.MNADModel import MNADModel
from model.MNAD.MNAD_woMem_Model import MNAD_woMem_Model
from model.MNAD.PAMAEModel import PAMAEModel
from model.MNAD.PAMAE_woMem_Model import PAMAE_woMem_Model
from model.MNAD.PBMAEModel import PBMAEModel
from model.STEAL.PseudoBoundModel import PseudoBoundModel

from model.ASTNet.ASTNetModel import ASTNetModel
from model.FastAno.FastAnoModel import FastAnoModel
from model.STEAL.STEALModel import STEALModel

import sys
import configparser

# Check if the library exists
import importlib.util

lib_neptune = "neptune"
if importlib.util.find_spec(lib_neptune):
    # print(f"The '{lib_neptune}' library is available.")
    import neptune
else:
    print(f"The '{lib_neptune}' library is not available") 

sys.path.append('../')


def transfer_scheduler_checkpoint(cfg, device, logger, run, output_dir):
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ')
    logger.info('>>>>>>>>>> CHECKPOINT PROCESS >>>>>>>>>> ')
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ')
    checkpoint_filepath_source = os.path.join('output', 'train',
                                              cfg.MODEL.name, cfg.MODEL.method, cfg.DATASET.name,
                                              cfg.TRAIN.checkpoint_source,
                                              'checkpoint', cfg.TRAIN.file_name)
    logger.info(f'checkpoint_filepath = {checkpoint_filepath_source}')

    checkpoint_filepath_destination = os.path.join('output', 'train',
                                                   cfg.MODEL.name, cfg.MODEL.method, cfg.DATASET.name,
                                                   cfg.TRAIN.checkpoint_destination,
                                                   'checkpoint', cfg.TRAIN.file_name)
    logger.info(f'checkpoint_filepath = {checkpoint_filepath_destination}')

    checkpoint_dir_path = make_dir_path(output_dir, cfg.SYSTEM.output.checkpoint_dir)
    data_clip = None
    if cfg.MODEL.name == 'PAMAE':
        model = PAMAEModel(cfg, device, logger, run, data_clip)
        model.transfer_scheduler_checkpoint(checkpoint_filepath_source, checkpoint_filepath_destination)


def train(cfg, device, logger, run, output_dir):
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ')
    logger.info('>>>>>>>>>> TRAINING PROCESS >>>>>>>>>> ')
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ')
    # ====================================
    # DATASET: Dataset, DataLoader
    # ====================================
    train_dataloader, data_clip = get_train_dataset(cfg)
    logger.info(f'len(train_dataloader) =  len(train_dataset) / batch_size = {len(train_dataloader)}')

    # ====================================
    # MODEL: model, loss_fn, optimizer, scheduler
    # ====================================
    checkpoint_filepath = os.path.join('output', 'train',
                                       cfg.MODEL.name, cfg.MODEL.method, cfg.DATASET.name,
                                       cfg.TRAIN.dir_name,
                                       'checkpoint', cfg.TRAIN.file_name)
    logger.info(f'checkpoint_filepath = {checkpoint_filepath}')
    checkpoint_dir_path = make_dir_path(output_dir, cfg.SYSTEM.output.checkpoint_dir)

    # Initialize model
    model = None
    if cfg.MODEL.name == 'MNAD':
        model = MNADModel(cfg, device, logger, run, data_clip)
    elif cfg.MODEL.name == 'MNAD_woMem':
        model = MNAD_woMem_Model(cfg, device, logger, run, data_clip)
    elif cfg.MODEL.name == 'PAMAE':
        model = PAMAEModel(cfg, device, logger, run, data_clip)
    elif cfg.MODEL.name == 'PAMAE_woMem':
        model = PAMAE_woMem_Model(cfg, device, logger, run, data_clip)
    elif cfg.MODEL.name == 'PBMAE':
        model = PBMAEModel(cfg, device, logger, run, data_clip)
        data_sf = get_data_sf(cfg)
        data_rf = get_data_rf(cfg)
        data_patch = get_data_patch(cfg)
        data_fusion = get_data_fusion(cfg)
        data_noise = get_data_noise(cfg)
        data_shake = get_data_shake(cfg)
        model.set_aug_data(data_sf, data_rf, data_patch, data_fusion, data_noise, data_shake)
    elif cfg.MODEL.name == 'PseudoBound':
        model = PseudoBoundModel(cfg, device, logger, run, data_clip)
        data_sf = get_data_sf(cfg)
        data_rf = get_data_rf(cfg)
        data_patch = get_data_patch(cfg)
        data_fusion = get_data_fusion(cfg)
        data_noise = get_data_noise(cfg)
        data_shake = get_data_shake(cfg)
        model.set_aug_data(data_sf, data_rf, data_patch, data_fusion, data_noise, data_shake)
    elif cfg.MODEL.name == 'ASTNet':
        model = ASTNetModel(cfg, device, logger, run)
    elif cfg.MODEL.name == 'FastAno':
        train_dataloader = get_train_dataset_FastAno(cfg)
        model = FastAnoModel(cfg, device, logger, run, data_clip)
    elif cfg.MODEL.name == 'STEAL':
        model = STEALModel(cfg, device, logger, run, data_clip)
        # set dataloader_skip_frame
        data_sf = get_data_sf(cfg)
        model.set_aug_data(data_sf)

    # Start training....
    if model is not None:
        model.train(train_dataloader, checkpoint_filepath, checkpoint_dir_path)
    if run is not None:
        run.stop()


def test(cfg, device, logger, run, output_dir):
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ')
    logger.info('>>>>>>>>>> TESTING PROCESS >>>>>>>>>> ')
    logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ')
    # ====================================
    # DATASET: Dataset, DataLoader
    # ====================================
    test_dataloader, data_clip = get_test_dataset(cfg)
    logger.info(f'len(test_dataloader) = len(test_dataset) / batch_size = {len(test_dataloader)}')

    # Danh sách label tương ứng với các video trong test_tset
    video_labels = LabelVideoDataset(cfg.DATASET.name, cfg.DATASET.test.test_set)()
    # print('label = ', label)

    # ====================================
    # MODEL: model, loss_fn, optimizer, scheduler
    # ====================================
    checkpoint_filepath = os.path.join('output', 'train',
                                       cfg.MODEL.name, cfg.MODEL.method, cfg.DATASET.name,
                                       cfg.TEST.dir_name,
                                       'checkpoint', cfg.TEST.file_name)
    logger.info(f'checkpoint_filepath = {checkpoint_filepath}')
    visualization_dir_path = make_dir_path(output_dir, cfg.SYSTEM.output.visualization_dir)

    model = None
    if cfg.MODEL.name == 'MNAD':
        model = MNADModel(cfg, device, logger, run, data_clip)
    elif cfg.MODEL.name == 'MNAD_woMem':
        model = MNAD_woMem_Model(cfg, device, logger, run, data_clip)
    elif cfg.MODEL.name == 'PAMAE':
        model = PAMAEModel(cfg, device, logger, run, data_clip)
    elif cfg.MODEL.name == 'PAMAE_woMem':
        model = PAMAE_woMem_Model(cfg, device, logger, run, data_clip)
    elif cfg.MODEL.name == 'PBMAE':
        model = PBMAEModel(cfg, device, logger, run, data_clip)
    elif cfg.MODEL.name == 'PseudoBound':
        model = PseudoBoundModel(cfg, device, logger, run, data_clip)
    elif cfg.MODEL.name == 'ASTNet':
        model = ASTNetModel(cfg, device, logger, run)
    elif cfg.MODEL.name == 'FastAno':
        model = FastAnoModel(cfg, device, logger, run, data_clip)
    elif cfg.MODEL.name == 'STEAL':
        model = STEALModel(cfg, device, logger, run, data_clip)

    if model is not None:
        model.test(test_dataloader, video_labels, checkpoint_filepath, visualization_dir_path)
    if run is not None:
        run.stop()


# ===========================================================
# args, device
# ===========================================================
# https://github.com/vt-le/astnet
def parse_args():
    parser = argparse.ArgumentParser(description='VAD')

    parser.add_argument("--model", help='MNAD or MNAD and more...',
                        default='MNAD', type=str)
    parser.add_argument("--method", help='PAMAE_SF, PAMAE_SF_KF_OF and more...',
                        default='PAMAE_SF', type=str)
    parser.add_argument("--phase", help='train or test',
                        default='train', type=str)
    parser.add_argument("--dataset", help='ped2, avenue, or shanghaitech',
                        default='ped2', type=str)
    parser.add_argument("--device", help='cuda:0, cuda:1, cpu',
                        default='cuda:0', type=str)

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

    # Merge "cfg" from "cfg_filepath" and 'args.opts"
    cfg_filepath = os.path.join(os.getcwd(), 'system', args.model, args.method, f'{args.dataset}.yaml')
    print(f'======>>> cfg_filepath = {cfg_filepath}')

    # Create a new CfgNode with modifications: SYSTEM.phase, DATASET.name
    modified_cfg = CfgNode()
    modified_cfg.SYSTEM = CfgNode()
    modified_cfg.SYSTEM.phase = args.phase
    modified_cfg.SYSTEM.device = args.device

    modified_cfg.DATASET = CfgNode()
    modified_cfg.DATASET.name = args.dataset

    # Load the configuration file
    with open(cfg_filepath, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    # Create a new CfgNode: cfg is the original configuration
    cfg = CfgNode(config)

    # Merge the modified CfgNode with the original configuration
    cfg.merge_from_other_cfg(modified_cfg)

    # Merge other config options with the original configuration
    # args.opts: SYSTEM.device, SYSTEM.neptune.use, TRAIN.resume
    cfg.merge_from_list(args.opts)
    # cfg.freeze()
    return cfg, cfg_filepath


def setup_device(cfg, logger):
    num_gpus_available = torch.cuda.device_count()
    if num_gpus_available >= 1:
        logger.info(f"The number of GPUs available = {num_gpus_available}")
        cudnn.enabled = cfg.SYSTEM.cudnn.enable  # true
        cudnn.benchmark = cfg.SYSTEM.cudnn.benchmark  # true
        cudnn.determinstic = cfg.SYSTEM.cudnn.deterministic  # false

        device = torch.device(cfg.SYSTEM.device)  # "cuda:0"
        logger.info(f'device = {device}')
    else:
        logger.info("GPU unavailable")
        device = torch.device("cpu")
    return device


def setup_neptune(cfg, cfg_filepath, output_dir):
    run = None
    neptune_config_filepath = os.path.join(os.getcwd(), 'system', 'neptune.config')
    if os.path.isfile(neptune_config_filepath) and cfg.SYSTEM.neptune.use:
        neptune_config = configparser.RawConfigParser()
        neptune_config.read(neptune_config_filepath)

        # get api_token and neptune_project
        api_token = neptune_config.get('neptune', 'api_token')
        neptune_project = neptune_config.get('neptune', 'neptune_project')

        #
        epochs = 0
        batch_size = 1
        if cfg.SYSTEM.phase == 'train':
            epochs = cfg.TRAIN.end_epoch
            batch_size = cfg.DATASET.train.batch_size_per_gpu
        else:
            # 'epoch_60.pth'
            epochs = int(cfg.TEST.file_name.split('.')[0].split('_')[1])
            batch_size = cfg.TEST.train_batch_size_per_gpu
        # define params
        params = {"output_dir": output_dir, "phase": cfg.SYSTEM.phase,
                  "device": cfg.SYSTEM.device, "model": cfg.MODEL.name, "method": cfg.MODEL.method,
                  "dataset": cfg.DATASET.name, "epochs": epochs, "batch_size": batch_size,
                  "optimizer": cfg.TRAIN.optimizer.name, "lr": cfg.TRAIN.optimizer.lr}

        run = neptune.init_run(
            project=neptune_project,
            api_token=api_token,
            name=cfg.MODEL.name + "_" + cfg.DATASET.name
        )
        run["parameters"] = params
        run["config"].upload(cfg_filepath)
    return run


def make_output_dir(cfg):
    time_str = time.strftime('%Y_%m_%d_%H_%M_%S')  # 2023_08_07_10_34
    if cfg.SYSTEM.phase == 'train':
        # Get the current working directory ("/home/asus/DATA/VAD")
        root_dir = os.path.join(os.getcwd(), 'output', cfg.SYSTEM.phase,
                                cfg.MODEL.name, cfg.MODEL.method, cfg.DATASET.name)
        dir_name = f'session_{time_str}'
        make_dir_path(root_dir, dir_name)
        output_dir = os.path.join(root_dir, dir_name)
        return output_dir
    if cfg.SYSTEM.phase == 'test':
        root_dir = os.path.join(os.getcwd(), 'output', cfg.SYSTEM.phase,
                                cfg.MODEL.name, cfg.MODEL.method, cfg.DATASET.name,
                                cfg.TEST.dir_name)
        dir_name = f'session_{time_str}'
        make_dir_path(root_dir, dir_name)
        output_dir = os.path.join(root_dir, dir_name)
        return output_dir


# ===========================================================
# ===========================================================
def run_train(cfg, cfg_filepath):
    output_dir = os.getcwd()
    # 2. Create output directory
    if cfg.TRAIN.resume:
        output_dir = os.path.join(os.getcwd(), 'output', cfg.SYSTEM.phase,
                                  cfg.MODEL.name, cfg.MODEL.method, cfg.DATASET.name,
                                  cfg.TRAIN.dir_name)
        print("Reusing output_dir = ", output_dir)
    else:
        output_dir = make_output_dir(cfg)
        print("Creating a new output_dir = ", output_dir)

    # 3. Create logger to print results into a text file (.log)
    logger, log_file = make_logger(output_dir, cfg.SYSTEM.output.log_dir)

    # 4. Setup device (CPU or GPU)
    device = setup_device(cfg, logger)

    # 5. Setup neptune.ai
    run = setup_neptune(cfg, cfg_filepath, output_dir)

    # 6. Training process
    logger.info(f'phase = {cfg.SYSTEM.phase}')
    train(cfg, device, logger, run, output_dir)


def run_test(cfg, cfg_filepath):
    # 2. Create output directory
    output_dir = make_output_dir(cfg)

    # 3. Create logger to print results into a text file (.log)
    logger, log_file = make_logger(output_dir, cfg.SYSTEM.output.log_dir)

    # 4. Setup device (CPU or GPU)
    device = setup_device(cfg, logger)

    # 5. Setup neptune.ai
    run = setup_neptune(cfg, cfg_filepath, output_dir)

    # 6. Testing process
    logger.info(f'phase = {cfg.SYSTEM.phase}')
    test(cfg, device, logger, run, output_dir)


def scheduler_checkpoint(cfg, cfg_filepath):
    output_dir = os.getcwd()
    # 2. Create output directory
    output_dir = os.path.join(os.getcwd(), 'output', cfg.SYSTEM.phase,
                              cfg.MODEL.name, cfg.MODEL.method, cfg.DATASET.name,
                              cfg.TRAIN.checkpoint_destination)
    print("checkpoint_destination = ", output_dir)

    # 3. Create logger to print results into a text file (.log)
    logger, log_file = make_logger(output_dir, cfg.SYSTEM.output.log_dir)

    # 4. Setup device (CPU or GPU)
    device = setup_device(cfg, logger)

    # 5. Setup neptune.ai
    run = setup_neptune(cfg, cfg_filepath, output_dir)

    # 6. Training process
    logger.info(f'phase = {cfg.SYSTEM.phase}')
    transfer_scheduler_checkpoint(cfg, device, logger, run, output_dir)


def main():
    # SYSTEM: cfg (config), output directory, logger, tensorboard, device ...
    # 1. Read file config + args from command line
    cfg, cfg_filepath = parse_args()

    if cfg.SYSTEM.phase == 'train':
        if cfg.TRAIN.resume:
            run_train(cfg, cfg_filepath)
        else:
            for i in range(cfg.SYSTEM.num_run):
                run_train(cfg, cfg_filepath)
    elif cfg.SYSTEM.phase == 'test':
        run_test(cfg, cfg_filepath)
    elif cfg.SYSTEM.phase == 'scheduler_checkpoint':  # scheduler_checkpoint
        scheduler_checkpoint(cfg, cfg_filepath)


if __name__ == '__main__':
    """
    #===================================================
    # MNAD, MNAD_woMem
    #===================================================
    python main.py --model MNAD --method MNAD --phase train --dataset ped2 --device cuda:0 TRAIN.resume False
    python main.py --model MNAD --method MNAD --phase test --dataset ped2 --device cuda:0 TEST.dir_name session_2024_01_11_23_22
    python main.py --model MNAD_woMem --method MNAD_woMem --phase train --dataset ped2 --device cuda:0
    
    #===================================================
    # PAMAE
    #===================================================
    python main.py --model PAMAE --method PAMAE_KF_OF --phase train --dataset ped2 --device cuda:0
    python main.py --model PAMAE --method PAMAE_KF_ES --phase train --dataset ped2 --device cuda:0 
    python main.py --model PAMAE --method PAMAE_KF_EM --phase train --dataset ped2 --device cuda:0 
    python main.py --model PAMAE --method PAMAE_KF_EN --phase train --dataset ped2 --device cuda:0 
   
    python main.py --model PAMAE --method PAMAE_SF --phase train --dataset ped2 --device cuda:0
    python main.py --model PAMAE --method PAMAE_SF_KF_OF --phase train --dataset ped2 --device cuda:0 TRAIN.resume True
    python main.py --model PAMAE --method PAMAE_SF_KF_OF --phase scheduler_checkpoint --dataset ped2 --device cuda:0
    python main.py --model PAMAE --method PAMAE_SF_KF_EN --phase train --dataset ped2
    #===================================================
    # PAMAE_woMem
    #===================================================
    python main.py --model PAMAE_woMem --method PAMAE_woMem_KF_OF --phase train --dataset ped2 --device cuda:0
    python main.py --model PAMAE_woMem --method PAMAE_woMem_KF_ES --phase train --dataset ped2 --device cuda:0
    python main.py --model PAMAE_woMem --method PAMAE_woMem_KF_EM --phase train --dataset ped2 --device cuda:0
    python main.py --model PAMAE_woMem --method PAMAE_woMem_KF_EN --phase train --dataset ped2 --device cuda:0
  
    
    python main.py --model PAMAE_woMem --method PAMAE_woMem_SF --phase train --dataset ped2 --device cuda:0
    python main.py --model PAMAE_woMem --method PAMAE_woMem_SF_KF --phase train --dataset ped2 --device cuda:0
   
    #===================================================
    # STEAL, PBMAE, PseudoBound
    #===================================================
    python main.py --model STEAL --method STEAL --phase train --dataset ped2 --device cuda:0
    python main.py --model PBMAE --method PBMAE --phase train --dataset ped2 --device cuda:0
    python main.py --model PseudoBound --method PseudoBound --phase train --dataset ped2 --device cuda:0
    
    
    #===================================================
    # ASTNet
    #===================================================
    python main.py --model ASTNet --method ASTNet --phase train --dataset ped2 --device cuda:0
    """
    main()
