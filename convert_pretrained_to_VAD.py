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
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from model.util.optimizer import get_optimizer, get_scheduler
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
                        default='test', type=str)
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
    cfg_filepath = "system/MNAD/MNAD/ped2.yaml"
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





if __name__ == '__main__':
    from torch import optim
    pretrained_model = torch.load("pretrained/Ped2_prediction_model.pth")
    m_items = torch.load("pretrained/Ped2_prediction_keys.pt")


    checkpoint_filepath = "pretrained_MNAD_checkpoint.pth"

    cfg, cfg_filepath = parse_args()
    
    output_dir = "abcd"
    logger, log_file = make_logger(output_dir, cfg.SYSTEM.output.log_dir)

    # 4. Setup device (CPU or GPU)
    device = setup_device(cfg, logger)

    # 5. Setup neptune.ai
    run = setup_neptune(cfg, cfg_filepath, output_dir)

    # 6. Testing process
    logger.info(f'phase = {cfg.SYSTEM.phase}')
    
    model = MNADModel(cfg, device, logger, run, data_clip=None)
                         
    print("save model!!!")
    torch.save({
        'epoch': 1,
        'model_state_dict': pretrained_model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
        'scheduler_state_dict': model.scheduler.state_dict(),
        'loss': 1000,
        'm_items': m_items
    }, checkpoint_filepath)
    
    model.load_checkpoint(checkpoint_filepath)
    print("load model successfully")

