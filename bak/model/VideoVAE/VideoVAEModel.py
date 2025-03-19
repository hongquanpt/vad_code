from model.AbstractModel import AbstractModel
from .model import VideoVAE
from .loss import VideoVAELoss

from model.util.loss import IntensityLoss
from model.util.optimizer import get_optimizer, get_scheduler
from model.util.train_test import decode_input, decode_input_stack_clip, get_lists_dict_with_prob, point_score, make_dir_path

from model.util.metrics import psnr_park, calculate_auc_scores, get_anomaly_rectanges
from model.util.auc_util import draw_auc_manual

from model.util.visualization import export_gt_img_cv2, export_output_img_cv2, export_mse_img_cv2, plot_anomaly_scores
 
from model.util.aug import patch_cifar_smoothborder
    
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

import tqdm
import time
import os

#=======================================
# NOTE: Model này lấy từ bài báo: 2018-ECCV-Probabilistic Video Generation using Holistic Attribute Control
# áp dụng cho bài toán Video Generation
# Trạng thái: mới chỉ cài đặt để chạy được phần training: Reconstruction
#=======================================
class VideoVAEModel(AbstractModel):
    def __init__(self, cfg, device, logger, writer):
        #===========================================
        # Procedure for both of training and testing
        #===========================================
        super().__init__(cfg, device, logger, writer)
        torch.manual_seed(2020)
        
        self.model = VideoVAE(cfg.DATASET.z_dim, cfg.DATASET.h_dim)
        self.model.to(self.device)
        
        self.loss_fn = VideoVAELoss(recon='L2')
        self.optimizer = get_optimizer(cfg, self.model)
        self.scheduler = get_scheduler(cfg, self.optimizer)
        self.logger.info('Finish: VideoVAEModel.__init__()')

    def save_model(self, model_filepath):
        print('call: save_model() in VideoVAEModel')
        return super().save_model(model_filepath)
    
    def load_model(self, model_filepath):
        print('call: load_model() in VideoVAEModel')
        return super().load_model(model_filepath)
    
    def save_model_state_dict(self, model_filepath):
        print('call: save_model_state_dict() in VideoVAEModel')
        return super().save_model_state_dict(model_filepath)
    
    def load_model_state_dict(self, model_filepath):
        print('call: load_model_state_dict() in VideoVAEModel')
        return super().load_model_state_dict(model_filepath)
    
    def save_checkpoint(self, epoch, model, optimizer, loss, checkpoint_filepath):
        print('call: save_checkpoint() in VideoVAEModel')
        return super().save_checkpoint(epoch, model, optimizer, loss, checkpoint_filepath)
    
    def load_checkpoint(self, checkpoint_filepath):
        print('call: load_checkpoint() in VideoVAEModel')
        return super().load_checkpoint(checkpoint_filepath)
    
    #**********************************************************
    #************************** TRAINING **********************
    #**********************************************************
    # training in each epoch
    def train_epoch(self, train_dataloader):
        self.model.train()
        loss_epoch = 0
        # The number of train_dataset (training clips) = len(train_dataset)
        # len(train_dataloader) =  len(train_dataset) / (batch_size * len(gpus))
        # train_dataloader:  Danh sách các clips (mỗi clip gồm 5 ảnh liên tiếp)
    
        for j, data in enumerate(train_dataloader):
            #self.logger.info(f'[{j+1}/{len(train_dataloader)}]')
            bz = self.cfg.DATASET.train.batch_size_per_gpu * len(self.gpus)
            
            h_prev, c_prev = self.model.module.reset(batch_size=bz)
            h_prev = h_prev.to(self.device)
            c_prev = c_prev.to(self.device)
                
            imgs, target = decode_input_stack_clip(input=data, 
                                                   model_type=self.cfg.MODEL.type)

            imgs = imgs.to(self.device)
            target = target.to(self.device)
            
            #imgs.shape = torch.Size([8, 15, 256, 256])
            #self.logger.info(f'imgs.shape = {imgs.shape}') 
            #self.logger.info(f'target.shape = {target.shape}') #torch.Size([8, 15, 256, 256])
            step = self.cfg.DATASET.channel # step=channel=3
            seq_len = self.cfg.DATASET.num_frames * step
            
            loss_recon_total = 0
            loss_kl_total = 0
            loss_t_total = 0
            for t in range(0, seq_len, step):
                x_t = imgs[:, t:t+step, :, :]
                #x_t.shape = torch.Size([8, 3, 64, 64])
                #self.logger.info(f'x_t.shape = {x_t.shape}') 
                # foward pass
                recon_x_t, z_t, lstm_output, [h_t, c_t], [mu_p, logvar_p], [mu_dy, logvar_dy] = self.model.module.forward(x_t, h_prev, c_prev)
                h_prev, c_prev = h_t, c_t
                
                loss_t, loss_recon, loss_kl = self.loss_fn(recon_x_t, x_t, 
                                                             [mu_p, logvar_p],
                                                             [mu_dy, logvar_dy])
                loss_t_total += loss_t
                loss_recon_total += loss_recon
                loss_kl_total += loss_kl
            
            loss = torch.mean(loss_t_total) # with multiple gpus
           
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            loss_epoch += loss
            
            if j % 100 == 0:
                self.logger.info(f'>>>>>>> loss_t_total: {loss_t_total}')
                self.logger.info(f'>>>>>>> loss_recon_total: {loss_recon_total}')
                self.logger.info(f'>>>>>>> loss_kl_total: {loss_kl_total}')
            
        return loss_epoch
        
    # training over all epochs
    def train(self, train_dataloader, checkpoint_dirpath):
        #======================================
        # Initalize some variables for training
        #======================================
        self.logger.info(f'skip_frame_prob = {self.cfg.TRAIN.other_params.skip_frames_prob}')
        #=========================================
        train_time_start = time.time()
        loss = 0
        begin_epoch = self.cfg.TRAIN.begin_epoch
        end_epoch = self.cfg.TRAIN.end_epoch
        if self.cfg.TRAIN.resume:
            #==================================
            self.logger.info(">>>>>>>> RESUME TRAINING: >>>>>>>>>>>")
            #==================================
            checkpoint_filepath = self.cfg.TRAIN.checkpoint_filepath
            epoch, self.model, self.optimizer, loss, self.m_items = self.load_checkpoint(checkpoint_filepath)
            if epoch == -1: return
            begin_epoch = epoch
        #==================================
        # Training process
        #==================================
        for epoch in range(begin_epoch, end_epoch):
            self.logger.info(f'epoch[{epoch+1}/{end_epoch}]')
            epoch_time_start = time.time()
            loss = self.train_epoch(train_dataloader)
            self.writer.add_scalar('training loss', loss, epoch)
            epoch_time_end = time.time()
            self.logger.info(f'>>>>>>> Epoch:{epoch+1} takes time: {epoch_time_end-epoch_time_start} seconds')
            self.logger.info(f'>>>>>>> loss: {loss}')
            self.scheduler.step()
            #==================================
            # Save the trained model by sequence
            #==================================
            if (epoch + 1) % self.cfg.TRAIN.save_freq == 0:
                checkpoint_filepath = os.path.join(checkpoint_dirpath,f'epoch_{epoch + 1}.pth')
                self.save_checkpoint(epoch + 1, self.model, self.optimizer, loss, checkpoint_filepath)

        train_time_end = time.time()
        self.logger.info(f'===> Training takes time: {train_time_end-train_time_start} seconds')
        
        #==================================
        # Save the final trained model
        #==================================
        checkpoint_filepath = os.path.join(checkpoint_dirpath,'final.pth')
        self.save_checkpoint(end_epoch, self.model, self.optimizer, loss, checkpoint_filepath)
        
        self.writer.flush() # to make sure that all pending events have been written to disk.
        self.writer.close() # If you do not need the summary writer anymore
        return 1