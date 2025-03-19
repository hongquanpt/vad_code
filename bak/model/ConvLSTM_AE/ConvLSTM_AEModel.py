from model.AbstractModel import AbstractModel
from .ConvLSTM_AE import ConvLSTM_AE

from model.util.loss import IntensityLoss
from model.util.optimizer import get_optimizer, get_scheduler
from model.util.train_test import make_dir_path

from model.util.metrics import psnr_park, calculate_auc_scores_ConvLSTM_AE, get_anomaly_rectanges
from model.util.auc import draw_auc_manual

from model.util.visualization import export_gt_img_cv2, export_output_img_cv2, export_mse_img_cv2, plot_anomaly_scores
    
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
# tham khảo github:
# - video_anomaly_detection_pytorch-master
# - spatio-temporal-autoencoder-for-videos-main

# Adapted from https://github.com/automan000/Convolution_LSTM_PyTorch/blob/master/convolution_lstm.py
#=======================================
class ConvLSTM_AEModel(AbstractModel):
    def __init__(self, cfg, device, logger, writer):
        #===========================================
        # Procedure for both of training and testing
        #===========================================
        super().__init__(cfg, device, logger, writer)
        kernel_size = 4 # (4, 4) kernel
        init_channels = 8 # initial number of filters
        image_channels = 1 # MNIST images are grayscale
        latent_dim = 16 # latent dimension for sampling
        
        self.model = ConvLSTM_AE()
        self.model.to(self.device)
        
        self.loss_fn = nn.MSELoss()
        self.optimizer = get_optimizer(cfg, self.model)
        self.scheduler = get_scheduler(cfg, self.optimizer)
        self.logger.info('Finish: ConvLSTM_AEModel.__init__()')

    def save_model(self, model_filepath):
        print('call: save_model() in ConvLSTM_AEModel')
        return super().save_model(model_filepath)
    
    def load_model(self, model_filepath):
        print('call: load_model() in ConvLSTM_AEModel')
        return super().load_model(model_filepath)
    
    def save_model_state_dict(self, model_filepath):
        print('call: save_model_state_dict() in ConvLSTM_AEModel')
        return super().save_model_state_dict(model_filepath)
    
    def load_model_state_dict(self, model_filepath):
        print('call: load_model_state_dict() in ConvLSTM_AEModel')
        return super().load_model_state_dict(model_filepath)
    
    def save_checkpoint(self, epoch, model, optimizer, loss, checkpoint_filepath):
        print('call: save_checkpoint() in ConvLSTM_AEModel')
        return super().save_checkpoint(epoch, model, optimizer, loss, checkpoint_filepath)
    
    def load_checkpoint(self, checkpoint_filepath):
        print('call: load_checkpoint() in ConvLSTM_AEModel')
        return super().load_checkpoint(checkpoint_filepath)
    
    #**********************************************************
    #************************** TRAINING **********************
    #**********************************************************
    # training in each epoch
    def train_epoch(self, train_dataloader):
        self.model.train()
        loss_epoch = 0
        total_step = len(train_dataloader)
        step = self.cfg.DATASET.channel # step=channel=1
        seq_len = (self.cfg.DATASET.num_frames-1) * step # =(10-1)*1=9 # for prediction
        for j, data in enumerate(train_dataloader):
            imgs=data # but we apply reconstruction
            imgs = imgs.to(self.device) # 10 frames ([1, 10, 1, 227, 227])
            #self.logger.info(f'imgs.shape = {imgs.shape}')
            '''
            reconstruction, mu, logvar = self.model(imgs)
            RE = self.loss_fn(reconstruction, imgs)#0.0205
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())#12864.4043
            loss = RE + KLD 
            '''
            rec_imgs = self.model(imgs)
            loss = self.loss_fn(rec_imgs, imgs)
           
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            loss_epoch += loss
            
            if j % 5 == 0:
                #self.logger.info('Step [{}/{}], Loss: {:.4f}, RE:{:.4f}, KLD:{:.4f}'.format(j+1, total_step, loss.item(), RE, KLD))
                self.logger.info('Step [{}/{}], Loss: {:.4f}'.format(j+1, total_step, loss.item()))
            
        return loss_epoch
        
    # training over all epochs
    def train(self, train_dataloader, checkpoint_dirpath):
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
    
    #**********************************************************
    #*************************** TESTING **********************
    #**********************************************************
    def test(self, test_dataloader, label_list, checkpoint_filepath, visualization_dirpath):
        #=========================
        # Load the pretrained model
        #=========================
        epoch, self.model, self.optimizer, loss = self.load_checkpoint(checkpoint_filepath)
        if epoch == -1: return
        #=========================
        # Evaluate model
        #=========================
        self.model.eval()
        video_losses = []
        video_ids = []
        pf = 1  # processed_frames
        with torch.no_grad():
            #=========================
            # For each video
            #=========================
            for i, data in enumerate(test_dataloader):
                video = [frame for frame in data]
                
                frame_losses = []
                frame_ids = []
                #=========================
                # For each frame
                #=========================
                for j in range(len(video)):
                    img = video[j].unsqueeze(0)
                    img = torch.Tensor(img).to(self.device)
                    frame_ids.append(j)
                    
                    #torch.Size([1, 1, 1, 227, 227])
                    #self.logger.info(f'img.shape = {img.shape}')
                    '''
                    rec_imgs, mu, logvar = self.model(imgs)
                    rec_loss = self.loss_fn(rec_imgs, imgs)
                    loss = rec_loss
                    '''
                    rec_img = self.model(img)
                    loss = self.loss_fn(rec_img, img)
                    
                    #print(f'loss = {loss}')
                    frame_losses.append(loss.cpu())
                
                video_losses.append(frame_losses)
                video_ids.append(frame_ids)
        
        assert len(video_losses) == len(label_list), f'Ground truth has {len(label_list)} videos, BUT got {len(video_losses)} detected videos!'
        
        export_dir = make_dir_path(visualization_dirpath, 'ano_scores')
        auc, fpr, tpr, score_list = calculate_auc_scores_ConvLSTM_AE(self.cfg, video_losses, label_list)
        for video_id in range(len(label_list)):
            rectangles_start, rectangles_end = get_anomaly_rectanges(label_list, video_id)
            scores_video = score_list[video_id]
            frame_ids = video_ids[video_id]
            plot_anomaly_scores(video_id, frame_ids, scores_video, rectangles_start, rectangles_end, export_dir)
        #======================================================
        # Export AUC graph
        #======================================================
        self.logger.info(f'AUC: {auc * 100:.2f}%')
        auc_filepath = os.path.join(export_dir, 'AUC.png') 
        draw_auc_manual(fpr, tpr, color='darkorange', filepath=auc_filepath)