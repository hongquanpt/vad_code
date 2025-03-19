from model.AbstractModel import AbstractModel
from .ConvVAE_Recon import ConvVAE_Recon

from model.util.loss import IntensityLoss
from model.util.optimizer import get_optimizer, get_scheduler
from model.util.train_test import decode_input_stack_clip_adaptive, make_dir_path

from model.util.metrics import psnr_park, calculate_auc_scores, get_anomaly_rectanges
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
# NOTE: 
# https://debuggercafe.com/convolutional-variational-autoencoder-in-pytorch-on-mnist-dataset/
# https://github.com/agrija9/Convolutional-VAE-for-3D-Turbulence-Data/blob/main/models.py
#=======================================
class ConvVAEModel2(AbstractModel):
    def __init__(self, cfg, device, logger, writer):
        #===========================================
        # Procedure for both of training and testing
        #===========================================
        super().__init__(cfg, device, logger, writer)
        kernel_size = 4 # (4, 4) kernel
        init_channels = 8 # initial number of filters
        image_channels = 1 # MNIST images are grayscale
        latent_dim = 16 # latent dimension for sampling
        
        self.model = ConvVAE_Recon(n_channel =3, num_frames = 2)
        self.model.to(self.device)
        
        #self.loss_fn = nn.BCELoss(reduction='sum')
        self.loss_fn = nn.MSELoss()
        self.optimizer = get_optimizer(cfg, self.model)
        self.scheduler = get_scheduler(cfg, self.optimizer)
        self.logger.info('Finish: ConvVAEModel2.__init__()')

    def save_model(self, model_filepath):
        print('call: save_model() in ConvVAEModel2')
        return super().save_model(model_filepath)
    
    def load_model(self, model_filepath):
        print('call: load_model() in ConvVAEModel2')
        return super().load_model(model_filepath)
    
    def save_model_state_dict(self, model_filepath):
        print('call: save_model_state_dict() in ConvVAEModel2')
        return super().save_model_state_dict(model_filepath)
    
    def load_model_state_dict(self, model_filepath):
        print('call: load_model_state_dict() in ConvVAEModel2')
        return super().load_model_state_dict(model_filepath)
    
    def save_checkpoint(self, epoch, model, optimizer, loss, checkpoint_filepath):
        print('call: save_checkpoint() in ConvVAEModel2')
        return super().save_checkpoint(epoch, model, optimizer, loss, checkpoint_filepath)
    
    def load_checkpoint(self, checkpoint_filepath):
        print('call: load_checkpoint() in ConvVAEModel2')
        return super().load_checkpoint(checkpoint_filepath)
    
    #**********************************************************
    #************************** TRAINING **********************
    #**********************************************************
    # training in each epoch
    def train_epoch(self, train_dataloader):
        self.model.train()
        loss_epoch = 0
        total_step = len(train_dataloader)
        step = self.cfg.DATASET.channel # step=channel=3
        seq_len = (self.cfg.DATASET.num_frames-1) * step # =(1-1)*3=0
        for j, data in enumerate(train_dataloader):
            imgs, gt = decode_input_stack_clip_adaptive(input=data, 
                                                   model_type=self.cfg.MODEL.type,seq_len=seq_len)
            imgs = imgs.to(self.device) # 1 frame ([1, 3, 256, 256])
            
            reconstruction, mu, logvar = self.model(imgs)
            RE = self.loss_fn(reconstruction, imgs)#0.0205
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())#12864.4043
            loss = RE + KLD 
           
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            loss_epoch += loss
            
            if j % 100 == 0:
                self.logger.info('Step [{}/{}], Loss: {:.4f}, RE:{:.4f}, KLD:{:.4f}'.format(j+1, total_step, loss.item(), RE, KLD))
            
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
        loss_list = []
        video_id_list = []
        
        with torch.no_grad():
            #=========================
            # For each video
            #=========================
            for i, data in enumerate(test_dataloader):
                loss_video = []
                frame_id_list = []
                #=========================
                # For each clips (1 frame)
                #=========================
                for f in range(len(data)):
                    imgs = data[f]
                    imgs = torch.Tensor(imgs).to(self.device)
                    frame_id_list.append(f)
                    
                    rec_imgs, mu, logvar = self.model(imgs)
                    rec_loss = self.loss_fn(rec_imgs, imgs)
                    loss = rec_loss
                    
                    #print(f'loss = {loss}')
                    loss_video.append(loss.cpu())
                
                loss_list.append(loss_video)
                video_id_list.append(frame_id_list)
        
        assert len(loss_list) == len(label_list), f'Ground truth has {len(label_list)} videos, BUT got {len(loss_list)} detected videos!'
        
        export_dir = make_dir_path(visualization_dirpath, 'ano_scores')
        auc, fpr, tpr, score_list = calculate_auc_scores_ConvVRNN(self.cfg, loss_list, label_list)
        for video_id in range(len(label_list)):
            rectangles_start, rectangles_end = get_anomaly_rectanges(label_list, video_id)
            scores_video = score_list[video_id]
            frame_id_list = video_id_list[video_id]
            plot_anomaly_scores(video_id, frame_id_list, scores_video, rectangles_start, rectangles_end, export_dir)
        #======================================================
        # Export AUC graph
        #======================================================
        self.logger.info(f'AUC: {auc * 100:.2f}%')
        auc_filepath = os.path.join(export_dir, 'AUC.png') 
        draw_auc_manual(fpr, tpr, color='darkorange', filepath=auc_filepath)