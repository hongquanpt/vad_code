from model.AbstractModel import AbstractModel
from .ConvLSTM import ConvLSTM
from .VRNN import VRNN
from .GDL import GDL

#from .util.loss_util import IntensityLoss
from .loss_function import loss_function

from model.util.optimizer import get_optimizer, get_scheduler
from model.util.train_test import decode_input_stack_clip_adaptive, make_dir_path

from model.util.metrics import calculate_auc_scores_ConvVRNN, get_anomaly_rectanges
from model.util.auc import draw_auc_manual

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
# NOTE: Model này lấy từ bài báo: 2019-Future Frame Prediction Using Convolutional VRNN for Anomaly Detection
# 
# Trạng thái: đã train và test, nhưng kết quả tệ?
#=======================================
class ConvVRNNModel(AbstractModel):
    def __init__(self, cfg, device, logger, writer):
        #===========================================
        # Procedure for both of training and testing
        #===========================================
        super().__init__(cfg, device, logger, writer)
        torch.manual_seed(2020)
        
        self.model=VRNN(input_size=(16, 16),
                 input_dim= 1024,
                 hidden_dim=[512,256, 128, 64, 32],
                 kernel_size=(3, 3),
                 num_layers=5)
        self.model.to(self.device)
        self.gl = GDL()
        
        self.optimizer = get_optimizer(cfg, self.model)
        self.scheduler = get_scheduler(cfg, self.optimizer)
        self.logger.info('Finish: ConvVRNNModel.__init__()')

    def save_model(self, model_filepath):
        print('call: save_model() in ConvVRNNModel')
        return super().save_model(model_filepath)
    
    def load_model(self, model_filepath):
        print('call: load_model() in ConvVRNNModel')
        return super().load_model(model_filepath)
    
    def save_model_state_dict(self, model_filepath):
        print('call: save_model_state_dict() in ConvVRNNModel')
        return super().save_model_state_dict(model_filepath)
    
    def load_model_state_dict(self, model_filepath):
        print('call: load_model_state_dict() in ConvVRNNModel')
        return super().load_model_state_dict(model_filepath)
    
    def save_checkpoint(self, epoch, model, optimizer, loss, checkpoint_filepath):
        print('call: save_checkpoint() in ConvVRNNModel')
        return super().save_checkpoint(epoch, model, optimizer, loss, checkpoint_filepath)
    
    def load_checkpoint(self, checkpoint_filepath):
        print('call: load_checkpoint() in ConvVRNNModel')
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
        # train_dataloader:  Danh sách các clips (mỗi clip gồm 4 ảnh liên tiếp)
        total_step = len(train_dataloader)
        step = self.cfg.DATASET.channel # step=channel=3
        seq_len = (self.cfg.DATASET.num_frames-1) * step # =(4-1)*3=9
        for j, data in enumerate(train_dataloader):
            #self.logger.info(f'[{j+1}/{len(train_dataloader)}]') 
            
            imgs, gt = decode_input_stack_clip_adaptive(input=data, 
                                                   model_type=self.cfg.MODEL.type,seq_len=seq_len)

            imgs = imgs.to(self.device) # 4 frames
            gt = gt.to(self.device) # 5th frame
            
            #imgs.shape = torch.Size([1, 9, 256, 256])
            #self.logger.info(f'imgs.shape = {imgs.shape}') 
            #self.logger.info(f'imgs.size() = {imgs.size()}') 
            
            #torch.Size([1, 3, 256, 256])
            #self.logger.info(f'gt.shape = {gt.shape}') 
            
            recon_batch, kld_loss, recon_loss = self.model(imgs, seq_len, step)
            msssim, f1, psnr_error = loss_function(recon_batch, gt)
            gdl_loss = self.gl(recon_batch, gt)
            loss = 100* gdl_loss + recon_loss + 10*(msssim + f1)+kld_loss
            
            #print(f'loss1 = {loss}')
            #loss = torch.mean(loss) # with multiple gpus
            #print(f'loss2 = {loss}')
           
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            loss_epoch += loss
            
            if j % 100 == 0:
                self.logger.info('Step [{}/{}], Loss: {:.4f}, psnr_error:{:.4f}, msssim:{:.4f}, gdl_loss:{:.4f}, kld_loss:{:.4f}, recon_loss:{:.4f}'.format(j+1, total_step, loss.item(), psnr_error,msssim, gdl_loss,kld_loss, recon_loss))
                #Loss: 7.2420, psnr_error:1.6291, msssim:0.2095, gdl_loss:0.0173, kld_loss:0.0001, recon_loss:0.5017
            
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
        ef = self.cfg.DATASET.num_frames - 1 # encoded_frames = 4(predict), =1(reconst)
        df = self.cfg.DATASET.frame_steps - 1 # decoded_frames = 0
        pf = ef + df  # processed_frames
        
        step = self.cfg.DATASET.channel # step=channel=3
        seq_len = (self.cfg.DATASET.num_frames-1) * step # =(5-1)*3=12
        with torch.no_grad():
            #=========================
            # For each video
            #=========================
            for i, data in enumerate(test_dataloader):
                video = [frame for frame in data]
                loss_video = []
                frame_id_list = []
                #=========================
                # For each clips (4 frames) and predicted frame
                #=========================
                for f in range(len(video) - pf):
                    imgs = np.concatenate(video[f:f + pf], axis=1) # vì axis=0 (lien quan den batch_size)?
                    imgs = torch.Tensor(imgs).to(self.device)
                    gt = torch.Tensor(video[f + pf]).to(self.device)
                    frame_id_list.append(f + pf)
                    
                    recon_batch, kld_loss, recon_loss = self.model(imgs,seq_len, step)
                    msssim, f1, psnr_error = loss_function(recon_batch, gt)
                    gdl_loss = self.gl(recon_batch, gt)
                    loss = 100* gdl_loss + recon_loss + 10*(msssim + f1)
                    
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