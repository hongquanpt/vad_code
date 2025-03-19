from model.AbstractModel import AbstractModel
from .ddpm_unconditional import *
from .ddpm_conditional import *
from .modules import UNet, UNet_conditional, EMA

from model.util.optimizer import get_optimizer
from model.util.train_test import decode_input_stack_clip
from model.util.visualization import export_images

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

import tqdm
import time
import os
import copy
#=======================================
# https://www.youtube.com/watch?v=a4Yfz2FxXiY
# NOTE: Chưa implement model này!
#=======================================
class DiffusionModel2(AbstractModel):
    def __init__(self, cfg, gpus, logger, writer):
        #===========================================
        # Procedure for both of training and testing
        #===========================================
        super().__init__(cfg, gpus, logger, writer)
        
        if self.cfg.DATASET.name == 'ped2':
            self.model = UNet()
            self.model = nn.DataParallel(self.model, device_ids=self.gpus).cuda()
            self.diffusion = Diffusion(img_size = cfg.DATASET.width)
        elif self.cfg.DATASET.name == 'cifar10-64':
            self.model = UNet_conditional(num_classes=10)
            self.model = nn.DataParallel(self.model, device_ids=self.gpus).cuda()
            self.ema = EMA(0.995)
            self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
            self.diffusion = Diffusion_conditional(img_size = cfg.DATASET.width)
            
        self.loss_fn = nn.MSELoss().cuda()
        self.optimizer = get_optimizer(cfg, self.model)
        
    def save_model(self, model_filepath):
        print('call: save_model() in DiffusionModel2')
        return super().save_model(model_filepath)
    
    def load_model(self, model_filepath):
        print('call: load_model() in DiffusionModel2')
        return super().load_model(model_filepath)
    
    def save_model_state_dict(self, model_filepath):
        print('call: save_model_state_dict() in DiffusionModel2')
        return super().save_model_state_dict(model_filepath)
    
    def load_model_state_dict(self, model_filepath):
        print('call: load_model_state_dict() in DiffusionModel2')
        return super().load_model_state_dict(model_filepath)
    
    def save_checkpoint(self, epoch, model, optimizer, loss, checkpoint_filepath):
        print('call: save_checkpoint() in DiffusionModel2')
        return super().save_checkpoint(epoch, model, optimizer, loss, checkpoint_filepath)
    
    def load_checkpoint(self, checkpoint_filepath):
        print('call: load_checkpoint() in DiffusionModel2')
        return super().load_checkpoint(checkpoint_filepath)
    
    #**********************************************************
    #************************** TRAINING **********************
    #**********************************************************
    
    def train(self, train_dataloader, checkpoint_dirpath):
        if self.cfg.DATASET.name == 'ped2':
            self.train_ped2(train_dataloader, checkpoint_dirpath)
        elif self.cfg.DATASET.name == 'cifar10-64':
            self.train_CIFAR10(train_dataloader, checkpoint_dirpath)
            
    def train_epoch_CIFAR10(self, train_dataloader):
        self.model.train()
        loss_epoch = 0
        self.logger.info(f'len(train_dataloader = {len(train_dataloader)}')
        
        for i, (images, labels) in enumerate(train_dataloader): # for cifar10
            #self.logger.info(f'i[{i+1}/{len(train_dataloader)}]')
        
        #pbar = tqdm(train_dataloader)
        #for i, (images, labels) in enumerate(pbar): # for cifar10-64
            
            # original images: torch.Size([8, 3, 64, 64]) 
            # 8 = batch_size_per_gpu(=4) * len(gpus)(=2)
            #self.logger.info(f'images.shape = {images.shape}')
            #self.logger.info(f'labels.shape = {labels.shape}')
            images = images.cuda()
            labels = labels.cuda()

            # sampling some timesteps t
            t = self.diffusion.sample_timesteps(images.shape[0]).cuda()

            # add noise to images at timesteps t
            x_t, noise = self.diffusion.noise_images(images, t)
            
            if np.random.random() < 0.1:
                labels = None

            # get predicted_noise from noised images x_t
            predicted_noise = self.model(x_t, t, labels)
            loss = self.loss_fn(noise, predicted_noise)
                
            # torch.Size([1, 3, 64, 64])
            #self.logger.info(f'loss.shape = {loss.shape}') 
            #self.logger.info(f'loss = {loss}')

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.ema.step_ema(self.ema_model, self.model)
            #pbar.set_postfix(MSE=loss.item())
            
            loss_epoch += loss
        return loss_epoch
    
    def train_CIFAR10(self, train_dataloader, checkpoint_dirpath):
        """
        Training a Conditional Diffusion on CIFAR10
        """
        #=========================================
        train_time_start = time.time()
        begin_epoch = self.cfg.TRAIN.begin_epoch
        end_epoch = self.cfg.TRAIN.end_epoch
        if self.cfg.TRAIN.resume:
            #==================================
            self.logger.info(">>>>>>>> RESUME TRAINING: >>>>>>>>>>>")
            #==================================
            checkpoint_filepath = self.cfg.TRAIN.checkpoint_filepath
            epoch, self.model, self.optimizer, loss = self.load_checkpoint(checkpoint_filepath)
            if epoch == -1: return
            begin_epoch = epoch
        #==================================
        # Training process
        #==================================
        loss = 0
        for epoch in range(begin_epoch, end_epoch):
            self.logger.info(f'epoch[{epoch+1}/{end_epoch}]')
            epoch_time_start = time.time()
            #==============================================
            loss = self.train_epoch_CIFAR10(train_dataloader)
            #sampled_images = self.diffusion.sample_images(self.model, n=images.shape[0])
            #==============================================
            self.logger.info(f'loss = {loss}')
            self.writer.add_scalar('training loss', loss, epoch)
            epoch_time_end = time.time()
            self.logger.info(f'>>>>>>> Epoch:{epoch+1} takes time: {epoch_time_end-epoch_time_start} seconds')
            #==================================
            # Save the trained model by sequence
            #==================================
            if (epoch + 1) % self.cfg.TRAIN.save_freq == 0:
                checkpoint_filepath = os.path.join(checkpoint_dirpath,f'epoch_{epoch + 1}.pth')
                self.save_checkpoint(epoch + 1, self.model, self.optimizer, loss, checkpoint_filepath)
                
                ema_checkpoint_filepath = os.path.join(checkpoint_dirpath,f'ema_epoch_{epoch + 1}.pth')
                self.save_checkpoint(epoch + 1, self.ema_model, self.optimizer, loss, ema_checkpoint_filepath)

        train_time_end = time.time()
        self.logger.info(f'===> Training takes time: {train_time_end-train_time_start} seconds')
        
        #==================================
        # Save the final trained model
        #==================================
        checkpoint_filepath = os.path.join(checkpoint_dirpath,'final.pth')
        self.save_checkpoint(end_epoch, self.model, self.optimizer, loss,  checkpoint_filepath)
        
        ema_checkpoint_filepath = os.path.join(checkpoint_dirpath,'ema_final.pth')
        self.save_checkpoint(end_epoch, self.ema_model, self.optimizer, loss,  ema_checkpoint_filepath)
        
        self.writer.flush() # to make sure that all pending events have been written to disk.
        self.writer.close() # If you do not need the summary writer anymore
        return 1
    
    def train_epoch_ped2(self, train_dataloader):
        self.model.train()
        loss_epoch = 0
        #for i, (images, labels) in enumerate(train_dataloader):
        for i, data in enumerate(train_dataloader):
            #self.logger.info(f'i[{i+1}/{len(train_dataloader)}]')
            # original images: torch.Size([8, 3, 64, 64]) 
            # 8 = batch_size_per_gpu(=4) * len(gpus)(=2)
            #self.logger.info(f'images.shape = {images.shape}')
            #self.logger.info(f'images.shape[0] = {images.shape[0]}')
            images, target = decode_input_stack_clip(input=data, model_type=self.cfg.MODEL.type)
            images = images.cuda()

            # sampling some timesteps t
            t = self.diffusion.sample_timesteps(images.shape[0]).cuda()

            # add noise to images at timesteps t
            x_t, noise = self.diffusion.noise_images(images, t)

            # get predicted_noise from noised images x_t
            predicted_noise = self.model(x_t, t)
            loss = self.loss_fn(noise, predicted_noise)
                
            # torch.Size([1, 3, 64, 64])
            #self.logger.info(f'loss.shape = {loss.shape}') 
            #self.logger.info(f'loss = {loss}')

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_epoch += loss
        return loss_epoch
        
    def train_ped2(self, train_dataloader, checkpoint_dirpath):
        """
        Training an Unconditional Diffusion on ped2
        """
        #=========================================
        train_time_start = time.time()
        begin_epoch = self.cfg.TRAIN.begin_epoch
        end_epoch = self.cfg.TRAIN.end_epoch
        if self.cfg.TRAIN.resume:
            #==================================
            self.logger.info(">>>>>>>> RESUME TRAINING: >>>>>>>>>>>")
            #==================================
            checkpoint_filepath = self.cfg.TRAIN.checkpoint_filepath
            epoch, self.model, self.optimizer, loss = self.load_checkpoint(checkpoint_filepath)
            if epoch == -1: return
            begin_epoch = epoch
        #==================================
        # Training process
        #==================================
        loss = 0
        for epoch in range(begin_epoch, end_epoch):
            self.logger.info(f'epoch[{epoch+1}/{end_epoch}]')
            epoch_time_start = time.time()
            #==============================================
            loss = self.train_epoch_ped2(train_dataloader)
            #sampled_images = self.diffusion.sample_images(self.model, n=images.shape[0])
            #==============================================
            self.logger.info(f'loss = {loss}')
            self.writer.add_scalar('training loss', loss, epoch)
            epoch_time_end = time.time()
            self.logger.info(f'>>>>>>> Epoch:{epoch+1} takes time: {epoch_time_end-epoch_time_start} seconds')
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
        self.save_checkpoint(end_epoch, self.model, self.optimizer, loss,  checkpoint_filepath)
        
        self.writer.flush() # to make sure that all pending events have been written to disk.
        self.writer.close() # If you do not need the summary writer anymore
        return 1
    
    #**********************************************************
    #*************************** TESTING **********************
    #**********************************************************
            
    def test(self, checkpoint_filepath, visualization_dirpath):
        #=========================
        # Load the pretrained model
        #=========================
        epoch, self.model, self.optimizer, loss = self.load_checkpoint(checkpoint_filepath)
        if epoch == -1: return
    
        n = 3
        if self.cfg.DATASET.name == 'ped2':
            x = self.diffusion.sample_images(self.model, n)
        elif self.cfg.DATASET.name == 'cifar10-64':
            y = torch.Tensor([6] * n).long().cuda()
            x = self.diffusion.sample_images(self.model, n, y, cfg_scale=0)
        export_images(x, visualization_dirpath)
            
        