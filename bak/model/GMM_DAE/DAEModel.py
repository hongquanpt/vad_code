from model.AbstractModel import AbstractModel
from .AE_model import SI_DAE, DI_DAE

from model.util.optimizer import get_optimizer, get_scheduler
from model.util.process_data import decode_clip_list

import torch
import time
import os

#=======================================
# NOTE: Đã implement
#=======================================
class DAEModel(AbstractModel):
    def __init__(self, cfg, device, logger, writer, img_type = 'Img'):
        #===========================================
        # Procedure for both of training and testing
        #===========================================
        super().__init__(cfg, device, logger, writer)
        # SI_DAE: static image, DI_DAE: dynamic image
        self.model_selector={"Img":SI_DAE,"Dimg":DI_DAE} 
        self.img_type = img_type
        self.model = self.model_selector[img_type]()
        self.model.to(self.device)
        
        self.optimizer = get_optimizer(cfg, self.model)
        self.scheduler = get_scheduler(cfg, self.optimizer)
        self.logger.info('Finish: DAEModel.__init__()')

    def save_model(self, model_filepath):
        print('call: save_model() in DAEModel')
        return super().save_model(model_filepath)
    
    def load_model(self, model_filepath):
        print('call: load_model() in DAEModel')
        return super().load_model(model_filepath)
    
    def save_model_state_dict(self, model_filepath):
        print('call: save_model_state_dict() in DAEModel')
        return super().save_model_state_dict(model_filepath)
    
    def load_model_state_dict(self, model_filepath):
        print('call: load_model_state_dict() in DAEModel')
        return super().load_model_state_dict(model_filepath)
    
    def save_checkpoint(self, epoch, model, optimizer, loss, checkpoint_filepath):
        print('call: save_checkpoint() in DAEModel')
        return super().save_checkpoint(epoch, model, optimizer, loss, checkpoint_filepath)
    
    def load_checkpoint(self, checkpoint_filepath):
        print('call: load_checkpoint() in DAEModel')
        return super().load_checkpoint(checkpoint_filepath)
    
    def save_checkpoint(self, epoch, model, optimizer, loss, checkpoint_filepath):
        try:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss,
                        }, 
                       checkpoint_filepath)
            self.logger.info(f'Success: save_checkpoint: {checkpoint_filepath}\n')
        except:
            self.logger.info(f'Error: save_checkpoint: {checkpoint_filepath} !!!\n')
            return 0
        return 1
    
    def load_checkpoint(self, checkpoint_filepath):
        try:
            checkpoint = torch.load(checkpoint_filepath)
            epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            loss = checkpoint['loss']
            self.logger.info(f'Success: load_checkpoint: {checkpoint_filepath} \n')
            return epoch
        except:
            self.logger.info(f'Error: load_checkpoint: {checkpoint_filepath} !!!\n')
            return -1
    
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
        #==============================================================
        for j, clip in enumerate(train_dataloader):
            #self.logger.info(f'[{j+1}/{len(train_dataloader)}]')
            img, target = decode_clip_list(data=clip, 
                                           model_type='reconstruction')
            img = img.to(self.device)
            out, fea, mse, ssim = self.model(img)
            loss = mse
            loss = torch.mean(loss) # with multiple gpus
            #print('loss = ', loss)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            loss_epoch += loss
        loss_epoch = loss_epoch / len(train_dataloader) # average the loss_epoch
        return loss_epoch
    
    def train(self, train_dataloader, checkpoint_dirpath):
        """
        train_dataloader: dataloader_Img, or dataloader_Dimg
        """
        #=========================================
        # Initalize some variables for training
        #=========================================
        train_time_start = time.time()
        loss = 0
        begin_epoch = self.cfg.TRAIN.begin_epoch
        end_epoch = self.cfg.TRAIN.end_epoch
        #==================================
        # Training process
        #==================================
        for epoch in range(begin_epoch, end_epoch):
            self.logger.info(f'epoch[{epoch+1}/{end_epoch}]')
            epoch_time_start = time.time()
            
            loss = self.train_epoch(train_dataloader)
            self.writer.add_scalar('training loss', loss, epoch)
            print('loss = ', loss)
            epoch_time_end = time.time()
            self.logger.info(f'>>>>>>> Epoch:{epoch+1} takes time: {epoch_time_end-epoch_time_start} seconds')
            self.scheduler.step()
            #==================================
            # Save the trained model by sequence
            #==================================
            if (epoch + 1) % self.cfg.TRAIN.save_freq == 0:
                checkpoint_filepath = os.path.join(checkpoint_dirpath,
                                                   f'{self.img_type}_epoch_{epoch + 1}.pt')
                self.save_checkpoint(epoch + 1, self.model, self.optimizer, loss, checkpoint_filepath)
        
        train_time_end = time.time()
        self.logger.info(f'===> Training takes time: {train_time_end-train_time_start} seconds')
        #==================================
        # Save the final trained model
        #==================================
        checkpoint_filepath = os.path.join(checkpoint_dirpath,
                                           f'{self.img_type}_{end_epoch}.pt')
        self.save_checkpoint(end_epoch, self.model, self.optimizer, loss, checkpoint_filepath)
        
        self.writer.flush() # to make sure that all pending events have been written to disk.
        self.writer.close() # If you do not need the summary writer anymore
        return 1