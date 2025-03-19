from model.AbstractModel import AbstractModel
from sklearn import mixture 
from .AE_model import SI_DAE, DI_DAE

from model.util.optimizer import get_optimizer, get_scheduler
from model.util.process_data import decode_clip_list

import torch
import numpy as np

import os, pickle

#=======================================
# NOTE: Đã implement
#=======================================
class GMMModel(AbstractModel):
    def __init__(self, cfg, device, logger, writer, img_type = 'Img'):
        #===========================================
        # Procedure for both of training and testing
        #===========================================
        super().__init__(cfg, device, logger, writer)
        self.model_selector={"Img":SI_DAE,"Dimg":DI_DAE} 
        self.img_type = img_type
        self.model = self.model_selector[img_type]()
        self.model.to(self.device)
        
        self.optimizer = get_optimizer(cfg, self.model)
        self.scheduler = get_scheduler(cfg, self.optimizer)
        
        if self.img_type == 'Img':
            checkpoint_filepath = cfg.TEST.checkpoint_DAE_Img
        else:
            checkpoint_filepath = cfg.TEST.checkpoint_DAE_Dimg
        _, self.model, _, _ = self.load_checkpoint(checkpoint_filepath)
        
        self.logger.info('Finish: GMMModel.__init__()')

    def save_model(self, model_filepath):
        print('call: save_model() in GMMModel')
        return super().save_model(model_filepath)
    
    def load_model(self, model_filepath):
        print('call: load_model() in GMMModel')
        return super().load_model(model_filepath)
    
    def save_model_state_dict(self, model_filepath):
        print('call: save_model_state_dict() in GMMModel')
        return super().save_model_state_dict(model_filepath)
    
    def load_model_state_dict(self, model_filepath):
        print('call: load_model_state_dict() in GMMModel')
        return super().load_model_state_dict(model_filepath)
    
    def save_checkpoint(self, epoch, model, optimizer, loss, checkpoint_filepath):
        print('call: save_checkpoint() in GMMModel')
        return super().save_checkpoint(epoch, model, optimizer, loss, checkpoint_filepath)
    
    def load_checkpoint(self, checkpoint_filepath):
        print('call: load_checkpoint() in GMMModel')
        return super().load_checkpoint(checkpoint_filepath)
    
    #**********************************************************
    #************************** TRAINING **********************
    #**********************************************************
    def get_feature(self, train_dataloader):
        self.logger.info(f'len(train_dataloader) = {len(train_dataloader)}')
        self.model.eval()
        Feature = None
        for i, clip in enumerate(train_dataloader):
            img, target = decode_clip_list(data=clip,
                                           model_type='reconstruction')
            img = img.to(self.device)
            out, fea, mse, ssim = self.model(img)
            self.logger.info(f'img.shape = {img.shape}')
            self.logger.info(f'fea.shape = {fea.shape}')
            if i==0:
                Feature=fea.cpu().detach().numpy()
            else:
                Feature=np.append(Feature,fea.cpu().detach().numpy(),axis=0)
        return np.reshape(Feature,(Feature.shape[0],-1))
    
    def train(self, train_dataloader, checkpoint_dirpath):
        self.logger.info(f'............>>> {self.img_type}: gmm start training ............>>> ')
        with torch.no_grad():
            X = self.get_feature(train_dataloader)
        gmm = mixture.GaussianMixture(n_components=self.cfg.TRAIN.other.gmm_components,verbose=1)
        gmm.fit(X)
        theta={}
        theta['pi'] = gmm.weights_
        theta['miu'] = gmm.means_
        theta['sigma'] = gmm.covariances_
        checkpoint_filepath = os.path.join(checkpoint_dirpath, "{}_GmmTheta.pkl".format(self.img_type))
        with open(checkpoint_filepath, "wb") as f:   
            try:
                pickle.dump(theta, f)
                self.logger.info(f'Success: save_checkpoint: {checkpoint_filepath}\n')
            except:
                self.logger.info(f'Error: save_checkpoint: {checkpoint_filepath} !!!\n')
        self.logger.info(f'==={self.img_type} gmm training done===')
        return 1

    