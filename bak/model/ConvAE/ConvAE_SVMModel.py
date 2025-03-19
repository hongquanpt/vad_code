from model.AbstractModel import AbstractModel
from .ConvAE_SVM import ConvAE_SVM

from model.util.loss import L2Loss
from model.util.optimizer import get_optimizer, get_scheduler
from model.util.train_test import decode_input_stack_clip_adaptive, make_dir_path, get_lists_dict_with_prob

from model.util.metrics import psnr_park, calculate_accuracy_ocsvm, get_anomaly_rectanges
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

# http://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
from sklearn import svm
#https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
from  sklearn.svm import LinearSVC
from  sklearn.metrics  import  accuracy_score , roc_curve , auc , f1_score, confusion_matrix, classification_report, roc_auc_score

from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

import numpy as np
import pickle # to save trained sklearn.svm.OneClassSVM
#=======================================
# NOTE: 
# https://debuggercafe.com/convolutional-variational-autoencoder-in-pytorch-on-mnist-dataset/
# https://github.com/agrija9/Convolutional-VAE-for-3D-Turbulence-Data/blob/main/models.py
#=======================================
class ConvAE_SVMModel(AbstractModel):
    def __init__(self, cfg, device, logger, writer):
        #===========================================
        # Procedure for both of training and testing
        #===========================================
        super().__init__(cfg, device, logger, writer)
        
        self.model = ConvAE_SVM(n_channel =3, num_frames = 1)
        self.model.to(self.device)
        
        # A higher nu means more emphasis on minimizing the volume, and a lower nu means more emphasis on minimizing the outliers.
        self.ocsvm = svm.OneClassSVM(kernel='linear', gamma='scale', nu=0.9) # svm.OneClassSVM: for one-class
        #self.ocsvm = LinearSVC (random_state = 0) # LinearSVC: for multi-class
        
        #self.loss_fn = nn.BCELoss(reduction='sum')
        self.loss_fn = nn.MSELoss()
        self.l2_loss = L2Loss()
        self.optimizer = get_optimizer(cfg, self.model)
        self.scheduler = get_scheduler(cfg, self.optimizer)
        self.logger.info('Finish: ConvAE_SVMModel.__init__()')

    def save_model(self, model_filepath):
        print('call: save_model() in ConvAE_SVMModel')
        return super().save_model(model_filepath)
    
    def load_model(self, model_filepath):
        print('call: load_model() in ConvAE_SVMModel')
        return super().load_model(model_filepath)
    
    def save_model_state_dict(self, model_filepath):
        print('call: save_model_state_dict() in ConvAE_SVMModel')
        return super().save_model_state_dict(model_filepath)
    
    def load_model_state_dict(self, model_filepath):
        print('call: load_model_state_dict() in ConvAE_SVMModel')
        return super().load_model_state_dict(model_filepath)
    
    def save_checkpoint(self, epoch, model, optimizer, loss, checkpoint_filepath):
        print('call: save_checkpoint() in ConvAE_SVMModel')
        return super().save_checkpoint(epoch, model, optimizer, loss, checkpoint_filepath)
    
    def load_checkpoint(self, checkpoint_filepath):
        print('call: load_checkpoint() in ConvAE_SVMModel')
        return super().load_checkpoint(checkpoint_filepath)
    
    #**********************************************************
    #************************** TRAINING **********************
    #**********************************************************
    # training in each epoch
    def train_epoch(self, train_dataloader):
        self.model.train()
        loss_epoch = 0
        total_step = len(train_dataloader)
   
        for j, data in enumerate(train_dataloader):
            # 'reconstruction'
            imgs, gt = decode_input_stack_clip_adaptive(input=data, 
                                                   model_type=self.cfg.MODEL.type) 
            imgs = imgs.to(self.device) # 1 frame ([1, 3, 256, 256])
            
            reconstruction, fea = self.model(imgs)
            z_flatten = torch.flatten(fea)
            
            RE = self.loss_fn(reconstruction, imgs)#0.0205
            tensor_zeros = torch.zeros(z_flatten.shape)
            regularizer = self.cfg.TRAIN.other_params.lamda * self.l2_loss(z_flatten, 0)
            loss = RE + regularizer 
           
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            loss_epoch += loss
            
            if j % 500 == 0:
                self.logger.info('Step [{}/{}], Loss: {:.4f}, RE:{:.4f}, regularizer:{:.4f}'.format(j+1, total_step, loss.item(), RE, regularizer))
            
        return loss_epoch
    
    def train_OCSVM(self, train_dataloader, checkpoint_dirpath, nu):
        
        train_features = []
        train_labels = []
        self.model.eval()
        with torch.no_grad():
            for j, data in enumerate(train_dataloader):
                # 'reconstruction'
                imgs, gt = decode_input_stack_clip_adaptive(input=data, 
                                                       model_type=self.cfg.MODEL.type) 
                imgs = imgs.to(self.device) # 1 frame ([1, 3, 256, 256])

                fea = self.model.module.get_features(imgs)
                # fea.shape = torch.Size([1, 512, 32, 32])
                #print(f'fea.shape = {fea.shape}') # 
                train_features.append(torch.flatten(fea).cpu().numpy())
                train_labels.append(-1)
        
        # converting list to array
        train_features = np.array(train_features)
        print(f'train_features.shape = {train_features.shape}')
        train_labels = np.array(train_labels)
        print(f'train_labels.shape = {train_labels.shape}')
        
        ocsvm_time_start = time.time()
        # train_features: X{array-like, sparse matrix} of shape (n_samples, n_features)
        #self.ocsvm.fit(train_features, train_labels) # LinearSVC: for multi-class
        self.ocsvm.fit(train_features) # svm.OneClassSVM: for one-class
        ocsvm_time_end = time.time()
        self.logger.info(f'>>>>>>> Train OCSVM with nu = {nu} takes time: {ocsvm_time_end-ocsvm_time_start} seconds')
        
        # Save the model
        ocsvm_filepath = os.path.join(checkpoint_dirpath, f'ocsvm_{nu}.pkl')
        with open(ocsvm_filepath, 'wb') as f:
            pickle.dump(self.ocsvm, f)
            self.logger.info(f'===> Save ocsvm model: {ocsvm_filepath}')
            
        '''
        # Load the ocsvm model
        with open(ocsvm_filepath, 'rb') as f:
            self.ocsvm = pickle.load(f)
        '''
          
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
            epoch, self.model, self.optimizer, loss = self.load_checkpoint(checkpoint_filepath)
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
                checkpoint_filepath = os.path.join(checkpoint_dirpath, f'epoch_{epoch + 1}.pth')
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
        
        #==================================
        # Train OCSVM classifier
        #==================================
        # nu : the upper limit ratio of anomaly data (0<=nu<=1)
        
        for nu in self.cfg.TEST.other_params.nu: # nu = [0.1, 0.5, 0.9]
            self.ocsvm = svm.OneClassSVM(kernel='linear', gamma='scale', nu=nu)
            self.train_OCSVM(train_dataloader, checkpoint_dirpath, nu)
        return 1
    
    #**********************************************************
    #*************************** TESTING **********************
    #**********************************************************
    def test_2(self, test_dataloader, label_list, checkpoint_filepath, ocsvm_filepath, visualization_dirpath):

        # Load the pretrained model
        epoch, self.model, self.optimizer, loss = self.load_checkpoint(checkpoint_filepath)
        if epoch == -1: return
    
        # Load the pretrained ocsvm model
        with open(ocsvm_filepath, 'rb') as f:
            self.ocsvm = pickle.load(f)
            self.logger.info(f'===> Load ocsvm model: {ocsvm_filepath}')
        
        # Evaluate model
        self.model.eval()
        video_ids = []
        video_features = []
        with torch.no_grad():
            
            # For each video
            for i, data in enumerate(test_dataloader):
                frame_ids = []
                frame_features = []
                
                # For each clips (1 frame)
                for f in range(len(data)):
                    imgs = data[f]
                    imgs = torch.Tensor(imgs).to(self.device)
                    frame_ids.append(f)
                    
                    fea = self.model.module.get_features(imgs)
                    # fea.shape = torch.Size([1, 512, 32, 32])
                    #print(f'fea.shape = {fea.shape}') # 
                    frame_features.append(torch.flatten(fea).cpu().numpy())
                
                video_ids.append(frame_ids)
                video_features.append(frame_features)
        
        assert len(video_features) == len(label_list), f'Ground truth has {len(label_list)} videos, BUT got {len(video_features)} detected videos!'
        
        
        pred_labels = []
        test_labels = []
        for i in range(len(video_features)):
            frame_features = video_features[i]
            
             # converting list to array
            frame_features = np.array(frame_features)
            print(f'frame_features.shape = {frame_features.shape}')
            
            # ocsvm.predict: return +1(inlier=normal) or -1 (outliers=anomaly)
            y_pred = self.ocsvm.predict(frame_features) * (-1) # convert to +1 (anomaly), -1 (normal)
            pred_labels.append(np.array(y_pred))
            
            test_labels.append(np.array(label_list[i]))
        
        pred_labels = np.concatenate(pred_labels, axis = 0)
        self.logger.info(f'pred_labels.shape = {pred_labels.shape}')
        self.logger.info(f'pred_labels = {pred_labels}')
        
        test_labels = np.concatenate(test_labels, axis = 0)
        self.logger.info(f'test_labels.shape = {test_labels.shape}')
        
        # mark test_labels[i]==0 (normal) to -1, test_labels[i]==1 (anomaly) to 1
        test_labels[test_labels == 0] = -1
        self.logger.info(f'test_labels = {test_labels}')
        
        # calculate F1-score
        # https://machinelearningmastery.com/one-class-classification-algorithms/
        score = f1_score(test_labels, pred_labels, pos_label=1)
        self.logger.info('F1 Score = %.3f' % score) # F1 Score = 0.119
        
    def test(self, test_dataloader, label_list, checkpoint_filepath, ocsvm_filepath, visualization_dirpath):
   
        # Load the pretrained model
        epoch, self.model, self.optimizer, loss = self.load_checkpoint(checkpoint_filepath)
        if epoch == -1: return
    
        # Load the pretrained ocsvm model
        with open(ocsvm_filepath, 'rb') as f:
            self.ocsvm = pickle.load(f)
            self.logger.info(f'===> Load ocsvm model: {ocsvm_filepath}')

        # Evaluate model
        self.model.eval()
        video_ids = []
        video_features = []
        with torch.no_grad():
            
            # For each video
            for i, data in enumerate(test_dataloader):
                frame_ids = []
                frame_features = []

                # For each clips (1 frame)
                for f in range(len(data)):
                    imgs = data[f]
                    imgs = torch.Tensor(imgs).to(self.device)
                    frame_ids.append(f)
                    
                    fea = self.model.module.get_features(imgs)
                    # fea.shape = torch.Size([1, 512, 32, 32])
                    #print(f'fea.shape = {fea.shape}') # 
                    frame_features.append(torch.flatten(fea).cpu().numpy())
                
                video_ids.append(frame_ids)
                video_features.append(frame_features)
        
        assert len(video_features) == len(label_list), f'Ground truth has {len(label_list)} videos, BUT got {len(video_features)} detected videos!'
        
        pred_labels = []
        test_labels = []
        for i in range(len(video_features)):
            frame_features = video_features[i]
            
             # converting list to array
            frame_features = np.array(frame_features)
            print(f'frame_features.shape = {frame_features.shape}')
            
            # For a one-class model, +1 or -1 is returned.
            # self.ocsvm.predict: +1 (inlier=normal), -1: (outliers=anomaly)
            #y_pred = self.ocsvm.predict(frame_features) * (-1) # convert to +1 (anomaly), -1 (normal)
            
            # self.ocsvm.decision_function: Signed distance to the separating hyperplane.
            # Signed distance is positive for an inlier and negative for an outlier.
            y_pred = self.ocsvm.decision_function(frame_features).ravel() * (-1)
            # y_pred is positive for anomaly and negative for normal.
            pred_labels.append(np.array(y_pred))
            test_labels.append(np.array(label_list[i]))
        
        pred_labels = np.concatenate(pred_labels, axis = 0)
        self.logger.info(f'pred_labels.shape = {pred_labels.shape}')
        self.logger.info(f'pred_labels = {pred_labels}')
        # [ -44.64120591  -41.268254    -44.83123008 ... -146.71229831 -140.28474772 -142.66202044]
        
        test_labels = np.concatenate(test_labels, axis = 0)
        self.logger.info(f'test_labels.shape = {test_labels.shape}') # test_labels.shape = (2010,)
    
        # mark test_labels[i]==0 (normal) to -1, test_labels[i]==1 (anomaly) to 1
        test_labels[test_labels == 0] = -1
        self.logger.info(f'test_labels = {test_labels}') # test_labels = [-1 -1 -1 ...  1  1  1]
        #=============================
        # roc_auc_score: compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
        # https://github.com/hiram64/ocsvm-anomaly-detection/blob/master/anomaly_detection_ocsvm.py
        
        # Cach 1:
        precision, recall, _ = precision_recall_curve(test_labels, pred_labels, pos_label=1)
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
        # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
        roc_auc  = roc_auc_score(test_labels, pred_labels) # 
        
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
        # Compute Area Under the Curve (AUC) using the trapezoidal rule.
        prc_auc  = auc(recall, precision)
        self.logger.info(f'roc_auc  = {roc_auc}') # 0.44747023011317927
        self.logger.info(f'prc_auc  = {prc_auc}') # 0.7970437884929797
        
        '''
        # Cach 2:
        fpr , tpr , _ = roc_curve(test_labels , pred_labels , pos_label =1,  drop_intermediate=False)
        auc_score = auc(fpr, tpr)
        self.logger.info(f'auc_score = {auc_score}') # auc_score = 0.4474886686692056
        '''
    
    def test_1(self, test_dataloader, label_list, checkpoint_filepath, ocsvm_filepath, visualization_dirpath):
        
        # Load the pretrained model
        epoch, self.model, self.optimizer, loss = self.load_checkpoint(checkpoint_filepath)
        if epoch == -1: return
    
        # Load the pretrained ocsvm model
        with open(ocsvm_filepath, 'rb') as f:
            self.ocsvm = pickle.load(f)
            self.logger.info(f'===> Load ocsvm model: {ocsvm_filepath}')
        
        # Evaluate model
        self.model.eval()
        video_ids = []
        video_features = []
        with torch.no_grad():

            # For each video
            for i, data in enumerate(test_dataloader):
                frame_ids = []
                frame_features = []

                # For each clips (1 frame)
                for f in range(len(data)):
                    imgs = data[f]
                    imgs = torch.Tensor(imgs).to(self.device)
                    frame_ids.append(f)
                    
                    fea = self.model.module.get_features(imgs)
                    # fea.shape = torch.Size([1, 512, 32, 32])
                    #print(f'fea.shape = {fea.shape}') # 
                    frame_features.append(torch.flatten(fea).cpu().numpy())
                
                video_ids.append(frame_ids)
                video_features.append(frame_features)
        
        assert len(video_features) == len(label_list), f'Ground truth has {len(label_list)} videos, BUT got {len(video_features)} detected videos!'
        
        pred_labels = []
        test_labels = []
        for i in range(len(video_features)):
            frame_features = video_features[i]
            
             # converting list to array
            frame_features = np.array(frame_features)
            print(f'frame_features.shape = {frame_features.shape}')
            
            # For a one-class model, +1 or -1 is returned.
            # +1: anomaly, -1: normal
            # ocsvm.predict: return +1(inlier=normal) or -1 (outliers=anomaly)
            y_pred = self.ocsvm.predict(frame_features)
            pred_labels.append(np.array(y_pred)) * (-1)
            test_labels.append(np.array(label_list[i]))
        
        pred_labels = np.concatenate(pred_labels, axis = 0)
        self.logger.info(f'pred_labels.shape = {pred_labels.shape}')
        self.logger.info(f'pred_labels = {pred_labels}')
        
        test_labels = np.concatenate(test_labels, axis = 0)
        self.logger.info(f'test_labels.shape = {test_labels.shape}')
        
        # mark test_labels[i]==0 (normal) to -1, test_labels[i]==1 (anomaly) to 1
        test_labels[test_labels == 0] = -1
        self.logger.info(f'test_labels = {test_labels}')
        
        # roc_curve: Note: this implementation is restricted to the binary classification task
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
        fpr , tpr , _ = roc_curve(test_labels , pred_labels , pos_label =1,  drop_intermediate=False)
        auc_score = auc(fpr, tpr)

        self.logger.info(f'auc_score = {auc_score}')