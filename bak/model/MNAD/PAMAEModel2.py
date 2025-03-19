from model.MNAD.PAMAEModel import PAMAEModel
from model.MNAD.PAMAE_Pred import PAMAE_Pred
from model.MNAD.PAMAE_Recon import PAMAE_Recon

from model.util.loss import IntensityLoss
from model.util.optimizer import get_optimizer, get_scheduler
from model.util.process_data import decode_clip_cat, get_lists_dict_with_prob, point_score, make_dir_path

from model.util.metrics import psnr_park, calculate_auc_scores_MNAD, get_anomaly_rectanges
from model.util.auc import draw_auc_manual

from model.util.plot import export_gt_img_cv2, export_output_img_cv2, export_mse_img_cv2, plot_anomaly_scores

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


# =======================================
# NOTE: Model này đã implement, kế thừa: PAMAEModel
# Model: PAMAE_Pred
# Dataset gồm 2 phần: skip_frames, patch
# =======================================
class PAMAEModel2(PAMAEModel):
    def __init__(self, cfg, device, logger, writer):
        # ===========================================
        # Procedure for both of training and testing
        # ===========================================
        super().__init__(cfg, device, logger, writer)
        self.logger.info('Finish: PAMAEModel2.__init__()')

    def save_model(self, model_filepath):
        print('call: save_model() in PAMAEModel2')
        return super().save_model(model_filepath)

    def load_model(self, model_filepath):
        print('call: load_model() in PAMAEModel2')
        return super().load_model(model_filepath)

    def save_model_state_dict(self, model_filepath):
        print('call: save_model_state_dict() in PAMAEModel2')
        return super().save_model_state_dict(model_filepath)

    def load_model_state_dict(self, model_filepath):
        print('call: load_model_state_dict() in PAMAEModel2')
        return super().load_model_state_dict(model_filepath)

    def save_checkpoint(self, epoch, model, optimizer, loss, checkpoint_filepath):
        print('call: save_checkpoint() in PAMAEModel2')
        return super().save_checkpoint(epoch, model, optimizer, loss, checkpoint_filepath)

    def load_checkpoint(self, checkpoint_filepath):
        print('call: load_checkpoint() in PAMAEModel2')
        return super().load_checkpoint(checkpoint_filepath)

    def save_checkpoint(self, epoch, model, optimizer, loss, m_items, checkpoint_filepath):
        return super().save_checkpoint(epoch, model, optimizer, loss, m_items, checkpoint_filepath)

    def load_checkpoint(self, checkpoint_filepath):
        return super().load_checkpoint(checkpoint_filepath)

    # **********************************************************#
    # ************************** TRAINING **********************#
    # **********************************************************#
    # training in each epoch
    def train_epoch(self, train_dataloader, train_dataloader_skipframes, cifar100_dataloader):
        self.model.train()
        loss_epoch = 0
        cifar_iter = iter(cifar100_dataloader)
        # ==============================================================
        population = ['skipframes', 'patch', 'normal']
        skip_frames_prob = self.cfg.TRAIN.other_params.skip_frames_prob
        patch_prob = self.cfg.TRAIN.other_params.patch_prob
        weights = [skip_frames_prob, patch_prob, 1.0 - (skip_frames_prob + patch_prob)]  # weights=[0.01, 0.01, 0.98]
        num_elements = len(train_dataloader)
        retry = 0
        has_anomaly = False
        skipframes_ids_list = None
        patch_ids_list = None
        while not has_anomaly and retry < 10:
            retry += 1
            self.logger.info(f'get_lists_dict_with_prob(): retry = {retry}')
            lists_dict = get_lists_dict_with_prob(population, weights, num_elements)
            skipframes_ids_list = lists_dict.get('skipframes')
            patch_ids_list = lists_dict.get('patch')
            if ((skipframes_ids_list is not None) and
                    (patch_ids_list is not None) and
                    (len(skipframes_ids_list) > 0) and
                    (len(patch_ids_list) > 0)): has_anomaly = True

        self.logger.info(f'skipframes_ids_list = {skipframes_ids_list}')
        self.logger.info(f'patch_ids_list = {patch_ids_list}')
        # ==============================================================
        for j, (data, data_skipframes) in enumerate(zip(train_dataloader, train_dataloader_skipframes)):
            # self.logger.info(f'[{j+1}/{len(train_dataloader)}]')
            # self.logger.info(f'[{j+1}/{len(train_dataloader_skipframes)}]')
            imgs, target = decode_clip_cat(input=data,
                                           model_type=self.cfg.MODEL.type,
                                           num_frames=self.cfg.DATASET.num_frames,
                                           channel=self.cfg.DATASET.channel
                                           )
            imgs_skipframes, target_skipframes = decode_clip_cat(input=data_skipframes,
                                                                 model_type=self.cfg.MODEL.type,
                                                                 num_frames=self.cfg.DATASET.num_frames,
                                                                 channel=self.cfg.DATASET.channel)

            imgs = imgs.to(self.device)
            target = target.to(self.device)
            imgs_skipframes = imgs_skipframes.to(self.device)
            target_skipframes = target_skipframes.to(self.device)

            has_anomaly = False
            if (skipframes_ids_list is not None) and (j in skipframes_ids_list):
                has_anomaly = True
                imgs = imgs_skipframes
                target = target_skipframes
            if (patch_ids_list is not None) and (j in patch_ids_list):
                has_anomaly = True
                try:
                    # Samples the batch
                    cifar_img, _ = next(cifar_iter)
                except StopIteration:
                    # restart the generator if the previous generator is exhausted.
                    cifar_iter = iter(cifar100_dataloader)
                    cifar_img, _ = next(cifar_iter)
                cifar_img = cifar_img.to(self.device)
                self.logger.info(f'cifar_img.shape = {cifar_img.shape}')  # torch.Size([1, 3, 32, 32])
                self.logger.info(f'imgs.shape = {imgs.shape}')  # torch.Size([8, 12, 256, 256])

                imgs[0], mask = patch_cifar_smoothborder(imgs[0], cifar_img[0])
                target[0], mask = patch_cifar_smoothborder(target[0], cifar_img[0])
            # self.logger.info('imgs.shape = ', imgs.shape) #torch.Size([8, 12, 256, 256])
            # self.logger.info('target.shape = ', target.shape) #torch.Size([8, 3, 256, 256])

            output, _, _, self.m_items, softmax_score_query, softmax_score_memory, separateness_loss, compactness_loss = self.model.forward(
                x=imgs, keys=self.m_items, train=True)

            pixel_loss = torch.mean(self.loss_fn(output, target))
            if not has_anomaly:
                loss = pixel_loss + self.cfg.TRAIN.other_params.loss_compact * compactness_loss + self.cfg.TRAIN.other_params.loss_separate * separateness_loss
            else:
                loss = -pixel_loss - self.cfg.TRAIN.other_params.loss_compact * compactness_loss - self.cfg.TRAIN.other_params.loss_separate * separateness_loss
            loss = torch.mean(loss)  # with multiple gpus

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_epoch += loss
        loss_epoch = loss_epoch / len(train_dataloader)  # average the loss_epoch
        return loss_epoch

    # training over all epochs
    def train(self, train_dataloader, train_dataloader_skipframes, cifar100_dataloader, checkpoint_filepath,
              checkpoint_dir_path):
        # ======================================
        # Initalize some variables for training
        # ======================================
        self.logger.info(f'skip_frame_prob = {self.cfg.TRAIN.other_params.skip_frames_prob}')
        # =========================================
        train_time_start = time.time()
        loss = 0
        begin_epoch = self.cfg.TRAIN.begin_epoch
        end_epoch = self.cfg.TRAIN.end_epoch
        if self.cfg.TRAIN.resume:
            # ==================================
            self.logger.info(">>>>>>>> RESUME TRAINING: >>>>>>>>>>>")
            # ==================================
            epoch, self.model, self.optimizer, loss, self.m_items = self.load_checkpoint(checkpoint_filepath)
            if epoch == -1: return
            self.model = self.model.to(self.device)
            self.m_items = self.m_items.to(self.device)
            begin_epoch = epoch + 1
        # ==================================
        # Training process
        # ==================================
        for epoch in range(begin_epoch, end_epoch):
            self.logger.info(f'epoch[{epoch + 1}/{end_epoch}]')
            epoch_time_start = time.time()
            loss = self.train_epoch(train_dataloader, train_dataloader_skipframes, cifar100_dataloader)
            self.writer.add_scalar('training loss', loss, epoch)
            epoch_time_end = time.time()
            self.logger.info(f'>>>>>>> Epoch:{epoch + 1} takes time: {epoch_time_end - epoch_time_start} seconds')
            self.scheduler.step()
            # ==================================
            # Save the trained model by sequence
            # ==================================
            if (epoch + 1) % self.cfg.TRAIN.save_freq == 0:
                checkpoint_filepath = os.path.join(checkpoint_dir_path, f'epoch_{epoch + 1}.pth')
                self.save_checkpoint(epoch + 1, self.model, self.optimizer, loss, self.m_items, checkpoint_filepath)

        train_time_end = time.time()
        self.logger.info(f'===> Training takes time: {train_time_end - train_time_start} seconds')

        # ==================================
        # Save the final trained model
        # ==================================
        checkpoint_filepath = os.path.join(checkpoint_dir_path, 'final.pth')
        self.save_checkpoint(end_epoch, self.model, self.optimizer, loss, self.m_items, checkpoint_filepath)

        self.writer.flush()  # to make sure that all pending events have been written to disk.
        self.writer.close()  # If you do not need the summary writer anymore
        return 1
