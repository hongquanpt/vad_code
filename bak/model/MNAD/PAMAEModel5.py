from .PAMAEModel4 import PAMAEModel4
from model.util.process_data import decode_clip_cat

import torch
import numpy as np

import time
import os

import torch.multiprocessing


# =======================================
# NOTE: Model này đã implement
# =======================================
class PAMAEModel5(PAMAEModel4):
    def __init__(self, cfg, device, logger, writer, run):
        # ===========================================
        # Procedure for both of training and testing
        # ===========================================
        super().__init__(cfg, device, logger, writer, run)
        self.train_dataloader_kf = None
        self.iter_kf = None
        self.logger.info('Finish: PAMAEModel5.__init__()')

    def save_model(self, model_filepath):
        print('call: save_model() in PAMAEModel5')
        return super().save_model(model_filepath)

    def load_model(self, model_filepath):
        print('call: load_model() in PAMAEModel5')
        return super().load_model(model_filepath)

    def save_model_state_dict(self, model_filepath):
        print('call: save_model_state_dict() in PAMAEModel5')
        return super().save_model_state_dict(model_filepath)

    def load_model_state_dict(self, model_filepath):
        print('call: load_model_state_dict() in PAMAEModel5')
        return super().load_model_state_dict(model_filepath)

    def save_checkpoint(self, checkpoint_filepath):
        print('call: save_checkpoint() in PAMAEModel5')
        return super().save_checkpoint(checkpoint_filepath)

    def load_checkpoint(self, checkpoint_filepath):
        print('call: load_checkpoint() in PAMAEModel5')
        return super().load_checkpoint(checkpoint_filepath)

    def set_train_dataloader_kf(self, train_dataloader_kf):
        self.train_dataloader_kf = train_dataloader_kf

    def get_data_kf(self):
        try:
            # Samples the batch
            data_kf = next(self.iter_kf)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_kf = iter(self.train_dataloader_kf)
            data_kf = next(self.iter_kf)
        return data_kf

    # training in each epoch
    def train_epoch(self, epoch, train_dataloader):
        if self.train_dataloader_kf is None:
            self.logger.info("Warning: self.train_dataloader_kf is None")
            return
        self.epoch = epoch + 1
        self.m_items = self.m_items.to(self.device)
        sf_prob = self.cfg.TRAIN.other_params.skip_frames_prob
        self.logger.info(f'skip_frames_prob = {sf_prob}') 
        bz = self.cfg.DATASET.train.batch_size_per_gpu
        
        sum_loss_epoch = 0.0
        pseudo_lossepoch = 0
        pseudo_lossepoch_counter = 0
        normal_lossepoch = 0
        normal_lossepoch_counter = 0
        self.iter_kf = iter(self.train_dataloader_kf)
        for j, data in enumerate(train_dataloader):
            # data.shape = torch.Size([4, 15, 256, 256])
            # self.logger.info(f'data.shape = {data.shape}') 
            is_pseudo_labels = []
            data_kf = self.get_data_kf()
            for b in range(bz):
                #  random samples from a uniform distribution over [0, 1).
                rand_number = np.random.rand()
                # skip frame pseudo anomaly
                is_pseudo = 0 <= rand_number < sf_prob
                is_pseudo_labels.append(is_pseudo)
                if is_pseudo:
                    data[b] = data_kf[b]

            inputs, target = decode_clip_cat(data=data.to(self.device),
                                             model_type=self.cfg.MODEL.type,
                                             num_frames=self.cfg.DATASET.num_frames)
            # output = self.model(inputs)
            output, _, _, self.m_items, softmax_score_query, softmax_score_memory, separate_loss, compact_loss = self.model.forward(
                x=inputs, keys=self.m_items, train=True)

            # self.logger.info(f'separate_loss = {separate_loss}')
            # self.logger.info(f'compact_loss = {compact_loss}')
            pixel_loss = self.loss_fn(output, target)
            loss_compact = self.cfg.TRAIN.other_params.loss_compact * compact_loss
            loss_separate = self.cfg.TRAIN.other_params.loss_separate * separate_loss
            modified_loss_mse = []
            for b in range(bz):
                if is_pseudo_labels[b]:
                    b_loss = torch.mean(-pixel_loss[b]) - loss_compact - loss_separate
                    modified_loss_mse.append(b_loss)
                    pseudo_lossepoch += b_loss.cpu().detach().item()
                    pseudo_lossepoch_counter += 1
                else:
                    b_loss = torch.mean(pixel_loss[b]) + loss_compact + loss_separate
                    modified_loss_mse.append(b_loss)
                    normal_lossepoch += b_loss.cpu().detach().item()
                    normal_lossepoch_counter += 1

            stacked_loss_mse = torch.stack(modified_loss_mse)
            loss = torch.mean(stacked_loss_mse)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            sum_loss_epoch += loss

        # average the loss_epoch
        self.loss_epoch = sum_loss_epoch / len(train_dataloader)
        self.logger.info('Loss Epoch {:.9f}'.format(self.loss_epoch))
        loss_pseudo = 0
        loss_normal = 0
        if pseudo_lossepoch_counter != 0:
            loss_pseudo = pseudo_lossepoch / pseudo_lossepoch_counter
            self.logger.info('Loss Pseudo: {:.9f}'.format(loss_pseudo))
        if normal_lossepoch_counter != 0:
            loss_normal = normal_lossepoch / normal_lossepoch_counter
            self.logger.info('Loss Normal: {:.9f}'.format(loss_normal))
        
        # save to neptune
        if self.run is not None:
            self.run["train/loss_epoch"].log(self.loss_epoch)
            self.run["train/loss_pseudo"].log(loss_pseudo)
            self.run["train/loss_normal"].log(loss_normal)
        return self.loss_epoch

    def train(self, train_dataloader, checkpoint_filepath, checkpoint_dir_path):
        self.logger.info('call: train() in PAMAE5Model')
        return super().train(train_dataloader, checkpoint_filepath, checkpoint_dir_path)

