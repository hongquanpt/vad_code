from model.MNAD.MNADModel import MNADModel
from model.util.process_data import decode_clip_cat
from dataset.data_types import dict_sf, dict_rf, dict_patch, dict_fusion, dict_noise, dict_shake
from dataset.aug import patch_SmoothMixS, apply_fusion, apply_gaussian_noise, apply_camera_shake
import random
import torch
import numpy as np
import copy
import torch.multiprocessing


class PBMAEModel(MNADModel):
    def __init__(self, cfg, device, logger, run, data_clip):
        super().__init__(cfg, device, logger, run, data_clip)

        self.data_sf = dict(dict_sf)
        self.data_rf = dict(dict_rf)
        self.data_patch = dict(dict_patch)
        self.data_fusion = dict(dict_fusion)
        self.data_noise = dict(dict_noise)
        self.data_shake = dict(dict_shake)

        # skip frames
        self.dataloader_sf = None
        self.iter_sf = None

        # repeat frames
        self.dataloader_rf = None
        self.iter_rf = None

        # patch
        self.dataloader_patch = None
        self.iter_patch = None

        # psnr for test
        self.psnr_types = cfg.TEST.psnr_types
        self.logger.info('Finish: PBMAEModel.__init__()')

    def save_model(self, model_filepath):
        print('call: save_model() in PBMAEModel')
        return super().save_model(model_filepath)

    def load_model(self, model_filepath):
        print('call: load_model() in PBMAEModel')
        return super().load_model(model_filepath)

    def save_model_state_dict(self, model_filepath):
        print('call: save_model_state_dict() in PBMAEModel')
        return super().save_model_state_dict(model_filepath)

    def load_model_state_dict(self, model_filepath):
        print('call: load_model_state_dict() in PBMAEModel')
        return super().load_model_state_dict(model_filepath)

    def save_checkpoint(self, checkpoint_filepath):
        print('call: save_checkpoint() in PBMAEModel')
        return super().save_checkpoint(checkpoint_filepath)

    def load_checkpoint(self, checkpoint_filepath):
        print('call: load_checkpoint() in PBMAEModel')
        return super().load_checkpoint(checkpoint_filepath)

    def set_aug_data(self, data_sf, data_rf, data_patch, data_fusion, data_noise, data_shake):
        self.data_sf = data_sf
        self.data_rf = data_rf
        self.data_patch = data_patch
        self.data_fusion = data_fusion
        self.data_noise = data_noise
        self.data_shake = data_shake

        # print(f"set_aug_data(): len(dataloader_sf) = {len(data_sf['dataloader'])}")
        self.dataloader_sf = data_sf['dataloader']
        self.dataloader_rf = data_rf['dataloader']
        self.dataloader_patch = data_patch['dataloader']

        if self.dataloader_sf is not None:
            self.iter_sf = iter(self.dataloader_sf)
        if self.dataloader_rf is not None:
            self.iter_rf = iter(self.dataloader_rf)
        if self.dataloader_patch is not None:
            self.iter_patch = iter(self.dataloader_patch)

    def get_data_sf(self):
        if self.dataloader_sf is None:
            # self.logger.info("Warning: self.dataloader_sf is None")
            return None
        try:
            # Samples the batch
            data_sf = next(self.iter_sf)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_sf = iter(self.dataloader_sf)
            data_sf = next(self.iter_sf)
        return data_sf

    def get_data_rf(self):
        if self.dataloader_rf is None:
            # self.logger.info("Warning: self.dataloader_rf is None")
            return None
        try:
            # Samples the batch
            data_rf = next(self.iter_rf)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_rf = iter(self.dataloader_rf)
            data_rf = next(self.iter_rf)
        return data_rf

    def get_data_patch(self):
        if self.dataloader_patch is None:
            # self.logger.info("Warning: self.dataloader_patch is None")
            return None
        try:
            # Samples the batch
            data_patch, _ = next(self.iter_patch)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_patch = iter(self.dataloader_patch)
            data_patch, _ = next(self.iter_patch)
        return data_patch, _

    def train(self, train_dataloader, checkpoint_filepath, checkpoint_dir_path):
        self.logger.info('call: train() in PBMAEModel')
        return super().train(train_dataloader, checkpoint_filepath, checkpoint_dir_path)

    def train_epoch(self, epoch, train_dataloader):
        self.logger.info('====>>>>> MNAD call: train_epoch()')
        self.logger.info(f'call: {self.cfg.MODEL.method}')
        # if self.cfg.MODEL.method == 'PBMAE':  # ALL: sf, kf, patch, fusion, noise
        return self.PBMAE(epoch, train_dataloader)

    def PBMAE(self, epoch, train_dataloader):
        self.epoch = epoch + 1
        self.m_items = self.m_items.to(self.device)

        bz = self.cfg.DATASET.train.batch_size_per_gpu
        self.logger.info(f'bz = {bz}')

        # Pseudo Anomalies
        p_sf = float(self.data_sf['p'])
        p_rf = float(self.data_rf['p'])
        p_patch = float(self.data_patch['p'])
        p_fusion = float(self.data_fusion['p'])
        p_noise = float(self.data_noise['p'])
        p_shake = float(self.data_shake['p'])

        sum_loss_epoch = 0.0
        pseudo_loss_epoch = 0.0
        pseudo_loss_epoch_counter = 0
        normal_loss_epoch = 0.0
        normal_loss_epoch_counter = 0
        for j, data in enumerate(train_dataloader):
            # data.shape = torch.Size([4, 15, 256, 256])
            # self.logger.info(f'data.shape = {data.shape}')
            # data[b] (15,256,256) => data[b].unsqueeze(0) (1,15,256,256)

            state_pseudo_sf = []
            state_pseudo_rf = []
            state_pseudo_patch = []
            state_pseudo_fusion = []
            state_pseudo_noise = []
            state_pseudo_shake = []

            data_sf = self.get_data_sf()
            data_rf = self.get_data_rf()
            data_patch, _ = self.get_data_patch()

            for b in range(bz):
                sum_p = 0
                # ===========================================
                # skip frame
                # ===========================================
                # random samples from a uniform distribution over [0, 1).
                rand_number = np.random.rand()
                is_pseudo = sum_p <= rand_number < sum_p + p_sf
                sum_p += p_sf
                state_pseudo_sf.append(is_pseudo)
                if is_pseudo:
                    # self.logger.info(f'skip frames (j={j},b={b}): data[b].shape = {data[b].shape}')
                    data[b] = data_sf[0]  # replace a normal sample by a pseudo abnormal sample

                # ===========================================
                # repeat frame
                # ===========================================
                is_pseudo = sum_p <= rand_number < sum_p + p_rf
                sum_p += p_rf
                state_pseudo_rf.append(is_pseudo)
                if is_pseudo:
                    # self.logger.info(f'repeat frames (j={j},b={b}): data[b].shape = {data[b].shape}')
                    data[b] = data_rf[0]  # replace a normal sample by a pseudo abnormal sample

                # ===========================================
                # patch
                # ===========================================
                is_pseudo = sum_p <= rand_number < sum_p + p_patch
                sum_p += p_patch
                state_pseudo_patch.append(is_pseudo)
                if is_pseudo:
                    if self.data_patch['technique'] == 'SmoothMixS':
                        # self.logger.info(f'patch (j={j},b={b}): data[b].shape = {data[b].shape}')
                        # data[b] (15,256,256) => data[b].unsqueeze(0) (1,15,256,256)
                        img_patch, mask = patch_SmoothMixS(data[b].unsqueeze(0),
                                                           data_patch.squeeze(0),
                                                           max_size=self.data_patch['alpha'],
                                                           c=3,
                                                           h=self.cfg.DATASET.width,
                                                           w=self.cfg.DATASET.height,
                                                           max_move=self.data_patch['beta'])
                        # img_patch (1,15,256,256)
                        data[b] = img_patch.squeeze(0)

                # ===========================================
                # fusion
                # ===========================================
                is_pseudo = sum_p <= rand_number < sum_p + p_fusion
                sum_p += p_fusion
                state_pseudo_fusion.append(is_pseudo)
                if is_pseudo:
                    # self.logger.info(f'fusion (j={j},b={b}): data[b].shape = {data[b].shape}')
                    if b < (bz - 1):
                        data[b] = apply_fusion(data[b], data[b + 1])
                    elif b == (bz - 1):
                        data[b] = apply_fusion(data[b], data[b - 1])

                # ===========================================
                # noise
                # ===========================================
                is_pseudo = sum_p <= rand_number < sum_p + p_noise
                sum_p += p_noise
                state_pseudo_noise.append(is_pseudo)
                if is_pseudo:
                    # self.logger.info(f'noise (j={j},b={b}): data[b].shape = {data[b].shape}')
                    data[b] = apply_gaussian_noise(data[b], mean=0.0, std=self.data_noise['sigma'])

                # ===========================================
                # camera shake
                # ===========================================
                is_pseudo = sum_p <= rand_number < sum_p + p_shake
                sum_p += p_shake
                state_pseudo_shake.append(is_pseudo)
                if is_pseudo:
                    # self.logger.info(f'shake (j={j},b={b}): data[b].shape = {data[b].shape}')
                    data[b] = apply_camera_shake(clip=data[b],
                                                 c=3,
                                                 rotation_range=self.data_shake['rot'],
                                                 translation_range=self.data_shake['trans'])

            # PREDICTION
            inputs, target = decode_clip_cat(data=data.to(self.device),
                                             num_frames=self.cfg.DATASET.num_frames)
            output, _, _, self.m_items, softmax_score_query, softmax_score_memory, separate_loss, compact_loss = self.model.forward(
                x=inputs, keys=self.m_items, train=True)

            pixel_loss = self.loss_fn(output, target)
            loss_compact = self.cfg.TRAIN.other_params.loss_compact * compact_loss
            loss_separate = self.cfg.TRAIN.other_params.loss_separate * separate_loss
            modified_loss_mse = []
            for b in range(bz):
                if state_pseudo_sf[b] or state_pseudo_rf[b] or state_pseudo_patch[b] \
                        or state_pseudo_fusion[b] or state_pseudo_noise[b]:
                    b_loss = torch.mean(-pixel_loss[b])
                    modified_loss_mse.append(b_loss)
                    pseudo_loss_epoch += b_loss.cpu().detach().item()
                    pseudo_loss_epoch_counter += 1
                elif state_pseudo_shake[b]:
                    b_loss = torch.mean(pixel_loss[b])
                    modified_loss_mse.append(b_loss)
                    normal_loss_epoch += b_loss.cpu().detach().item()
                    normal_loss_epoch_counter += 1
                else:  # normal
                    b_loss = torch.mean(pixel_loss[b])
                    modified_loss_mse.append(b_loss)
                    normal_loss_epoch += b_loss.cpu().detach().item()
                    normal_loss_epoch_counter += 1

            stacked_loss_mse = torch.stack(modified_loss_mse)
            loss = torch.mean(stacked_loss_mse) + loss_compact + loss_separate

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            sum_loss_epoch += loss

        # average the loss_epoch
        self.loss_epoch = sum_loss_epoch / len(train_dataloader)
        self.logger.info('Loss Epoch {:.9f}'.format(self.loss_epoch))
        loss_pseudo = 0
        loss_normal = 0
        if pseudo_loss_epoch_counter != 0:
            loss_pseudo = pseudo_loss_epoch / pseudo_loss_epoch_counter
            self.logger.info('Loss Pseudo: {:.9f}'.format(loss_pseudo))
        if normal_loss_epoch_counter != 0:
            loss_normal = normal_loss_epoch / normal_loss_epoch_counter
            self.logger.info('Loss Normal: {:.9f}'.format(loss_normal))

        # save to neptune
        if self.run is not None:
            self.run["train/loss_epoch"].log(self.loss_epoch)
            self.run["train/loss_pseudo"].log(loss_pseudo)
            self.run["train/loss_normal"].log(loss_normal)
        return self.loss_epoch

    def test(self, test_dataloader, video_labels, checkpoint_filepath, visualization_dir_path):
        self.logger.info('call: test() in PBMAEModel')
        return super().test(test_dataloader, video_labels, checkpoint_filepath, visualization_dir_path)
