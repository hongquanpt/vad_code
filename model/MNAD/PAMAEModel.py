from model.MNAD.MNADModel import MNADModel
from .PAMAE_Pred import PAMAE_Pred
from .PAMAE_Recon import PAMAE_Recon

from model.util.optimizer import get_optimizer, get_scheduler
from model.util.process_data import decode_clip_cat
from model.util.process_data import decode_clip_cat_last_input

from dataset.aug import patch_SmoothMixS, apply_camera_shake
from dataset.process import get_data_sf, get_data_kf, get_data_patch, get_data_shake, get_data_obj
from dataset.util import get_sample_sf, get_sample_kf, get_sample_patch, get_sample_obj
from model.util.loss import SF_MultiLoss, SF_KF_MultiLoss

import torch
import numpy as np
import torch.multiprocessing
from tqdm import tqdm


class PAMAEModel(MNADModel):
    def __init__(self, cfg, device, logger, run, data_clip):
        super().__init__(cfg, device, logger, run, data_clip)
        # ===================================================
        self.data_sf = None
        self.iter_sf = None

        self.data_kf = None
        self.iter_kf = None

        self.data_obj = None
        self.iter_obj = None

        self.data_patch = None
        self.iter_patch = None

        self.data_shake = None
        if self.cfg.MODEL.name == 'PAMAE':
            if self.cfg.MODEL.type == 'prediction':
                self.logger.info('====>>>>> call MODEL: PAMAE_Model prediction')
                self.model = PAMAE_Pred(cfg.DATASET.channel, cfg.DATASET.num_frames,
                                        cfg.TRAIN.other_params.memory_size,
                                        cfg.TRAIN.other_params.feature_dim,
                                        cfg.TRAIN.other_params.memory_dim)
            elif self.cfg.MODEL.type == 'reconstruction':
                self.logger.info('====>>>>> call MODEL: PAMAE_Model reconstruction')
                self.model = PAMAE_Recon(cfg.DATASET.channel, cfg.DATASET.num_frames,
                                         cfg.TRAIN.other_params.memory_size,
                                         cfg.TRAIN.other_params.feature_dim,
                                         cfg.TRAIN.other_params.memory_dim)
            else:
                self.logger.info('Warning: cfg.model_type must be prediction or reconstruction!')

        # ===================================================
        loss_mapping = {
            "SF_MultiLoss": SF_MultiLoss(self.cfg.DATASET.channel, self.device),
            "SF_KF_MultiLoss": SF_KF_MultiLoss(self.cfg.DATASET.channel, self.device)
        }
        self.loss_fn = loss_mapping.get(self.cfg.MODEL.method, torch.nn.MSELoss(reduction='none'))
        if self.model is not None:
            self.model = self.model.to(self.device)
            self.optimizer = get_optimizer(self.cfg, self.model)
            self.scheduler = get_scheduler(self.cfg, self.optimizer)
        self.setup_aug_data()
        self.psnr_types = cfg.TEST.psnr_types
        # ===================================================
        self.logger.info('Finish: PAMAEModel.__init__()')

    def save_model(self, model_filepath):
        print('call: save_model() in PAMAEModel')
        return super().save_model(model_filepath)

    def load_model(self, model_filepath):
        print('call: load_model() in PAMAEModel')
        return super().load_model(model_filepath)

    def save_model_state_dict(self, model_filepath):
        print('call: save_model_state_dict() in PAMAEModel')
        return super().save_model_state_dict(model_filepath)

    def load_model_state_dict(self, model_filepath):
        print('call: load_model_state_dict() in PAMAEModel')
        return super().load_model_state_dict(model_filepath)

    def save_checkpoint(self, checkpoint_filepath):
        print('call: save_checkpoint() in PAMAEModel')
        return super().save_checkpoint(checkpoint_filepath)

    def load_checkpoint(self, checkpoint_filepath, is_load_scheduler=True):
        print('call: load_checkpoint() in PAMAEModel')
        return super().load_checkpoint(checkpoint_filepath, is_load_scheduler)

    def transfer_scheduler_checkpoint(self, checkpoint_filepath_source, checkpoint_filepath_destination):
        print('call: transfer_scheduler_checkpoint() in PAMAEModel')
        return super().transfer_scheduler_checkpoint(checkpoint_filepath_source, checkpoint_filepath_destination)

    def train(self, train_dataloader, checkpoint_filepath, checkpoint_dir_path):
        self.logger.info(f'====>>>>> call: train() of {self.cfg.MODEL.method}')
        return super().train(train_dataloader, checkpoint_filepath, checkpoint_dir_path)

    def train_epoch(self, epoch, train_dataloader):
        self.logger.info(f'====>>>>> call: train_epoch() of {self.cfg.MODEL.method}')

        # PAMAE_SF_KF_PA
        if "_PA" in self.cfg.MODEL.method:
            return self.SF_KF_PA(epoch, train_dataloader)
        if "_OBJ" in self.cfg.MODEL.method:
            return self.OBJ(epoch, train_dataloader)
        # PAMAE_KF_ES, PAMAE_KF_EM, PAMAE_KF_EN, PAMAE_SF_KF, PAMAE_SF
        elif "_KF" in self.cfg.MODEL.method or "_SF" in self.cfg.MODEL.method:
            return self.SF_KF(epoch, train_dataloader)

    def setup_aug_data(self):
        if 'dict_sf' in self.cfg.DATASET:
            self.data_sf = get_data_sf(self.cfg)
        if 'dict_kf' in self.cfg.DATASET:
            self.data_kf = get_data_kf(self.cfg)
        if 'dict_obj' in self.cfg.DATASET:
            self.data_obj = get_data_obj(self.cfg)
        if 'dict_patch' in self.cfg.DATASET:
            self.data_patch = get_data_patch(self.cfg)

        if self.data_sf is not None and self.data_sf['dataloader'] is not None:
            self.iter_sf = iter(self.data_sf['dataloader'])
        if self.data_kf is not None and self.data_kf['dataloader'] is not None:
            self.iter_kf = iter(self.data_kf['dataloader'])
        if self.data_obj is not None and self.data_obj['dataloader'] is not None:
            self.iter_obj = iter(self.data_obj['dataloader'])
        if self.data_patch is not None and self.data_patch['dataloader'] is not None:
            self.iter_patch = iter(self.data_patch['dataloader'])

    def calculate_loss_sf_multi(self, clip, is_pseudo=False):
        # inputs, target, input_last = decode_clip_cat_last_input(data=clip.to(self.device), num_frames=self.cfg.DATASET.num_frames)
        a = (self.cfg.DATASET.num_frames - 2) * self.cfg.DATASET.channel  # i.e, (5-2)*3 = 9
        b = (self.cfg.DATASET.num_frames - 1) * self.cfg.DATASET.channel  # i.e, (5-1)*3 = 12
        input_last = clip[:, a:b].to(self.device)
        inputs = clip[:, :b].to(self.device)  # torch.Size([bz, 12, 256, 256])
        target = clip[:, b:].to(self.device)  # torch.Size([bz, 3, 256, 256])
        output, self.m_items, separate_loss, compact_loss = self.model(x=inputs, keys=self.m_items,
                                                                       train=True, is_pseudo=is_pseudo)

        intensity_loss, grad_loss, msssim, l2_loss, flow_loss = self.loss_fn(input_last, output, target)
        self.logger.info(f'intensity_loss.shape = {intensity_loss.shape}')
        self.logger.info(f'grad_loss.shape = {grad_loss.shape}')
        # self.logger.info(f'msssim.shape = {msssim.shape}')
        self.logger.info(f'l2_loss.shape = {l2_loss.shape}')
        self.logger.info(f'flow_loss.shape = {flow_loss.shape}')
        pixel_loss = intensity_loss + grad_loss + l2_loss + flow_loss
        self.logger.info(f'pixel_loss.shape = {pixel_loss.shape}')

        # del output, inputs, target, input_last
        return pixel_loss, separate_loss, compact_loss

    def calculate_loss_sf_kf_multi(self, clip, is_pseudo=False):
        # inputs, target = decode_clip_cat(data=clip.to(self.device), num_frames=self.cfg.DATASET.num_frames)
        seq_len = (self.cfg.DATASET.num_frames - 1) * self.cfg.DATASET.channel
        inputs = clip[:, :seq_len].to(self.device)
        target = clip[:, seq_len:].to(self.device)
        output, self.m_items, separate_loss, compact_loss = self.model(x=inputs, keys=self.m_items,
                                                                       train=True, is_pseudo=is_pseudo)

        # Compute the loss separately on each GPU
        intensity_loss, grad_loss, msssim, l2_loss = self.loss_fn(output, target)
        pixel_loss = intensity_loss + grad_loss + msssim + l2_loss

        # Properly manage tensors by deleting them when they are no longer needed
        # del output, inputs, target
        return pixel_loss, separate_loss, compact_loss

    def calculate_loss_default(self, clip, is_pseudo=False):
        # inputs, target = decode_clip_cat(data=clip.to(self.device), num_frames=self.cfg.DATASET.num_frames)
        seq_len = (self.cfg.DATASET.num_frames - 1) * self.cfg.DATASET.channel
        inputs = clip[:, :seq_len].to(self.device)
        target = clip[:, seq_len:].to(self.device)

        output, self.m_items, separate_loss, compact_loss = self.model(x=inputs, keys=self.m_items,
                                                                       train=True, is_pseudo=is_pseudo)

        # Compute the loss separately on each GPU
        pixel_loss = self.loss_fn(output, target)

        # Properly manage tensors by deleting them when they are no longer needed
        del output, inputs, target, clip
        return pixel_loss, separate_loss, compact_loss

    """def calculate_loss_OBJ(self, clip):
        bz = self.cfg.DATASET.train.batch_size_per_gpu
        p_obj = float(self.data_obj['p'])

        data_obj = None
        modified_loss_mse = []
        for b in range(bz):
            sum_p = 0
            #  random samples from a uniform distribution over [0, 1).
            rand_number = np.random.rand()
            # Object-level
            is_pseudo_obj = sum_p <= rand_number < sum_p + p_obj
            sum_p += p_obj
            # state_pseudo_sf.append(is_pseudo_sf)
            if is_pseudo_obj:
                # print(f'is_pseudo_obj = {is_pseudo_obj} with b = {b}')
                # self.logger.info(f'object-level (j={j},b={b}): data[b].shape = {clip[b].shape}')
                data_obj = get_sample_obj(self.data_obj['dataloader'], self.iter_obj, self.data_clip)
                clip[b] = data_obj[0]  # replace a normal sample by a pseudo abnormal sample

            is_pseudo = is_pseudo_obj

            if "SF_MultiLoss" in self.cfg.MODEL.method:
                pixel_loss, separate_loss, compact_loss = self.calculate_loss_sf_multi(clip[b].unsqueeze(0), is_pseudo)
            elif "SF_KF_MultiLoss" in self.cfg.MODEL.method:
                pixel_loss, separate_loss, compact_loss = self.calculate_loss_sf_kf_multi(clip[b].unsqueeze(0),
                                                                                          is_pseudo)
            else:
                pixel_loss, separate_loss, compact_loss = self.calculate_loss_default(clip[b].unsqueeze(0), is_pseudo)

            # pixel_loss: tensor(1,3,256,256); compact_loss: tensor(1,1024,512); separate_loss: tensor(1,1024)
            compact_loss = self.cfg.TRAIN.other_params.loss_compact * compact_loss
            separate_loss = self.cfg.TRAIN.other_params.loss_separate * separate_loss

            mean_loss_pixel = torch.mean(pixel_loss)
            mean_loss_compact = torch.mean(compact_loss)
            mean_loss_separate = torch.mean(separate_loss)

            b_loss = 0.0
            if is_pseudo:
                if 'loss_type' in self.cfg.TRAIN.other_params:
                    if self.cfg.TRAIN.other_params.loss_type == 'IC':
                        b_loss = -mean_loss_pixel - mean_loss_compact
                    else:  # 'ICS'
                        b_loss = -mean_loss_pixel - mean_loss_compact - mean_loss_separate
                else:
                    b_loss = -mean_loss_pixel - mean_loss_compact
            else:  # normal
                b_loss = mean_loss_pixel + mean_loss_compact + mean_loss_separate
            modified_loss_mse.append(b_loss)

        stacked_loss_mse = torch.stack(modified_loss_mse)
        loss = torch.mean(stacked_loss_mse)

        # Explicitly delete tensors to release memory
        del clip, data_obj  # , state_pseudo_sf, state_pseudo_kf
        return loss
"""
#Không cộng dồn tỷ lệ
    def calculate_loss_OBJ(self, clip):
        bz = self.cfg.DATASET.train.batch_size_per_gpu
        p_obj = float(self.data_obj['p'])

        data_obj = None
        modified_loss_mse = []
        for b in range(bz):
            # Generate a random number between 0 and 1
            rand_number = np.random.rand()

            # Determine if a pseudo abnormal sample should replace the normal sample
            is_pseudo_obj = rand_number < p_obj

            if is_pseudo_obj:
                # Replace the normal sample with a pseudo abnormal sample
                data_obj = get_sample_obj(self.data_obj['dataloader'], self.iter_obj, self.data_clip)
                clip[b] = data_obj[0]  # Replace the normal sample

            is_pseudo = is_pseudo_obj

            if "SF_MultiLoss" in self.cfg.MODEL.method:
                pixel_loss, separate_loss, compact_loss = self.calculate_loss_sf_multi(clip[b].unsqueeze(0), is_pseudo)
            elif "SF_KF_MultiLoss" in self.cfg.MODEL.method:
                pixel_loss, separate_loss, compact_loss = self.calculate_loss_sf_kf_multi(clip[b].unsqueeze(0), is_pseudo)
            else:
                pixel_loss, separate_loss, compact_loss = self.calculate_loss_default(clip[b].unsqueeze(0), is_pseudo)

            # Adjust the losses
            compact_loss = self.cfg.TRAIN.other_params.loss_compact * compact_loss
            separate_loss = self.cfg.TRAIN.other_params.loss_separate * separate_loss

            mean_loss_pixel = torch.mean(pixel_loss)
            mean_loss_compact = torch.mean(compact_loss)
            mean_loss_separate = torch.mean(separate_loss)

            if is_pseudo:
                if 'loss_type' in self.cfg.TRAIN.other_params:
                    if self.cfg.TRAIN.other_params.loss_type == 'IC':
                        b_loss = -mean_loss_pixel - mean_loss_compact
                    else:  # 'ICS'
                        b_loss = -mean_loss_pixel - mean_loss_compact - mean_loss_separate
                else:
                    b_loss = -mean_loss_pixel - mean_loss_compact
            else:  # Normal sample
                b_loss = mean_loss_pixel + mean_loss_compact + mean_loss_separate

            modified_loss_mse.append(b_loss)

        stacked_loss_mse = torch.stack(modified_loss_mse)
        loss = torch.mean(stacked_loss_mse)

        # Explicitly delete tensors to release memory
        del clip, data_obj
        return loss

    def calculate_loss_SF_KF(self, clip):
        bz = self.cfg.DATASET.train.batch_size_per_gpu
        p_sf = float(self.data_sf['p'])
        p_kf = float(self.data_kf['p'])
        data_sf = None
        data_kf = None

        modified_loss_mse = []
        for b in range(bz):
            sum_p = 0
            #  random samples from a uniform distribution over [0, 1).
            rand_number = np.random.rand()
            # Skip Frames
            is_pseudo_sf = sum_p <= rand_number < sum_p + p_sf
            sum_p += p_sf
            # state_pseudo_sf.append(is_pseudo_sf)
            if is_pseudo_sf:
                # print(f'is_pseudo_sf = {is_pseudo_sf} with b = {b}')
                # self.logger.info(f'skip frames (j={j},b={b}): data[b].shape = {clip[b].shape}')
                data_sf = get_sample_sf(self.data_sf['dataloader'], self.iter_sf, self.data_clip)
                clip[b] = data_sf[0]  # replace a normal sample by a pseudo abnormal sample

            # Key Frames
            is_pseudo_kf = sum_p <= rand_number < sum_p + p_kf
            sum_p += p_kf
            # state_pseudo_kf.append(is_pseudo_kf)
            if is_pseudo_kf:
                # print(f'is_pseudo_kf = {is_pseudo_kf} with b = {b}')
                # self.logger.info(f'key frames (j={j},b={b}): data[b].shape = {clip[b].shape}')
                data_kf = get_sample_kf(self.data_kf['dataloader'], self.iter_kf, self.data_clip)
                clip[b] = data_kf[0]  # replace a normal sample by a pseudo abnormal sample

            is_pseudo = is_pseudo_sf or is_pseudo_kf

            if "SF_MultiLoss" in self.cfg.MODEL.method:
                pixel_loss, separate_loss, compact_loss = self.calculate_loss_sf_multi(clip[b].unsqueeze(0), is_pseudo)
            elif "SF_KF_MultiLoss" in self.cfg.MODEL.method:
                pixel_loss, separate_loss, compact_loss = self.calculate_loss_sf_kf_multi(clip[b].unsqueeze(0), is_pseudo)
            else:
                pixel_loss, separate_loss, compact_loss = self.calculate_loss_default(clip[b].unsqueeze(0), is_pseudo)

            # pixel_loss: tensor(1,3,256,256); compact_loss: tensor(1,1024,512); separate_loss: tensor(1,1024)
            compact_loss = self.cfg.TRAIN.other_params.loss_compact * compact_loss
            separate_loss = self.cfg.TRAIN.other_params.loss_separate * separate_loss

            mean_loss_pixel = torch.mean(pixel_loss)
            mean_loss_compact = torch.mean(compact_loss)
            mean_loss_separate = torch.mean(separate_loss)

            b_loss = 0.0
            if is_pseudo_sf or is_pseudo_kf:
                if 'loss_type' in self.cfg.TRAIN.other_params:
                    if self.cfg.TRAIN.other_params.loss_type == 'IC':
                        b_loss = -mean_loss_pixel - mean_loss_compact
                    else:  # 'ICS'
                        b_loss = -mean_loss_pixel - mean_loss_compact - mean_loss_separate
                else:
                    b_loss = -mean_loss_pixel - mean_loss_compact
            else:  # normal
                b_loss = mean_loss_pixel + mean_loss_compact + mean_loss_separate
            modified_loss_mse.append(b_loss)

        stacked_loss_mse = torch.stack(modified_loss_mse)
        loss = torch.mean(stacked_loss_mse)

        # Explicitly delete tensors to release memory
        del clip, data_sf, data_kf
        return loss

    def calculate_loss_SF_KF_PA(self, clip):
        """
        SF_SK_PA: Skip Frames + Key Frames + Patch
        """
        bz = self.cfg.DATASET.train.batch_size_per_gpu
        p_sf = float(self.data_sf['p'])
        p_kf = float(self.data_kf['p'])
        p_patch = float(self.data_patch['p'])
        data_sf = None
        data_kf = None
        data_patch = None
        state_pseudo_sf = []
        state_pseudo_kf = []
        state_pseudo_patch = []

        for b in range(bz):
            sum_p = 0
            #  random samples from a uniform distribution over [0, 1).
            rand_number = np.random.rand()
            # Skip Frames
            is_pseudo = sum_p <= rand_number < sum_p + p_sf
            sum_p += p_sf
            state_pseudo_sf.append(is_pseudo)
            if is_pseudo:
                # self.logger.info(f'skip frames (j={j},b={b}): data[b].shape = {clip[b].shape}')
                data_sf = get_sample_sf(self.data_sf['dataloader'], self.iter_sf, self.data_clip)
                clip[b] = data_sf[0]  # replace a normal sample by a pseudo abnormal sample

            # Key Frames
            is_pseudo = sum_p <= rand_number < sum_p + p_kf
            sum_p += p_kf
            state_pseudo_kf.append(is_pseudo)
            if is_pseudo:
                # self.logger.info(f'key frames (j={j},b={b}): data[b].shape = {clip[b].shape}')
                data_kf = get_sample_kf(self.data_kf['dataloader'], self.iter_kf, self.data_clip)
                clip[b] = data_kf[0]  # replace a normal sample by a pseudo abnormal sample

            # Patch
            is_pseudo = sum_p <= rand_number < sum_p + p_patch
            sum_p += p_patch
            state_pseudo_patch.append(is_pseudo)
            if is_pseudo and self.data_patch['technique'] == 'SmoothMixS':
                # clip[b] (15,256,256) => clip[b].unsqueeze(0) (1,15,256,256)
                data_patch, _ = get_sample_patch(self.data_patch['dataloader'], self.iter_patch)
                img_patch, mask = patch_SmoothMixS(clip[b].unsqueeze(0),
                                                   data_patch[0],
                                                   max_size=self.data_patch['alpha'],
                                                   c=3,
                                                   h=self.cfg.DATASET.width,
                                                   w=self.cfg.DATASET.height,
                                                   max_move=self.data_patch['beta'])
                # img_patch (1,15,256,256)
                clip[b] = img_patch.squeeze(0)

        if "SF_MultiLoss" in self.cfg.MODEL.method:
            pixel_loss, separate_loss, compact_loss = self.calculate_loss_sf_multi(clip)
        elif "SF_KF_MultiLoss" in self.cfg.MODEL.method:
            pixel_loss, separate_loss, compact_loss = self.calculate_loss_sf_kf_multi(clip)
        else:
            pixel_loss, separate_loss, compact_loss = self.calculate_loss_default(clip)

        loss_compact = self.cfg.TRAIN.other_params.loss_compact * compact_loss
        loss_separate = self.cfg.TRAIN.other_params.loss_separate * separate_loss

        modified_loss_mse = []
        for b in range(bz):
            mean_loss_pixel = torch.mean(pixel_loss[b])
            mean_loss_compact = torch.mean(loss_compact[b])
            mean_loss_separate = torch.mean(loss_separate[b])
            if state_pseudo_sf[b] or state_pseudo_kf[b] or state_pseudo_patch[b]:
                b_loss = -mean_loss_pixel - mean_loss_compact  # - mean_loss_separate
            else:  # normal
                b_loss = mean_loss_pixel + mean_loss_compact + mean_loss_separate
            modified_loss_mse.append(b_loss)

        stacked_loss_mse = torch.stack(modified_loss_mse)
        loss = torch.mean(stacked_loss_mse)

        # Explicitly delete tensors to release memory
        del clip, data_sf, data_kf, data_patch, state_pseudo_sf, state_pseudo_kf, state_pseudo_patch
        del pixel_loss, separate_loss, compact_loss
        return loss

    def OBJ(self, epoch, train_dataloader):
        """
        SF_SK: Skip Frames + Key Frames
        """
        self.logger.info(f'====>>>>> call: OBJ() of {self.cfg.MODEL.method}')
        bz = self.cfg.DATASET.train.batch_size_per_gpu
        p_obj = float(self.data_obj['p'])

        self.logger.info(f'bz = {bz}')
        self.logger.info(f'p_obj = {p_obj}')
        print(f'bz = {bz}')
        print(f'p_obj = {p_obj}')

        self.epoch = epoch + 1
        sum_loss_epoch = 0.0

        if 'loss_type' in self.cfg.TRAIN.other_params:
            self.logger.info(
                f'ABNORMAL: self.cfg.TRAIN.other_params.loss_type = {self.cfg.TRAIN.other_params.loss_type}')
            print(f'ABNORMAL: self.cfg.TRAIN.other_params.loss_type = {self.cfg.TRAIN.other_params.loss_type}')
        else:  # 'ICS'
            self.logger.info(f'ABNORMAL: self.cfg.TRAIN.other_params.loss_type NOT EXIST!!!!!')
            print(f'ABNORMAL: self.cfg.TRAIN.other_params.loss_type NOT EXIST!!!!!')

        for clip in tqdm(train_dataloader):
            # clip.shape = torch.Size([4, 15, 256, 256]);  clip.device='cpu'
            loss = self.calculate_loss_OBJ(clip)

            # Assuming model is your PyTorch model
            # for name, param in self.model.named_parameters():
            #    print(f"Parameter '{name}' device:", param.device)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # to avoid RAM usage increases for all epoch
            sum_loss_epoch += loss.item()  # detach() or .item()
            # Explicitly delete clip tensor to release memory
            del clip

        # average the loss_epoch
        self.loss_epoch = sum_loss_epoch / len(train_dataloader)
        self.logger.info(f'Loss Epoch: {self.loss_epoch:.9f}')
        print(f'Loss Epoch: {self.loss_epoch:.9f}')

        # save to neptune
        if self.run is not None:
            self.run["train/loss_epoch"].log(self.loss_epoch)
        return self.loss_epoch

    def SF_KF(self, epoch, train_dataloader):
        """
        SF_SK: Skip Frames + Key Frames
        """
        self.logger.info(f'====>>>>> call: SF_KF() of {self.cfg.MODEL.method}')
        bz = self.cfg.DATASET.train.batch_size_per_gpu
        p_sf = float(self.data_sf['p'])
        p_kf = float(self.data_kf['p'])

        self.logger.info(f'bz = {bz}')
        self.logger.info(f'p_sf = {p_sf}')
        self.logger.info(f'p_kf = {p_kf}')
        print(f'bz = {bz}')
        print(f'p_sf = {p_sf}')
        print(f'p_kf = {p_kf}')

        self.epoch = epoch + 1
        sum_loss_epoch = 0.0

        if 'loss_type' in self.cfg.TRAIN.other_params:
            self.logger.info(
                f'ABNORMAL: self.cfg.TRAIN.other_params.loss_type = {self.cfg.TRAIN.other_params.loss_type}')
            print(f'ABNORMAL: self.cfg.TRAIN.other_params.loss_type = {self.cfg.TRAIN.other_params.loss_type}')
        else:
            self.logger.info(f'ABNORMAL: self.cfg.TRAIN.other_params.loss_type NOT EXIST!!!!!')
            print(f'ABNORMAL: self.cfg.TRAIN.other_params.loss_type NOT EXIST!!!!!')

        for clip in tqdm(train_dataloader):
            # clip.shape = torch.Size([4, 15, 256, 256]);  clip.device='cpu'
            loss = self.calculate_loss_SF_KF(clip)

            # Assuming model is your PyTorch model
            # for name, param in self.model.named_parameters():
            #    print(f"Parameter '{name}' device:", param.device)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # to avoid RAM usage increases for all epoch
            sum_loss_epoch += loss.item()  # detach() or .item()
            # Explicitly delete clip tensor to release memory
            del clip

        # average the loss_epoch
        self.loss_epoch = sum_loss_epoch / len(train_dataloader)
        self.logger.info(f'Loss Epoch: {self.loss_epoch:.9f}')
        print(f'Loss Epoch: {self.loss_epoch:.9f}')

        # save to neptune
        if self.run is not None:
            self.run["train/loss_epoch"].log(self.loss_epoch)
        return self.loss_epoch

    def SF_KF_PA(self, epoch, train_dataloader):
        """
        SF_SK_PA: Skip Frames + Key Frames + Patch
        """
        self.logger.info(f'====>>>>> call: SF_KF_PA() of {self.cfg.MODEL.method}')
        bz = self.cfg.DATASET.train.batch_size_per_gpu
        p_sf = float(self.data_sf['p'])
        p_kf = float(self.data_kf['p'])
        p_patch = float(self.data_patch['p'])
        self.logger.info(f'bz = {bz}')
        self.logger.info(f'p_sf = {p_sf}')
        self.logger.info(f'p_kf = {p_kf}')
        self.logger.info(f'p_patch = {p_patch}')
        self.epoch = epoch + 1
        sum_loss_epoch = 0.0

        for clip in tqdm(train_dataloader):
            # clip.shape = torch.Size([4, 15, 256, 256]);  clip.device='cpu'
            loss = self.calculate_loss_SF_KF_PA(clip)

            # Assuming model is your PyTorch model
            # for name, param in self.model.named_parameters():
            #    print(f"Parameter '{name}' device:", param.device)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # to avoid RAM usage increases for all epoch
            sum_loss_epoch += loss.item()  # detach() or .item()
            # Explicitly delete clip tensor to release memory
            del clip

            # average the loss_epoch
        self.loss_epoch = sum_loss_epoch / len(train_dataloader)
        self.logger.info(f'Loss Epoch: {self.loss_epoch:.9f}')

        # save to neptune
        if self.run is not None:
            self.run["train/loss_epoch"].log(self.loss_epoch)
        return self.loss_epoch

    def test(self, test_dataloader, video_labels, checkpoint_filepath, visualization_dir_path):
        self.logger.info(f'====>>>>> call: test() of {self.cfg.MODEL.method}')
        return super().test(test_dataloader, video_labels, checkpoint_filepath, visualization_dir_path)
