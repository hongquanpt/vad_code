from model.AbstractModel import AbstractModel
from .convAE import convAE

from model.util.optimizer import get_optimizer, get_scheduler
from model.util.process_data import make_dir_path
from model.util.metrics import psnr_1, psnr_frame
from model.util.metrics import calculate_auc_scores, calculate_eer
from model.util.auc import draw_auc_manual

from dataset.process import get_data_sf
from dataset.util import get_sample_sf

from PIL import Image
from dataset.image_reader import CV2_load_image

import torch
import torch.nn as nn
import numpy as np
import os
import torch.multiprocessing

# Check if the library exists
import importlib.util

lib_neptune = "neptune"
if importlib.util.find_spec(lib_neptune):
    # print(f"The '{lib_neptune}' library is available.")
    from neptune.types import File
else:
    print(f"The '{lib_neptune}' library is not available.")


# =======================================
# NOTE: Model này đã implement
# Không có phần skip_frames
# =======================================
class STEALModel(AbstractModel):
    def __init__(self, cfg, device, logger, run, data_clip):
        # ===========================================
        # Procedure for both of training and testing
        # ===========================================
        super().__init__(cfg, device, logger, run)
        self.data_clip = data_clip

        self.data_sf = None
        self.iter_sf = None

        # ===================================================
        self.model = convAE()
        self.loss_fn = nn.MSELoss(reduction='none')
        # ===================================================
        if self.model is not None:
            self.model = self.model.to(self.device)
            self.optimizer = get_optimizer(cfg, self.model)
            self.scheduler = get_scheduler(cfg, self.optimizer)
        self.setup_aug_data()
        self.psnr_types = cfg.TEST.psnr_types
        # ===================================================
        # to fix error: Too many open files
        # torch.multiprocessing.set_sharing_strategy('file_system')
        self.logger.info('Finish: STEALModel.__init__()')

    def save_model(self, model_filepath):
        print('call: save_model() in STEALModel')
        return super().save_model(model_filepath)

    def load_model(self, model_filepath):
        print('call: load_model() in STEALModel')
        return super().load_model(model_filepath)

    def save_model_state_dict(self, model_filepath):
        print('call: save_model_state_dict() in STEALModel')
        return super().save_model_state_dict(model_filepath)

    def load_model_state_dict(self, model_filepath):
        print('call: load_model_state_dict() in STEALModel')
        return super().load_model_state_dict(model_filepath)

    def save_checkpoint(self, checkpoint_filepath):
        print('call: save_checkpoint() in STEALModel')
        return super().save_checkpoint(checkpoint_filepath)

    def load_checkpoint(self, checkpoint_filepath):
        print('call: load_checkpoint() in STEALModel')
        return super().load_checkpoint(checkpoint_filepath)

    def train(self, train_dataloader, checkpoint_filepath, checkpoint_dir_path):
        self.logger.info('call: train() in STEALModel')
        return super().train(train_dataloader, checkpoint_filepath, checkpoint_dir_path)

    def train_epoch(self, epoch, train_dataloader):
        if self.cfg.MODEL.method == 'STEAL':
            return self.STEAL(epoch, train_dataloader)

    def setup_aug_data(self):
        if 'dict_sf' in self.cfg.DATASET:
            self.data_sf = get_data_sf(self.cfg)
        if self.data_sf is not None and self.data_sf['dataloader'] is not None:
            self.iter_sf = iter(self.data_sf['dataloader'])

    def STEAL(self, epoch, train_dataloader):
        self.logger.info(f'====>>>>> call: STEAL() of {self.cfg.MODEL.method}')
        bz = self.cfg.DATASET.train.batch_size_per_gpu
        p_sf = float(self.data_sf['p'])

        self.logger.info(f'bz = {bz}')
        self.logger.info(f'p_sf = {p_sf}')
        print(f'bz = {bz}')
        print(f'p_sf = {p_sf}')

        self.epoch = epoch + 1

        sum_loss_epoch = 0.0
        pseudo_loss_epoch = 0.0
        pseudo_loss_epoch_counter = 0
        normal_loss_epoch = 0.0
        normal_loss_epoch_counter = 0
        for j, data in enumerate(train_dataloader):
            # data.shape = torch.Size([4, 1, 16, 256, 256])
            # self.logger.info(f'data.shape = {data.shape}')
            state_pseudo_sf = []
            for b in range(bz):
                #  random samples from a uniform distribution over [0, 1).
                rand_number = np.random.rand()
                # skip frame pseudo anomaly
                is_pseudo = 0 <= rand_number < p_sf
                state_pseudo_sf.append(is_pseudo)
                if is_pseudo:
                    data_sf = get_sample_sf(self.data_sf['dataloader'], self.iter_sf, self.data_clip)
                    data[b] = data_sf[0]

            # RECONSTRUCTION
            output = self.model(data.to(self.device))
            loss_mse = self.loss_fn(output, data)
            modified_loss_mse = []
            for b in range(bz):
                if state_pseudo_sf[b]:
                    b_loss = torch.mean(-loss_mse[b])
                    modified_loss_mse.append(b_loss)
                    pseudo_loss_epoch += b_loss.cpu().detach().item()
                    pseudo_loss_epoch_counter += 1
                else:
                    b_loss = torch.mean(loss_mse[b])
                    modified_loss_mse.append(b_loss)
                    normal_loss_epoch += b_loss.cpu().detach().item()
                    normal_loss_epoch_counter += 1

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

    def get_clip_tensor(self, clip):
        clip_tensor = []
        if self.data_clip['read_format'] == 'PIL':
            for i in range(len(clip)):
                raw_frame = Image.open(clip[i][0]).convert('RGB')  # a PIL object image
                clip_tensor.append(self.data_clip['transform'](raw_frame))  # append a tensor (a transformed image)
        elif self.data_clip['read_format'] == 'CV2':
            for i in range(len(clip)):
                raw_frame = CV2_load_image(clip[i][0], self.data_clip['channel'],
                                           self.data_clip['width'], self.data_clip['height'])
                clip_tensor.append(self.data_clip['transform'](raw_frame))  # append a tensor (a transformed image)

        if self.data_clip['clip_mode'] == 'cat':  # for models: MNAD
            clip_tensor = torch.cat(clip_tensor, dim=self.data_clip['clip_dim'])
        elif self.data_clip['clip_mode'] == 'stack':  # for models: STEAL
            clip_tensor = torch.stack(clip_tensor, dim=self.data_clip['clip_dim'])
        clip_tensor = clip_tensor.unsqueeze(0)  # torch.Size([1, 1, 16, 256, 256])
        # print(f'clip_tensor.shape = {clip_tensor.shape}')
        return clip_tensor

    def test(self, test_dataloader, video_labels, checkpoint_filepath, visualization_dir_path):
        self.logger.info('call: test() in STEALModel')
        if len(video_labels) == 0:
            self.logger.info('WARNING: len(video_labels) == 0. Check PATH to label files')
            return
        # =========================
        # Load the pretrained model
        # =========================
        epoch = self.load_checkpoint(checkpoint_filepath)
        if epoch == -1:
            return
        # if torch.cuda.device_count() > 1:
        #    self.model = nn.DataParallel(self.model, device_ids=self.cfg.SYSTEM.gpus)
        self.model = self.model.to(self.device)
        # =========================
        # Evaluate model
        # =========================
        self.model.eval()
        self.loss_fn = nn.MSELoss(reduction='none')
        video_psnr = []
        center_id = self.cfg.DATASET.num_frames // 2
        with torch.no_grad():
            for idx, clips_of_video in enumerate(test_dataloader):  # For each video
                self.logger.info("=========================================")
                self.logger.info(f'[{idx + 1}/{len(test_dataloader)}]')
                self.logger.info(f'psnr_types = {self.psnr_types}')

                frame_ids = []
                frame_psnr = []
                frame_targets = []
                frame_outputs = []
                j = 0
                for clip in clips_of_video:  # For each clips in the idx-th video
                    frame_ids.append(center_id * (j + 1))
                    clip_tensor = self.get_clip_tensor(clip)

                    # RECONSTRUCTION
                    # clip.shape = target.shape = torch.Size([1, 1, 16, 256, 256])
                    clip = clip_tensor.to(self.device)
                    output = self.model(clip)
                    # 2. Compute PSNR for each frame
                    if 'psnr_1' in self.psnr_types:
                        pixel_loss = psnr_1(output[0, :, center_id], clip[0, :, center_id])
                        frame_psnr.append(pixel_loss)
                    elif 'psnr_max' in self.psnr_types:
                        pixel_loss = psnr_frame(output[0, :, center_id], clip[0, :, center_id])
                        pixel_loss = torch.mean(pixel_loss).item()
                        frame_psnr.append(pixel_loss)

                    # frame_targets.append(target[0, :, center_id])
                    # frame_outputs.append(output[0, :, center_id])
                    # frame_targets.append(clip[0, :, center_id])

                video_psnr.append(frame_psnr)

                '''
                export_dir = make_dir_path(visualization_dir_path, '{:02d}'.format(idx + 1))
                scores, starts, ends, scores_graph = plot_anomaly_scores(idx, video_labels[idx], frame_psnr, frame_ids, export_dir)
                if self.run is not None:
                    # Logging a series of images
                    self.run["test/scores_graphs"].append(
                        File(scores_graph),
                        description=f"Video: {'{:02d}'.format(idx + 1)}",
                    )
                    for k in range(0, len(scores)):
                        self.run["test/scores"].log(scores[k])
                '''

                '''
                # Export [target image, output image, error image, graph] to GIF image
                plot_gif(idx, frame_targets, frame_outputs,
                         frame_ids, scores, starts, ends, export_dir)
                '''

        assert len(video_psnr) == len(
            video_labels), f'Ground truth has {len(video_labels)} videos, BUT got {len(video_psnr)} detected videos!'

        # Calculate AUC scores and Export AUC graph to image
        self.plot_auc(video_psnr, video_labels, visualization_dir_path)

    def plot_auc(self, video_psnr, video_labels, visualization_dir_path):
        # Calculate AUC scores
        pf = self.cfg.DATASET.num_frames // 2  # pf=center_id = 16//2=8
        auc, fpr, tpr, thresholds = calculate_auc_scores(pf, video_psnr, video_labels)

        # Export AUC graph to image
        self.logger.info(f'AUC: {auc * 100:.2f}')
        export_dir = visualization_dir_path
        auc_filepath = os.path.join(export_dir, 'AUC.png')
        draw_auc_manual(fpr, tpr, color='darkorange', filepath=auc_filepath)

        if self.run is not None:
            self.run["test/best_AUC"].log(auc * 100)
            self.run["test/AUC"].upload(auc_filepath)
