from model.AbstractModel import AbstractModel
from .PAMAE_Pred import PAMAE_Pred
from .PAMAE_Recon import PAMAE_Recon

from model.util.optimizer import get_optimizer, get_scheduler
from model.util.process_data import decode_clip_cat, point_score, make_dir_path
from model.util.process_data import decode_clip_cat_last_input
from model.util.metrics import psnr_1, psnr_frame, psnr_patch
from model.util.metrics import calculate_auc_scores_MNAD, calculate_eer, calculate_accuracy, calculate_f1_score
from model.util.auc import draw_auc_manual
from model.util.plot import plot_anomaly_scores_MNAD, plot_gif, plot_video
from model.util.plot import export_anomaly_scores_to_csv, plot_confusion_matrix
from dataset.util import is_exists_file, get_clip_tensor
from model.util.loss import SF_MultiLoss

import torch
import torch.nn.functional as functional
import os
import torch.multiprocessing
from tqdm import tqdm
import numpy as np

# Check if the library exists
import importlib.util

lib_neptune = "neptune"
if importlib.util.find_spec(lib_neptune):
    # print(f"The '{lib_neptune}' library is available.")
    from neptune.types import File
else:
    print(f"The '{lib_neptune}' library is not available.")


class MNADModel(AbstractModel):
    def __init__(self, cfg, device, logger, run, data_clip):
        super().__init__(cfg, device, logger, run)
        self.data_clip = data_clip
        torch.manual_seed(2020)
        # ===================================================
        if self.cfg.MODEL.name == 'MNAD':
            if self.cfg.MODEL.type == 'prediction':
                self.logger.info('====>>>>> call MODEL: MNAD_Model prediction')
                print(f'cfg.TRAIN.other_params.memory_size = {cfg.TRAIN.other_params.memory_size}')
                self.model = PAMAE_Pred(cfg.DATASET.channel, cfg.DATASET.num_frames,
                                        cfg.TRAIN.other_params.memory_size,
                                        cfg.TRAIN.other_params.feature_dim,
                                        cfg.TRAIN.other_params.memory_dim)
            elif self.cfg.MODEL.type == 'reconstruction':
                self.logger.info('====>>>>> call MODEL: MNAD_Model reconstruction')
                self.model = PAMAE_Recon(cfg.DATASET.channel, cfg.DATASET.num_frames,
                                         cfg.TRAIN.other_params.memory_size,
                                         cfg.TRAIN.other_params.feature_dim,
                                         cfg.TRAIN.other_params.memory_dim)
            else:
                self.logger.info('Warning: cfg.model_type must be prediction or reconstruction!')

        self.m_items = functional.normalize(torch.rand((cfg.TRAIN.other_params.memory_size,
                                                        cfg.TRAIN.other_params.memory_dim),
                                                       dtype=torch.float), dim=1)

        self.logger.info(f'self.m_items.shape = {self.m_items.shape}')
        print(f'self.m_items.shape = {self.m_items.shape}')
        # ===================================================
        self.loss_fn = torch.nn.MSELoss(reduction='none')
        if self.model is not None:
            self.model = self.model.to(self.device)
            self.optimizer = get_optimizer(self.cfg, self.model)
            self.scheduler = get_scheduler(self.cfg, self.optimizer)
        self.psnr_types = cfg.TEST.psnr_types
        # ===================================================
        # this command to fix Error: Too many open files
        """
        To set the sharing strategy for multiprocessing. 
        It determines how the data is shared between processes spawned by PyTorch's multiprocessing utilities,
        such as torch.multiprocessing.Process or torch.utils.data.DataLoader.
         
        By default, PyTorch uses "shared memory" to efficiently share data between processes. 
        When you call torch.multiprocessing.set_sharing_strategy('file_system'), 
        you are instructing PyTorch to use the "file system" as the mechanism for sharing data between processes instead of "shared memory".
        This means that data will be serialized and written to temporary files on disk, which are then read by the child processes.
        """
        # torch.multiprocessing.set_sharing_strategy('file_system')
        self.logger.info('Finish: MNADModel.__init__()')

    def save_model(self, model_filepath):
        self.logger.info('call: save_model() in MNADModel')
        return super().save_model(model_filepath)

    def load_model(self, model_filepath):
        self.logger.info('call: load_model() in MNADModel')
        return super().load_model(model_filepath)

    def save_model_state_dict(self, model_filepath):
        self.logger.info('call: save_model_state_dict() in MNADModel')
        return super().save_model_state_dict(model_filepath)

    def load_model_state_dict(self, model_filepath):
        self.logger.info('call: load_model_state_dict() in MNADModel')
        return super().load_model_state_dict(model_filepath)

    def save_checkpoint(self, checkpoint_filepath):
        self.logger.info('call: save_checkpoint() in MNADModel')
        self.logger.info(f'save_checkpoint: {checkpoint_filepath}')
        try:
            torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'loss': self.loss_epoch,
                'm_items': self.m_items
            }, checkpoint_filepath)
            self.logger.info(f'Success: save_checkpoint: {checkpoint_filepath}\n')
            return self.epoch
        except FileNotFoundError:
            self.logger.info(f"Error: Checkpoint file not found at {checkpoint_filepath}")
            return -1
        except Exception as e:
            self.logger.info(f'Error: save_checkpoint: {checkpoint_filepath} !!!\n')
            self.logger.info(f"Error reason: {str(e)}")
            return -1

    def load_checkpoint(self, checkpoint_filepath, is_load_scheduler=True):
        self.logger.info('call: load_checkpoint() in MNADModel')
        try:
            is_exists_file(checkpoint_filepath)
            checkpoint = torch.load(checkpoint_filepath,
                                    map_location=self.cfg.SYSTEM.device)
            self.epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # check if optimizer_state_dict exists or not
            self.optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', {}))
            # Try moving optimizer state to the GPU memory manually after loading it from the checkpoint.
            # torch.is_tensor(v): check if a variable is a PyTorch tensor and not any of its subclasses
            # isinstance(v, torch.Tensor):  If include subclass instances as well, 
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

            # check if scheduler_state_dict exists or not
            if 'scheduler_state_dict' in checkpoint:
                print("=======>>> scheduler_state_dict EXIST in checkpoint!!!")
            else:
                print("=======>>> scheduler_state_dict NOT_EXIST in checkpoint!!!")
            if is_load_scheduler:
                print("=======>>> load_scheduler")
                self.scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', {}))
            else:
                print("=======>>> NOT load_scheduler")
            self.loss_epoch = checkpoint['loss']
            self.m_items = checkpoint['m_items']
            self.logger.info(f'self.m_items.shape = {self.m_items.shape}')
            print(f'self.m_items.shape = {self.m_items.shape}')
            self.logger.info(f'Success: load_checkpoint: {checkpoint_filepath} \n')
            return self.epoch
        except FileNotFoundError:
            self.logger.info(f"Error: Checkpoint file not found at {checkpoint_filepath}")
            return -1
        except Exception as e:
            self.logger.info(f'Error: load_checkpoint: {checkpoint_filepath} !!!\n')
            self.logger.info(f"Error reason: {str(e)}")
            return -1

    def transfer_scheduler_checkpoint(self, checkpoint_filepath_source, checkpoint_filepath_destination):
        print('call: transfer_scheduler_checkpoint() in MNADModel')
        # load to get self.scheduler from checkpoint_source
        self.load_checkpoint(checkpoint_filepath_source, is_load_scheduler=True)

        # load to get: model, optimizer, memory items
        self.load_checkpoint(checkpoint_filepath_destination, is_load_scheduler=False)
        self.save_checkpoint(checkpoint_filepath_destination)

    def train(self, train_dataloader, checkpoint_filepath, checkpoint_dir_path):
        self.logger.info(f'====>>>>> call: train() of {self.cfg.MODEL.method}')
        return super().train(train_dataloader, checkpoint_filepath, checkpoint_dir_path)

    def train_epoch(self, epoch, train_dataloader):
        self.logger.info(f'====>>>>> call: train_epoch() of {self.cfg.MODEL.method}')
        if self.cfg.MODEL.method == 'MNAD':
            return self.MNAD(epoch, train_dataloader)
        elif self.cfg.MODEL.method == 'MNAD_MultiLoss':
            return self.MNAD_MultiLoss(epoch, train_dataloader)

    def MNAD(self, epoch, train_dataloader):
        self.loss_fn = torch.nn.MSELoss(reduction='none')
        self.epoch = epoch + 1
        sum_loss_epoch = 0.0
        sum_loss_compact = 0.0
        sum_loss_separate = 0.0
        # The number of train_dataset (training clips) = len(train_dataset)
        # len(train_dataloader) =  len(train_dataset) * batch_size * len(gpus)
        # train_dataloader:  Danh sách các clips (mỗi clip gồm 5 ảnh liên tiếp)
        for clip in tqdm(train_dataloader):
            # inputs, target = decode_clip_cat(data=clip.to(self.device), num_frames=self.cfg.DATASET.num_frames, channel=self.cfg.DATASET.channel)
            seq_len = (self.cfg.DATASET.num_frames - 1) * self.cfg.DATASET.channel
            inputs = clip[:, :seq_len].to(self.device)
            target = clip[:, seq_len:].to(self.device)

            output, self.m_items, separate_loss, compact_loss = self.model(x=inputs, keys=self.m_items, train=True)

            pixel_loss = torch.mean(self.loss_fn(output, target))
            loss_compact = torch.mean(self.cfg.TRAIN.other_params.loss_compact * compact_loss)
            loss_separate = torch.mean(self.cfg.TRAIN.other_params.loss_separate * separate_loss)
            loss = pixel_loss + loss_compact + loss_separate

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            sum_loss_epoch += loss.item()
            sum_loss_compact += loss_compact.item()
            sum_loss_separate += loss_separate.item()

            # Memory Management
            del clip, inputs, target, output
            del pixel_loss, loss_compact, loss_separate
            # torch.cuda.empty_cache()

        # average the loss_epoch
        self.loss_epoch = sum_loss_epoch / len(train_dataloader)
        loss_compact = sum_loss_compact / len(train_dataloader)
        loss_separate = sum_loss_separate / len(train_dataloader)

        # save to neptune
        if self.run is not None:
            self.run["train/loss_epoch"].log(self.loss_epoch)
            self.run["train/loss_compact"].log(loss_compact)
            self.run["train/loss_separate"].log(loss_separate)

        self.logger.info('loss_epoch {:.9f}'.format(self.loss_epoch))
        self.logger.info('loss_compact: {:.9f}'.format(loss_compact))
        self.logger.info('loss_separate: {:.9f}'.format(loss_separate))
        return self.loss_epoch

    def MNAD_MultiLoss(self, epoch, train_dataloader):
        self.loss_fn = SF_MultiLoss(self.cfg.DATASET.channel, self.device).to(self.device)
        self.epoch = epoch + 1
        sum_loss_epoch = 0.0
        sum_loss_compact = 0.0
        sum_loss_separate = 0.0
        # The number of train_dataset (training clips) = len(train_dataset)
        # len(train_dataloader) =  len(train_dataset) * batch_size * len(gpus)
        # train_dataloader:  Danh sách các clips (mỗi clip gồm 5 ảnh liên tiếp)
        intensity_loss = 0.0
        grad_loss = 0.0
        msssim = 0.0
        l2_loss = 0.0
        flow_loss = 0.0
        loss_compact = 0.0
        loss_separate = 0.0
        for j, clip in enumerate(tqdm(train_dataloader)):
            # inputs, target, input_last = decode_clip_cat_last_input(data=clip.to(self.device), num_frames=self.cfg.DATASET.num_frames, channel=self.cfg.DATASET.channel)
            a = (self.cfg.DATASET.num_frames - 2) * self.cfg.DATASET.channel  # i.e, (5-2)*3 = 9
            b = (self.cfg.DATASET.num_frames - 1) * self.cfg.DATASET.channel  # i.e, (5-1)*3 = 12
            input_last = clip[:, a:b].to(self.device)
            inputs = clip[:, :b].to(self.device)  # torch.Size([bz, 12, 256, 256])
            target = clip[:, b:].to(self.device)  # torch.Size([bz, 3, 256, 256])

            output, self.m_items, separate_loss, compact_loss = self.model(x=inputs, keys=self.m_items, train=True)

            # pixel_loss = torch.mean(self.loss_fn(output, target)) # with multiple batches
            intensity_loss, grad_loss, msssim, l2_loss, flow_loss = self.loss_fn(input_last, output, target)
            self.logger.info(f'intensity_loss.shape = {intensity_loss.shape}')
            self.logger.info(f'grad_loss.shape = {grad_loss.shape}')
            # self.logger.info(f'msssim.shape = {msssim.shape}')
            self.logger.info(f'l2_loss.shape = {l2_loss.shape}')
            self.logger.info(f'flow_loss.shape = {flow_loss.shape}')

            pixel_loss = torch.mean(intensity_loss + grad_loss + l2_loss + flow_loss)  # with multiple batches
            self.logger.info(f'compact_loss = {compact_loss}')
            self.logger.info(f'separate_loss = {separate_loss}')

            loss_compact = self.cfg.TRAIN.other_params.loss_compact * compact_loss
            loss_separate = self.cfg.TRAIN.other_params.loss_separate * separate_loss

            loss = pixel_loss + loss_compact + loss_separate
            loss = torch.mean(loss)  # with multiple gpus

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            sum_loss_epoch += loss.item()
            sum_loss_compact += loss_compact.item()
            sum_loss_separate += loss_separate.item()

            # Memory Management
            del clip, inputs, target, output
            del pixel_loss, loss_compact, loss_separate
            # torch.cuda.empty_cache()

        # average the loss_epoch
        self.loss_epoch = sum_loss_epoch / len(train_dataloader)
        loss_compact = sum_loss_compact / len(train_dataloader)
        loss_separate = sum_loss_separate / len(train_dataloader)

        # save to neptune
        if self.run is not None:
            self.run["train/loss_epoch"].log(self.loss_epoch)
            self.run["train/loss_compact"].log(loss_compact)
            self.run["train/loss_separate"].log(loss_separate)

        self.logger.info('loss_epoch {:.9f}'.format(self.loss_epoch))
        self.logger.info('loss_compact: {:.9f}'.format(loss_compact))
        self.logger.info('loss_separate: {:.9f}'.format(loss_separate))
        return self.loss_epoch

    def test(self, test_dataloader, video_labels, checkpoint_filepath, visualization_dir_path):
        self.logger.info(f'====>>>>> call: test() of {self.cfg.MODEL.method}')

        # Load the pretrained model
        epoch = self.load_checkpoint(checkpoint_filepath)
        if epoch == -1:
            return
        # if torch.cuda.device_count() > 1:
        #    self.model = nn.DataParallel(self.model, device_ids=self.cfg.SYSTEM.gpus)
        self.model = self.model.to(self.device)

        # Evaluate model
        self.model.eval()
        video_ids = []
        video_psnr = []
        video_feas_distance = []
        idx = 0
        nf = self.cfg.DATASET.num_frames  # nf= 5
        with torch.no_grad():
            for video in test_dataloader:  # For each video
                self.logger.info("=========================================")
                self.logger.info(f'[{idx + 1}/{len(test_dataloader)}]')
                print(f'[{idx + 1}/{len(test_dataloader)}]')
                frame_ids = []
                frame_psnr = []
                frame_feas_distance = []
                j = 0
                for f in tqdm(range(len(video) - nf + 1)):  # For each clips in the i-th video
                    clip = video[f: f + nf]
                    if self.data_clip['clip_mode'] == 'cat':  # for models: MNAD
                        clip = torch.cat(clip, dim=self.data_clip['clip_dim']+1)  # tensor.size(1, 15,256,256)
                    elif self.data_clip['clip_mode'] == 'stack':  # for models: ConvLSTM_AE
                        clip = torch.stack(clip, dim=self.data_clip['clip_dim']+1)  # tensor.size(1, 10,3,256,256)
                    frame_ids.append(self.cfg.DATASET.num_frames + j)

                    # 1. Compute losses
                    # inputs.shape = target.shape = torch.Size([1, 12, 256, 256])
                    # inputs, target = decode_clip_cat(data=clip.to(self.device), num_frames=self.cfg.DATASET.num_frames)
                    seq_len = (self.cfg.DATASET.num_frames - 1) * self.cfg.DATASET.channel
                    inputs = clip[:, :seq_len].to(self.device)
                    target = clip[:, seq_len:].to(self.device)

                    output, feas, self.m_items, compact_loss = self.model(x=inputs, keys=self.m_items, train=False)

                    # 2. Compute PSNR for each frame
                    if 'psnr_1' in self.psnr_types:
                        pixel_loss = psnr_1(output, target)
                        frame_psnr.append(pixel_loss)
                    elif 'psnr_max' in self.psnr_types:
                        if 'psnr_frame' in self.psnr_types:
                            pixel_loss = psnr_frame(output, target)
                        elif 'psnr_patch' in self.psnr_types:
                            pixel_loss = psnr_patch(output, target)
                        pixel_loss = torch.mean(pixel_loss).item()
                        frame_psnr.append(pixel_loss)

                    # 3. Compute feas_distance
                    mse_feas = compact_loss.item()
                    frame_feas_distance.append(mse_feas)
                    point_sc = point_score(output, target)

                    # 4. Update memory or not
                    if point_sc < self.cfg.TEST.other_params.update_threshold:
                        query = functional.normalize(feas, dim=1)
                        query = query.permute(0, 2, 3, 1)  # B x H x W x D
                        # print('query.shape =', query.shape) # torch.Size([1, 32, 32, 512])
                        self.m_items = self.model.memory.update(query, self.m_items, train=False)
                    j = j + 1
                    del clip, inputs, target, output
                    del pixel_loss, compact_loss, feas
                video_ids.append(frame_ids)
                video_psnr.append(frame_psnr)
                video_feas_distance.append(frame_feas_distance)
                del frame_ids, frame_psnr, frame_feas_distance
                idx += 1

        assert len(video_psnr) == len(
            video_labels), f'Ground truth has {len(video_labels)} videos, BUT got {len(video_psnr)} detected videos!'

        # Get the best alpha
        best_alpha = self.get_best_alpha(video_psnr, video_feas_distance, video_labels, visualization_dir_path)

        if "none" not in self.cfg.TEST.plot_types:
            # self.plot_ano_scores_mnad(test_dataloader, video_labels, best_alpha,
            #                           checkpoint_filepath, visualization_dir_path)
            self.plot(test_dataloader, video_labels, checkpoint_filepath, visualization_dir_path,
                      video_psnr, video_feas_distance, best_alpha)

    def get_best_alpha(self, video_psnr, video_feas_distance, video_labels, visualization_dir_path):
        best_fpr = 0
        best_tpr = 0
        best_auc = 0
        best_alpha = 0
        best_idx = 0
        best_threshold = 0
        best_labels = []
        best_scores = []
        for alpha in self.cfg.TEST.other_params.ano_score_alpha:
            # Calculate AUC scores of videos
            pf = self.cfg.DATASET.num_frames - 1  # pf(processed_frames) = 4
            auc, fpr, tpr, optimal_idx, optimal_threshold, labels, scores = calculate_auc_scores_MNAD(pf, video_psnr,
                                                                                                      video_feas_distance,
                                                                                                      video_labels,
                                                                                                      alpha)

            self.logger.info('-------------------')
            self.logger.info(f'auc = {auc * 100:.2f} alpha = {alpha}')

            if auc > best_auc:
                best_auc = auc
                best_fpr = fpr
                best_tpr = tpr
                best_idx = optimal_idx
                best_threshold = optimal_threshold
                best_alpha = alpha
                best_labels = labels
                best_scores = scores

                self.logger.info(f'best_AUC = {best_auc * 100:.2f} best_alpha(AUC) = {best_alpha}')

        best_eer = calculate_eer(fpr=best_fpr, fnr=1 - best_tpr)
        best_accuracy = calculate_accuracy(fpr=best_fpr, tpr=best_tpr, idx=best_idx)
        best_f1_score = calculate_f1_score(fpr=best_fpr, tpr=best_tpr, idx=best_idx)

        # Draw AUC graph to image
        export_dir = make_dir_path(visualization_dir_path, str(best_alpha))
        auc_filepath = os.path.join(export_dir, 'AUC.png')
        draw_auc_manual(best_fpr, best_tpr, color='darkorange', filepath=auc_filepath)

        # Draw Confusion matrix to image
        confusion_matrix_filepath = os.path.join(export_dir, 'confusion_matrix.png')
        plot_confusion_matrix(y_true=best_labels, y_score=best_scores,
                              optimal_threshold=best_threshold, filepath=confusion_matrix_filepath)

        self.logger.info(f'export_dir={export_dir}')
        self.logger.info(f'best_AUC={best_auc * 100:.2f} best_alpha(AUC)={best_alpha}')
        print(f'export_dir={export_dir}')
        print(f'best_AUC={best_auc * 100:.2f} best_alpha(AUC)={best_alpha}')

        self.logger.info(f'best_EER={best_eer * 100:.2f}')
        self.logger.info(f'best_accuracy={best_accuracy * 100:.2f}')
        self.logger.info(f'best_f1_score={best_f1_score * 100:.2f}')
        self.logger.info(f'best_threshold={best_threshold}')

        # Upload results (best_AUC, AUC image) to neptune
        if self.run is not None:
            self.run["test/best_AUC"].log(best_auc * 100)
            self.run["test/best_alpha(AUC)"].log(best_alpha)
            self.run["test/best_EER"].log(best_eer * 100)
            self.run["test/best_accuracy"].log(best_accuracy * 100)
            self.run["test/best_f1_score"].log(best_f1_score * 100)
            self.run["test/AUC"].upload(auc_filepath)
            self.run["test/ConfusionMatrix"].upload(confusion_matrix_filepath)
        return best_alpha

    def plot(self, test_dataloader, video_labels, checkpoint_filepath, visualization_dir_path,
             video_psnr, video_feas_distance, best_alpha):
        self.logger.info(f'====>>>>> call: plot_ano_scores_mnad() f : {self.cfg.MODEL.method}')
        # =========================
        # Load the pretrained model
        # =========================
        epoch = self.load_checkpoint(checkpoint_filepath)
        if epoch == -1:
            return
        self.model = self.model.to(self.device)
        # =========================
        # Evaluate model
        # =========================
        self.model.eval()
        idx = 0
        nf = self.cfg.DATASET.num_frames  # nf= 5
        with torch.no_grad():
            # for clips_of_video in test_dataloader:  # For each video
            for video in test_dataloader:  # For each video
                self.logger.info("=========================================")
                video_id = idx + 1
                self.logger.info(f'[{video_id}/{len(test_dataloader)}]')
                self.logger.info(f'psnr_types = {self.psnr_types}')

                frame_ids = []
                frame_targets = []
                frame_outputs = []
                j = 0
                export_dir = os.path.join(visualization_dir_path, str(best_alpha))
                export_dir = make_dir_path(export_dir, '{:02d}'.format(video_id))
                # for clip in clips_of_video:  # For each clips in the i-th video
                for f in tqdm(range(len(video) - nf + 1)):  # For each clips in the i-th video
                    clip = video[f: f + nf]
                    if self.data_clip['clip_mode'] == 'cat':  # for models: MNAD
                        clip = torch.cat(clip, dim=self.data_clip['clip_dim']+1)  # tensor.size(1, 15,256,256)
                    elif self.data_clip['clip_mode'] == 'stack':  # for models: ConvLSTM_AE
                        clip = torch.stack(clip, dim=self.data_clip['clip_dim']+1)  # tensor.size(1, 10,3,256,256)
                    frame_ids.append(self.cfg.DATASET.num_frames + j)
                    # clip = get_clip_tensor(1, clip, self.data_clip)
                    # 1. Compute losses
                    # inputs.shape = target.shape = torch.Size([1, 12, 256, 256])
                    seq_len = (self.cfg.DATASET.num_frames - 1) * self.cfg.DATASET.channel
                    inputs = clip[:, :seq_len].to(self.device)
                    target = clip[:, seq_len:].to(self.device)
                    output, feas, self.m_items, _ = self.model(x=inputs, keys=self.m_items, train=False)

                    # 3. Compute feas_distance
                    point_sc = point_score(output, target)

                    # 4. Update memory or not
                    if point_sc < self.cfg.TEST.other_params.update_threshold:
                        query = functional.normalize(feas, dim=1)
                        query = query.permute(0, 2, 3, 1)  # B x H x W x D
                        # print('query.shape =', query.shape) # torch.Size([1, 32, 32, 512])
                        self.m_items = self.model.memory.update(query, self.m_items, train=False)

                    frame_targets.append(target)
                    frame_outputs.append(output)

                    del clip, inputs, target, output
                    del feas

                # Plot scores_graph of a video
                scores, starts, ends, scores_graph = plot_anomaly_scores_MNAD(video_id, frame_ids, video_labels[idx],
                                                                              video_psnr[idx], video_feas_distance[idx],
                                                                              best_alpha, export_dir)
                export_anomaly_scores_to_csv(export_dir, video_id, frame_ids, scores)

                # Upload scores_graph of a video to neptune
                if self.run is not None:
                    # Logging a series of images (scores_graph of a video)
                    self.run["test/scores_graphs"].append(
                        File(scores_graph),
                        description=f"Video: {'{:02d}'.format(video_id)}\nAlpha: {best_alpha}",
                    )
                    for k in range(0, len(scores)):
                        self.run["test/scores"].log(scores[k])

                # Export [target image, output image, error image, graph] to GIF image or video
                if video_id in self.cfg.TEST.plot_video_ids or -1 in self.cfg.TEST.plot_video_ids:
                    if "gif" in self.cfg.TEST.plot_types:
                        plot_gif(video_id, frame_targets, frame_outputs,
                                 frame_ids, scores, starts, ends, export_dir)
                    if "video" in self.cfg.TEST.plot_types:
                        plot_video(video_id, frame_targets, frame_outputs,
                                   frame_ids, scores, starts, ends, export_dir)
                del frame_targets, frame_outputs
                idx += 1
