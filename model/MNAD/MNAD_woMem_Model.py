from model.AbstractModel import AbstractModel
from model.MNAD.MNADModel import MNADModel
from .PAMAE_woMem_Pred import PAMAE_woMem_Pred
from .PAMAE_woMem_Recon import PAMAE_woMem_Recon

from model.util.optimizer import get_optimizer, get_scheduler
from model.util.process_data import decode_clip_cat, make_dir_path
from model.util.metrics import psnr_1, psnr_frame, psnr_patch
from model.util.metrics import calculate_auc_scores, calculate_eer, calculate_accuracy, calculate_f1_score
from model.util.auc import draw_auc_manual
from model.util.plot import plot_anomaly_scores, plot_gif, plot_video
from model.util.plot import plot_confusion_matrix

import torch
import os
import torch.multiprocessing
from tqdm import tqdm

# Check if the library exists
import importlib.util

lib_neptune = "neptune"
if importlib.util.find_spec(lib_neptune):
    # print(f"The '{lib_neptune}' library is available.")
    from neptune.types import File
else:
    print(f"The '{lib_neptune}' library is not available.")


class MNAD_woMem_Model(AbstractModel):
    def __init__(self, cfg, device, logger, run, data_clip):
        super().__init__(cfg, device, logger, run)
        self.data_clip = data_clip
        # ===================================================
        if self.cfg.MODEL.name == 'MNAD_woMem':
            if self.cfg.MODEL.type == 'prediction':
                self.logger.info('====>>>>> call MODEL: MNAD_woMem prediction')
                self.model = PAMAE_woMem_Pred(cfg.DATASET.channel, cfg.DATASET.num_frames)
            elif self.cfg.MODEL.type == 'reconstruction':
                self.logger.info('====>>>>> call MODEL: MNAD_woMem reconstruction')
                self.model = PAMAE_woMem_Recon(cfg.DATASET.channel, cfg.DATASET.num_frames)
            else:
                self.logger.info('Warning: cfg.model_type must be prediction or reconstruction!')

        # ===================================================
        self.loss_fn = torch.nn.MSELoss(reduction='none')
        if self.model is not None:
            self.model = self.model.to(self.device)
            self.optimizer = get_optimizer(self.cfg, self.model)
            self.scheduler = get_scheduler(self.cfg, self.optimizer)
        self.psnr_types = cfg.TEST.psnr_types
        # ===================================================
        self.logger.info('Finish: MNAD_woMem_Model.__init__()')

    def save_model(self, model_filepath):
        print('call: save_model() in MNAD_woMem_Model')
        return super().save_model(model_filepath)

    def load_model(self, model_filepath):
        print('call: load_model() in MNAD_woMem_Model')
        return super().load_model(model_filepath)

    def save_model_state_dict(self, model_filepath):
        print('call: save_model_state_dict() in MNAD_woMem_Model')
        return super().save_model_state_dict(model_filepath)

    def load_model_state_dict(self, model_filepath):
        print('call: load_model_state_dict() in MNAD_woMem_Model')
        return super().load_model_state_dict(model_filepath)

    def save_checkpoint(self, checkpoint_filepath):
        print('call: save_checkpoint() in MNAD_woMem_Model')
        return super().save_checkpoint(checkpoint_filepath)

    def load_checkpoint(self, checkpoint_filepath):
        print('call: load_checkpoint() in MNAD_woMem_Model')
        return super().load_checkpoint(checkpoint_filepath)

    def train(self, train_dataloader, checkpoint_filepath, checkpoint_dir_path):
        self.logger.info(f'====>>>>> call: train() f : {self.cfg.MODEL.method}')
        return super().train(train_dataloader, checkpoint_filepath, checkpoint_dir_path)

    def train_epoch(self, epoch, train_dataloader):
        self.logger.info(f'====>>>>> call: train_epoch() f : {self.cfg.MODEL.method}')
        return self.MNAD_woMem(epoch, train_dataloader)

    def MNAD_woMem(self, epoch, train_dataloader):
        self.epoch = epoch + 1

        bz = self.cfg.DATASET.train.batch_size_per_gpu
        self.logger.info(f'bz = {bz}')

        sum_loss_epoch = 0.0
        for clip in tqdm(train_dataloader):
            # data.shape = torch.Size([4, 15, 256, 256]);  data.device='cpu'
            # inputs, target = decode_clip_cat(data=clip.to(self.device), num_frames=self.cfg.DATASET.num_frames)
            seq_len = (self.cfg.DATASET.num_frames - 1) * self.cfg.DATASET.channel
            inputs = clip[:, :seq_len].to(self.device)
            target = clip[:, seq_len:].to(self.device)

            output = self.model(x=inputs)

            # Compute the loss separately on each GPU
            pixel_loss = self.loss_fn(output, target)
            loss = torch.mean(pixel_loss)

            # Assuming model is your PyTorch model
            # for name, param in self.model.named_parameters():
            #    print(f"Parameter '{name}' device:", param.device)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            sum_loss_epoch += loss.item()

            # Memory Management
            del clip, inputs, target, output

        # average the loss_epoch
        self.loss_epoch = sum_loss_epoch / len(train_dataloader)
        self.logger.info('Loss Epoch {:.9f}'.format(self.loss_epoch))

        # save to neptune
        if self.run is not None:
            self.run["train/loss_epoch"].log(self.loss_epoch)
        return self.loss_epoch

    def test(self, test_dataloader, video_labels, checkpoint_filepath, visualization_dir_path):
        self.logger.info(f'====>>>>> call: test() of {self.cfg.MODEL.method}')
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
        video_psnr = []
        video_feas_distance = []
        idx = 0
        nf = self.cfg.DATASET.num_frames  # nf= 5
        with torch.no_grad():
            # for clips_of_video in test_dataloader:  # For each video
            for video in test_dataloader:  # For each video
                self.logger.info("=========================================")
                video_id = idx + 1
                self.logger.info(f'[{video_id}/{len(test_dataloader)}]')

                frame_ids = []
                frame_psnr = []
                frame_targets = []
                frame_outputs = []
                j = 0
                export_dir = make_dir_path(visualization_dir_path, '{:02d}'.format(video_id))
                # for clip in clips_of_video:  # For each clips in the i-th video
                for f in tqdm(range(len(video) - nf + 1)):  # For each clips in the i-th video
                    clip = video[f: f + nf]
                    if self.data_clip['clip_mode'] == 'cat':  # for models: MNAD
                        clip = torch.cat(clip, dim=self.data_clip['clip_dim'] + 1)  # tensor.size(15,256,256)
                    elif self.data_clip['clip_mode'] == 'stack':  # for models: ConvLSTM_AE
                        clip = torch.stack(clip, dim=self.data_clip['clip_dim'] + 1)  # tensor.size(10,3,256,256)
                    frame_ids.append(self.cfg.DATASET.num_frames + j)
                    # clip = get_clip_tensor(1, clip, self.data_clip)
                    # 1. Compute losses
                    # inputs.shape = target.shape = torch.Size([1, 12, 256, 256])
                    # inputs, target = decode_clip_cat(data=clip.to(self.device), num_frames=self.cfg.DATASET.num_frames)
                    seq_len = (self.cfg.DATASET.num_frames - 1) * self.cfg.DATASET.channel
                    inputs = clip[:, :seq_len].to(self.device)
                    target = clip[:, seq_len:].to(self.device)
                    output = self.model(x=inputs)

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

                    frame_targets.append(target)
                    frame_outputs.append(output)

                    del clip, inputs, target, output

                    j = j + 1
                video_psnr.append(frame_psnr)

                # Plot scores_graph of a video
                scores, starts, ends, scores_graph = plot_anomaly_scores(video_id, frame_ids, video_labels[idx],
                                                                         frame_psnr, export_dir)
                # Upload scores_graph of a video to neptune
                if self.run is not None:
                    # Logging a series of images (scores_graph of a video)
                    self.run["test/scores_graphs"].append(
                        File(scores_graph),
                        description=f"Video: {'{:02d}'.format(video_id)}\n",
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

        assert len(video_psnr) == len(
            video_labels), f'Ground truth has {len(video_labels)} videos, BUT got {len(video_psnr)} detected videos!'

        # Get the best alpha
        pf = self.cfg.DATASET.num_frames - 1  # pf(processed_frames) = 4
        auc, fpr, tpr, optimal_idx, optimal_threshold, labels, scores = calculate_auc_scores(pf, video_psnr,
                                                                                             video_labels)

        eer = calculate_eer(fpr=fpr, fnr=1 - tpr)
        accuracy = calculate_accuracy(fpr=fpr, tpr=tpr, idx=optimal_idx)
        f1_score = calculate_f1_score(fpr=fpr, tpr=tpr, idx=optimal_idx)

        # Draw AUC graph to image
        export_dir = visualization_dir_path
        auc_filepath = os.path.join(export_dir, 'AUC.png')
        draw_auc_manual(fpr, tpr, color='darkorange', filepath=auc_filepath)

        # Draw Confusion matrix to image
        confusion_matrix_filepath = os.path.join(export_dir, 'confusion_matrix.png')
        plot_confusion_matrix(y_true=labels, y_score=scores,
                              optimal_threshold=optimal_threshold, filepath=confusion_matrix_filepath)

        self.logger.info(f'export_dir={export_dir}')
        self.logger.info(f'best_AUC={auc * 100:.2f}')
        self.logger.info(f'best_EER={eer * 100:.2f}')
        self.logger.info(f'best_accuracy={accuracy * 100:.2f}')
        self.logger.info(f'best_f1_score={f1_score * 100:.2f}')

        # Upload results (best_AUC, AUC image) to neptune
        if self.run is not None:
            self.run["test/best_AUC"].log(auc * 100)
            self.run["test/best_EER"].log(eer * 100)
            self.run["test/AUC"].upload(auc_filepath)
