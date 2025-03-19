from model.AbstractModel import AbstractModel
from .FastAnoAE import FastAnoAE

from model.util.loss import MultiLossFunction_FastAno
from model.util.optimizer import get_optimizer, get_scheduler
from model.util.process_data import decode_clip_stack, make_dir_path

from model.util.metrics import psnr_park, calculate_auc_scores
from model.util.auc import draw_auc_manual

from model.util.plot import plot_anomaly_scores, plot_gif

from PIL import Image
from dataset.image_reader import CV2_load_image

import torch
import torch.nn as nn
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
# =======================================
class FastAnoModel(AbstractModel):
    def __init__(self, cfg, device, logger, run, data_clip):
        # ===========================================
        # Procedure for both of training and testing
        # ===========================================
        super().__init__(cfg, device, logger, run)
        self.data_clip = data_clip

        bz = cfg.DATASET.train.batch_size_per_gpu
        self.model = FastAnoAE(batch_size=bz, channels=cfg.DATASET.channel)

        self.loss_fn = MultiLossFunction_FastAno()
        self.optimizer = get_optimizer(cfg, self.model)
        self.scheduler = get_scheduler(cfg, self.optimizer)
        # to fix error: Too many open files
        # torch.multiprocessing.set_sharing_strategy('file_system')
        self.logger.info('Finish: FastAnoModel.__init__()')

    def save_model(self, model_filepath):
        print('call: save_model() in FastAnoModel')
        return super().save_model(model_filepath)

    def load_model(self, model_filepath):
        print('call: load_model() in FastAnoModel')
        return super().load_model(model_filepath)

    def save_model_state_dict(self, model_filepath):
        print('call: save_model_state_dict() in FastAnoModel')
        return super().save_model_state_dict(model_filepath)

    def load_model_state_dict(self, model_filepath):
        print('call: load_model_state_dict() in FastAnoModel')
        return super().load_model_state_dict(model_filepath)

    def save_checkpoint(self, checkpoint_filepath):
        print('call: save_checkpoint() in FastAnoModel')
        return super().save_checkpoint(checkpoint_filepath)

    def load_checkpoint(self, checkpoint_filepath):
        print('call: load_checkpoint() in FastAnoModel')
        return super().load_checkpoint(checkpoint_filepath)

    def train(self, train_dataloader, checkpoint_filepath, checkpoint_dir_path):
        self.logger.info('call: train() in FastAnoModel')
        return super().train(train_dataloader, checkpoint_filepath, checkpoint_dir_path)

    def train_epoch(self, epoch, train_dataloader):
        self.epoch = epoch + 1
        loss_epoch = 0
        # The number of train_dataset (training clips) = len(train_dataset)
        # len(train_dataloader) =  len(train_dataset) / (batch_size * len(gpus))
        # train_dataloader:  Danh sách các clips (mỗi clip gồm 5 ảnh liên tiếp)
        # ==============================================================
        for j, clip in enumerate(train_dataloader):
            # self.logger.info(f'[{j+1}/{len(train_dataloader)}]')
            # self.logger.info(f'data.shape = {data.shape}') # torch.Size([bz, N=6, C=1, 256, 256])
            # inputs: torch.Size([bz, N=5, C=1, 256, 256])
            # target: torch.Size([bz, N=1, C=1, 256, 256])
            # PREDICTION
            inputs, target = decode_clip_stack(data=clip.to(self.device),
                                               num_frames=self.cfg.DATASET.num_frames)
            inputs = inputs.transpose(1, 2)  # torch.Size([bz, C=1, N=5, 256, 256])
            target = target.transpose(1, 2)  # torch.Size([bz, C=1, N=1, 256, 256])
            # self.logger.info(f'inputs.shape = {inputs.shape}')
            # self.logger.info(f'target.shape = {target.shape}')
            output = self.model(inputs)

            target = target.transpose(1, 2)  # torch.Size([bz, N=1, C=1, 256, 256])
            output = output.transpose(1, 2)  # torch.Size([bz, N=1, C=1, 256, 256])
            target = target.squeeze(dim=1)
            output = output.squeeze(dim=1)

            l1_loss, ssim_loss = self.loss_fn(output, target)
            loss = 0.25 * l1_loss + 0.75 * ssim_loss

            # print('l1_loss = ', l1_loss)
            # print('ssim_loss = ', ssim_loss)
            loss = torch.mean(loss)  # with multiple gpus
            # print('loss = ', loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_epoch += loss

        # average the loss_epoch
        self.loss_epoch = loss_epoch / len(train_dataloader)
        self.logger.info('Loss Epoch {:.9f}'.format(self.loss_epoch))
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
        elif self.data_clip['clip_mode'] == 'stack':  # for models: STEAL, FastAno
            clip_tensor = torch.stack(clip_tensor, dim=self.data_clip['clip_dim'])
        clip_tensor = clip_tensor.unsqueeze(0)  # torch.Size([1, 15, 256, 256])
        # print(f'clip_tensor.shape = {clip_tensor.shape}')
        return clip_tensor

    def test(self, test_dataloader, video_labels, checkpoint_filepath, visualization_dir_path):
        self.logger.info('call: test() in FastAnoModel')
        if len(video_labels) == 0:
            self.logger.info('WARNING: len(video_labels) == 0. Check PATH to label files')
            return
        # =========================
        # Load the pretrained model
        # =========================
        epoch = self.load_checkpoint(checkpoint_filepath)
        if epoch == -1:
            return
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.cfg.SYSTEM.gpus)
        self.model = self.model.to(self.device)
        # =========================
        # Evaluate model
        # =========================
        self.model.eval()
        video_psnr = []
        criterionL2 = torch.nn.MSELoss()
        export_dir = make_dir_path(visualization_dir_path, "result")

        # https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        with torch.no_grad():
            for idx, clips_of_video in enumerate(test_dataloader):  # For each video
                self.logger.info("=========================================")
                self.logger.info(f'[{idx + 1}/{len(test_dataloader)}]')
                frame_psnr = []
                frame_ids = []
                frame_targets = []
                frame_outputs = []
                j = 0
                for clip in clips_of_video:  # For each clips in the idx-th video
                    frame_ids.append(self.cfg.DATASET.num_frames + j)
                    clip_tensor = self.get_clip_tensor(clip)
                    inputs, target = decode_clip_stack(data=clip_tensor.to(self.device),
                                                       num_frames=self.cfg.DATASET.num_frames)
                    inputs = inputs.transpose(1, 2)  # torch.Size([bz, C=1, N=5, 256, 256])
                    target = target.transpose(1, 2)  # torch.Size([bz, C=1, N=1, 256, 256])
                    # self.logger.info(f'inputs.shape = {inputs.shape}')
                    # self.logger.info(f'target.shape = {target.shape}')
                    output = self.model(inputs)

                    # 2023-12-26: Xem lai cho nay nhe!
                    target = target.transpose(1, 2).squeeze(dim=1)  # torch.Size([bz=1, N=1, C=1, 256, 256])
                    output = output.transpose(1, 2).squeeze(dim=1)  # torch.Size([bz=1, N=1, C=1, 256, 256])

                    # target = target.squeeze(dim=1)
                    # output = output.squeeze(dim=1)
                    frame_targets.append(target)
                    frame_outputs.append(output)
                    # =========================
                    # Compute PSNR for each frame
                    # =========================
                    # https://github.com/cvlab-yonsei/MNAD/blob/d6d1e446e0ed80765b100d92e24f5ab472d27cc3/utils.py#L20
                    # (a+1)/2 <=> convert [-1,1] to [0,1]
                    # mse_img = criterionL2((output[0] + 1) / 2, (target[0] + 1) / 2)
                    # pixel_loss = criterionL2(output, target)
                    pixel_loss = torch.mean(criterionL2(output, target)).item()
                    frame_psnr.append(psnr_park(pixel_loss))
                    j = j + 1
                video_psnr.append(frame_psnr)
                export_dir = make_dir_path(visualization_dir_path, '{:02d}'.format(idx + 1))
                scores, starts, ends, scores_graph = plot_anomaly_scores(idx, frame_ids, video_labels[idx],
                                                                         frame_psnr, export_dir)
                if self.run is not None:
                    # Logging a series of images
                    self.run["test/scores_graphs"].append(
                        File(scores_graph),
                        description=f"Video: {'{:02d}'.format(idx + 1)}",
                    )
                plot_gif(idx, frame_targets, frame_outputs,
                         frame_ids, scores, starts, ends, export_dir)

        assert len(video_psnr) == len(
            video_labels), f'Ground truth has {len(video_labels)} videos, BUT got {len(video_psnr)} detected videos!'

        # ======================================================
        # Calcuate inference time and FPS
        # ======================================================
        ender.record()
        torch.cuda.synchronize()  # # Waits for everything to finish running
        inference_time = starter.elapsed_time(ender) * 1e-3  # milisecond to second
        self.logger.info(f'Inference time: {inference_time} seconds')
        total_frames = test_dataloader.batchsize * len(test_dataloader)
        self.logger.info(f'Speed: {total_frames / inference_time} FPS')

        # Calculate AUC scores of videos
        pf = self.cfg.DATASET.num_frames - 1
        auc, fpr, tpr, thresholds = calculate_auc_scores(pf, video_psnr, video_labels)

        # Export AUC graph to image
        self.logger.info(f'AUC: {auc * 100:.2f}')
        auc_filepath = os.path.join(export_dir, 'AUC.png')
        draw_auc_manual(fpr, tpr, color='darkorange', filepath=auc_filepath)

        if self.run is not None:
            self.run["test/best_auc"].log(auc * 100)
            # Logging a series of images
            self.run["test/AUC"].upload(auc_filepath)
