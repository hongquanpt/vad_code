from model.AbstractModel import AbstractModel
from .wresnet1024_cattn_tsm import ASTNet as get_net1
from .wresnet2048_multiscale_cattn_tsmplus_layer6 import ASTNet as get_net2

from model.util.loss import MultiLossFunction_ASTNet
from model.util.optimizer import get_optimizer, get_scheduler
from model.util.process_data import decode_clip_list, make_dir_path

from model.util.metrics import psnr_park, calculate_auc_scores
from model.util.auc import draw_auc_manual

from model.util.plot import plot_anomaly_scores, plot_gif

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
# NOTE: Đã implement model này!
# =======================================
class ASTNetModel(AbstractModel):
    def __init__(self, cfg, device, logger, run):
        # ===========================================
        # Procedure for both of training and testing
        # ===========================================
        super().__init__(cfg, device, logger, run)
        torch.manual_seed(2020)
        if self.cfg.DATASET.name == 'ped2':
            self.model = get_net1(self.cfg)
        elif self.cfg.DATASET.name == 'avenue' or self.cfg.DATASET.name == 'shanghaitech':
            self.model = get_net2(self.cfg)
        else:
            self.logger.info('Warning: cfg.DATASET.name must be ped2, avenue, shanghaitech!')

        self.loss_fn = MultiLossFunction_ASTNet(cfg.DATASET.channel, self.device)
        self.optimizer = get_optimizer(cfg, self.model)
        self.scheduler = get_scheduler(cfg, self.optimizer)
        # torch.multiprocessing.set_sharing_strategy('file_system')  # to fix error: Too many open files
        self.logger.info('Finish: ASTNetModel.__init__()')

    def save_model(self, model_filepath):
        print('call: save_model() in ASTNetModel')
        return super().save_model(model_filepath)

    def load_model(self, model_filepath):
        print('call: load_model() in ASTNetModel')
        return super().load_model(model_filepath)

    def save_model_state_dict(self, model_filepath):
        print('call: save_model_state_dict() in ASTNetModel')
        return super().save_model_state_dict(model_filepath)

    def load_model_state_dict(self, model_filepath):
        print('call: load_model_state_dict() in ASTNetModel')
        return super().load_model_state_dict(model_filepath)

    def save_checkpoint(self, checkpoint_filepath):
        print('call: save_checkpoint() in ASTNetModel')
        return super().save_checkpoint(checkpoint_filepath)

    def load_checkpoint(self, checkpoint_filepath):
        print('call: load_checkpoint() in ASTNetModel')
        return super().load_checkpoint(checkpoint_filepath)

    def train(self, train_dataloader, checkpoint_filepath, checkpoint_dir_path):
        self.logger.info('call: train() in ASTNetModel')
        return super().train(train_dataloader, checkpoint_filepath, checkpoint_dir_path)

    def train_epoch(self, epoch, train_dataloader):
        loss_func_mse = nn.MSELoss(reduction='none')
        self.epoch = epoch + 1
        loss_epoch = 0

        loss_inte = 0
        loss_grad = 0
        loss_msssim = 0
        loss_l2 = 0
        # The number of train_dataset (training clips) = len(train_dataset)
        # len(train_dataloader) =  len(train_dataset) * batch_size * len(gpus) 
        # train_dataloader:  Danh sách các clips (mỗi clip gồm 5 ảnh liên tiếp)
        # seq_len = (self.cfg.DATASET.num_frames-1) * self.cfg.DATASET.channel # =(5-1)*3=12
        for j, data in enumerate(train_dataloader):
            # self.logger.info(f'[{j+1}/{len(train_dataloader)}]')

            # if data == clip is a list of 5 tensors (3x256x256) 
            # =>len(data) = 5;  data[0].shape = torch.Size([8, 3, 256, 256])
            # PREDICTION
            # phai co lenh nay vi data la 1 list cac tensor, khong phai 1 tensor
            for i in range(len(data)):
                data[i] = data[i].to(self.device)
            inputs, target = decode_clip_list(data=data)
            # inputs: a list of 4 tensor
            output = self.model(inputs)

            # compute loss
            # target = target.cuda(non_blocking=True)
            # print('output.device.type = ', output.device.type)
            # print('target.device.type = ', target.device.type)
            inte_loss, grad_loss, msssim_loss, l2_loss = self.loss_fn(output, target)
            loss = inte_loss + grad_loss + msssim_loss + l2_loss

            # compute PSNR
            mse_imgs = torch.mean(loss_func_mse((output + 1) / 2, (target + 1) / 2)).item()
            psnr = psnr_park(mse_imgs)

            # optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # log
            cur_lr = self.optimizer.param_groups[0]['lr']
            if (j + 1) % self.cfg.TRAIN.other_params.print_seq == 0:
                msg = 'train_dataloader: [{0}/{1}]\t' \
                      'Lr {lr:.6f}\t' \
                      '[inte {inte:.5f} + grad {grad:.4f} + msssim {msssim:.4f} + L2 {l2:.4f}]\t' \
                      'PSNR {psnr:.2f}'.format(j + 1, len(train_dataloader),
                                               lr=cur_lr,
                                               inte=inte_loss, grad=grad_loss, msssim=msssim_loss, l2=l2_loss,
                                               psnr=psnr)
                self.logger.info(msg)

            loss_epoch += loss
            loss_inte += inte_loss
            loss_grad += grad_loss
            loss_msssim += msssim_loss
            loss_l2 += l2_loss
        loss_epoch /= len(train_dataloader)  # average the loss_epoch
        loss_inte /= len(train_dataloader)
        loss_grad /= len(train_dataloader)
        loss_msssim /= len(train_dataloader)
        loss_l2 /= len(train_dataloader)

        self.loss_epoch = loss_epoch
        return self.loss_epoch

    def test(self, test_dataloader, video_labels, checkpoint_filepath, visualization_dir_path):
        self.logger.info('call: test() in ASTNetModel')
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
        #     self.model = nn.DataParallel(self.model, device_ids=self.cfg.SYSTEM.gpus)
        self.model = self.model.to(self.device)
        # =========================
        # Evaluate model
        # =========================
        self.model.eval()
        self.loss_fn = nn.MSELoss(reduction='none')
        video_psnr = []
        export_dir = make_dir_path(visualization_dir_path, "result")

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
                    for i in range(len(clip)):
                        clip[i].to(self.device)
                    inputs, target = decode_clip_list(data=clip)
                    output = self.model(inputs)

                    frame_targets.append(target)
                    frame_outputs.append(output)

                    pixel_loss = self.loss_fn((output[0] + 1) / 2, (target[0] + 1) / 2)
                    pixel_loss = torch.mean(pixel_loss).item()
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

        # Calculate AUC scores
        pf = self.cfg.DATASET.num_frames - 1
        auc, fpr, tpr, thresholds = calculate_auc_scores(pf, video_psnr, video_labels)

        # ======================================================
        # Export AUC graph
        # ======================================================
        self.logger.info(f'AUC: {auc * 100:.2f}')
        auc_filepath = os.path.join(visualization_dir_path, 'AUC.png')
        draw_auc_manual(fpr, tpr, color='darkorange', filepath=auc_filepath)

        if self.run is not None:
            self.run["test/best_auc"].log(auc * 100)
            # Logging a series of images
            self.run["test/AUC"].upload(auc_filepath)
