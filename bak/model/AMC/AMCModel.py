from model.AbstractModel import AbstractModel
from .amc_networks import AMCGenerator, AMCDiscriminiator
from .utils import flow_batch_estimate, ParamSet
from .utils_evaluate import simple_diff, find_max_patch, calc_w, psnr_error, amc_score

from collections import OrderedDict

from model.util.basic_loss import *
from model.util.optimizer import get_optimizer_lr, get_scheduler
from model.util.process_data import decode_clip_cat, point_score, make_dir_path

from model.util.metrics import psnr_park, calculate_auc_scores, get_anomaly_rectanges
from model.util.auc import draw_auc_manual

from model.util.plot import plot_anomaly_scores

import torch
import torch.nn as nn
import time
import os

from PIL import Image
import torchvision.transforms as transforms
import torch.multiprocessing


# =======================================
# NOTE: 2019- Anomaly Detection in Video Sequence with Appearance-Motion Correspondence
# =======================================
def get_model_amc(cfg):
    if cfg.DATASET.normalized:
        rgb_max = 1.0
    else:
        rgb_max = 255.0
    if cfg.MODEL.flownet == 'flownet2':
        from collections import namedtuple
        from libs.flownet2.models import FlowNet2
        temp = namedtuple('Args', ['fp16', 'rgb_max'])
        args = temp(False, rgb_max)
        flow_model = FlowNet2(args)
        flow_model.load_state_dict(torch.load(cfg.MODEL.flow_model_path)['state_dict'])
    elif cfg.MODEL.flownet == 'liteflownet':
        from libs.liteflownet.models import LiteFlowNet
        flow_model = LiteFlowNet()
        flow_model.load_state_dict({strKey.replace('module', 'net'): weight for strKey, weight in
                                    torch.load(cfg.MODEL.flow_model_path).items()})
    else:
        raise Exception('Not support optical flow methods')

    generator_model = AMCGenerator(c_in=3, opticalflow_channel_num=2,
                                   image_channel_num=cfg.DATASET.channel,
                                   dropout_prob=0.7)
    discriminator_model = AMCDiscriminiator(c_in=5, filters=64)
    model_dict = OrderedDict()
    model_dict['Generator'] = generator_model
    model_dict['Discriminator'] = discriminator_model
    model_dict['FlowNet'] = flow_model
    return model_dict


class AMCModel(AbstractModel):
    def __init__(self, cfg, device, logger, writer, run):
        super().__init__(cfg, device, logger, writer, run)
        torch.manual_seed(2020)
        self.model = get_model_amc(cfg)

        self.G = self.model['Generator'].to(device)
        self.D = self.model['Discriminator'].to(device)
        self.F = self.model['FlowNet'].to(device)

        self.optimizer_G = get_optimizer_lr(cfg, cfg.TRAIN.optimizer.g_lr, self.model)
        self.scheduler_G = get_scheduler(cfg, self.optimizer_G)

        self.optimizer_D = get_optimizer_lr(cfg, cfg.TRAIN.optimizer.d_lr, self.model)
        self.scheduler_D = get_scheduler(cfg, self.optimizer_D)

        # get loss functions
        self.gan_loss = GANLoss(gan_mode='vanilla')
        self.gd_loss = GradientLoss()
        self.int_loss = IntensityLoss()
        self.op_loss = nn.L1Loss()

        # torch.multiprocessing.set_sharing_strategy('file_system')  # to fix error: Too many open files
        self.normalize = ParamSet(name='normalize',
                                  train={'use': True,
                                         'mean': [0.5, 0.5, 0.5],
                                         'std': [0.5, 0.5, 0.5]},
                                  val={'use': True,
                                       'mean': [0.5, 0.5, 0.5],
                                       'std': [0.5, 0.5, 0.5]})
        self.train_dataloader = None
        self.logger.info('Finish: AMCModel.__init__()')

    def save_model(self, model_filepath):
        print('call: save_model() in AMCModel')
        return super().save_model(model_filepath)

    def load_model(self, model_filepath):
        print('call: load_model() in AMCModel')
        return super().load_model(model_filepath)

    def save_model_state_dict(self, model_filepath):
        print('call: save_model_state_dict() in AMCModel')
        return super().save_model_state_dict(model_filepath)

    def load_model_state_dict(self, model_filepath):
        print('call: load_model_state_dict() in AMCModel')
        return super().load_model_state_dict(model_filepath)

    def save_checkpoint(self, checkpoint_filepath):
        print('call: save_checkpoint() in AMCModel')
        try:
            checkpoint = {
                'epoch': self.epoch,
                'G': self.G.state_dict(),
                'D': self.D.state_dict(),
                'F': self.F.state_dict(),
                'optimizer_G': self.optimizer_G.state_dict(),
                'optimizer_D': self.optimizer_D.state_dict(),
                'scheduler_G': self.scheduler_G.state_dict(),
                'scheduler_D': self.scheduler_D.state_dict(),
            }
            torch.save(checkpoint, checkpoint_filepath)
            self.logger.info(f'Success: save_checkpoint: {checkpoint_filepath}\n')
        except FileNotFoundError:
            self.logger.info(f"Error: Checkpoint file not found at {checkpoint_filepath}")
            return -1
        except Exception as e:
            self.logger.info(f'Error: save_checkpoint: {checkpoint_filepath} !!!\n')
            self.logger.info(f"Error reason: {str(e)}")
            return -1

    def load_checkpoint(self, checkpoint_filepath):
        print('call: load_checkpoint() in AMCModel')
        try:
            checkpoint = torch.load(checkpoint_filepath)
            self.epoch = checkpoint['epoch']
            self.G.load_state_dict(checkpoint['G'])
            self.D.load_state_dict(checkpoint['D'])
            self.F.load_state_dict(checkpoint['F'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
            self.scheduler_G.load_state_dict(checkpoint['scheduler_G'])
            self.scheduler_D.load_state_dict(checkpoint['scheduler_D'])

            self.logger.info(f'Success: load_checkpoint: {checkpoint_filepath} \n')
            return self.epoch
        except FileNotFoundError:
            self.logger.info(f"Error: Checkpoint file not found at {checkpoint_filepath}")
            return -1
        except Exception as e:
            self.logger.info(f'Error: load_checkpoint: {checkpoint_filepath} !!!\n')
            self.logger.info(f"Error reason: {str(e)}")
            return -1

    def set_requires_grad(self, nets, requires_grad=False):
        '''
        Parameters:
            nets(list) --- a list of networks
            requores_grad(bool) --- whether the networks require gradients or not  
        '''
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def train(self, train_dataloader, checkpoint_filepath, checkpoint_dir_path):
        self.logger.info('call: train() in AMCModel')
        return super().train(train_dataloader, checkpoint_filepath, checkpoint_dir_path)

    def train_epoch(self, epoch, train_dataloader):
        self.epoch = epoch + 1
        self.set_requires_grad(self.F, False)
        self.set_requires_grad(self.D, True)
        self.set_requires_grad(self.G, True)
        self.G.train()
        self.D.train()
        self.F.eval()

        loss_epoch = 0
        bz = self.cfg.DATASET.train.batch_size_per_gpu
        loss_G_total = 0
        loss_D_total = 0
        for j, data in enumerate(train_dataloader):
            # data.shape = torch.Size([4, 15, 256, 256])
            # self.logger.info(f'data.shape = {data.shape}')
            input_data, target = decode_clip_cat(data=data.to(self.device), model_type=self.cfg.MODEL.type,
                                                 num_frames=self.cfg.DATASET.num_frames)

            # update optim_G
            self.set_requires_grad(self.D, False)
            output_flow_G, output_frame_G = self.G(input_data)
            gt_flow_esti_tensor = torch.cat([input_data, target], 1)
            flow_gt_vis, flow_gt = flow_batch_estimate(self.F, gt_flow_esti_tensor,
                                                       self.normalize.param['train'],
                                                       optical_size=[self.cfg.DATASET.height, self.cfg.DATASET.width],
                                                       # [H, W]=[384, 512]
                                                       output_format='Y')
            fake_g = self.D(torch.cat([target, output_flow_G], dim=1))

            loss_g_adv = self.gan_loss(fake_g, True)
            loss_gd = self.gd_loss(output_frame_G, target)
            loss_int = self.int_loss(output_frame_G, target)
            loss_op = self.op_loss(output_flow_G, flow_gt)

            loss_G = 0.25 * loss_g_adv + loss_gd + loss_int + 2 * loss_op
            loss_G = torch.mean(loss_G)

            self.optimizer_G.zero_grad()
            loss_G.backward()
            self.optimizer_G.step()

            if self.cfg.TRAIN.lr_scheduler.use:
                self.scheduler_G.step()
                self.scheduler_D.step()
            # update optim_D
            self.set_requires_grad(self.D, True)
            real_d = self.D(torch.cat([target, flow_gt], dim=1))
            fake_d = self.D(torch.cat([target, output_flow_G.detach()], dim=1))
            loss_d_1 = self.gan_loss(real_d, True)
            loss_d_2 = self.gan_loss(fake_d, False)

            loss_D = (loss_d_1 + loss_d_2) * 0.5
            loss_D = torch.mean(loss_D)

            self.optimizer_D.zero_grad()
            loss_D.backward()
            self.optimizer_D.step()

            loss_G_total += loss_G
            loss_D_total += loss_D

        loss_G_total /= len(train_dataloader)  # average the loss_G_total
        loss_D_total /= len(train_dataloader)  # average the loss_D_total
        self.logger.info('Loss Generator {:.9f}'.format(loss_G_total))
        self.logger.info('Loss Discriminator {:.9f}'.format(loss_D_total))
        self.loss_epoch = loss_D_total
        return self.loss_epoch

    def set_train_dataloader(self, train_dataloader):
        self.train_dataloader = train_dataloader

    def test(self, test_dataloader, video_labels, checkpoint_filepath, visualization_dir_path):
        # Code: PyAnomaly/pyanomaly/hook/amc_hooks.py: class AMCEvaluateHook(EvaluateHook):
        self.logger.info('call: test() in AMCModel')
        # =========================
        # Load the pretrained model
        # =========================
        epoch = self.load_checkpoint(checkpoint_filepath)
        if epoch == -1:
            return
        self.G = self.G.to(self.device)
        self.D = self.D.to(self.device)
        self.F = self.F.to(self.device)
        # =========================
        # Evaluate model
        # =========================
        self.set_requires_grad(self.F, False)
        self.set_requires_grad(self.D, False)
        self.set_requires_grad(self.G, False)
        self.G.eval()
        self.D.eval()
        self.F.eval()

        with torch.no_grad():
            # ==============================================
            # calc the scores for the training set
            # ==============================================
            w_dict = OrderedDict()
            len_dataset = len(self.train_dataloader)
            for idx, clips_of_video in enumerate(self.train_dataloader):  # For each video
                patch_scores = []
                for clip in clips_of_video:  # For each clips in the i-th video
                    input_data_test, target_test = decode_clip_cat(data=clip.to(self.device),
                                                                   model_type=self.cfg.MODEL.type,
                                                                   num_frames=self.cfg.DATASET.num_frames)

                    output_flow_G, output_frame_G = self.G(input_data_test)
                    gtFlowEstim = torch.cat([input_data_test, target_test], 1)
                    gtFlow_vis, gtFlow = flow_batch_estimate(self.F, gtFlowEstim,
                                                             self.normalize.param['val'],
                                                             output_format='Y',
                                                             optical_size=[self.cfg.DATASET.height,
                                                                           self.cfg.DATASET.width])
                    diff_appe, diff_flow = simple_diff(target_test, output_frame_G, gtFlow, output_flow_G)
                    patch_score_appe, patch_score_flow, _, _ = find_max_patch(diff_appe, diff_flow)
                    patch_scores.append([patch_score_appe, patch_score_flow])
                patch_scores = torch.tensor(patch_scores)
                patch_scores = torch.mean(patch_scores)
                frame_w = torch.mean(patch_scores[:, 0])
                flow_w = torch.mean(patch_scores[:, 1])
                w_dict[idx] = [len_dataset, frame_w, flow_w]
            wf, wi = calc_w(w_dict)
            print(f'wf:{wf}, wi:{wi}')
            # ==============================================
            # calc the score for the test dataset
            # ==============================================
            video_psnr = []
            video_score = []
            video_ids = []
            for i, clips_of_video in enumerate(test_dataloader):  # For each video
                frame_psnr = []
                frame_score = []
                frame_ids = []
                j = 0
                # amc_hooks.py
                for clip in clips_of_video:  # For each clips in the i-th video
                    frame_ids.append(self.cfg.DATASET.num_frames + j)
                    # inputs.shape = target.shape = torch.Size([1, 12, 256, 256])
                    input_data_test, target_test = decode_clip_cat(data=clip.to(self.device),
                                                                   model_type=self.cfg.MODEL.type,
                                                                   num_frames=self.cfg.DATASET.num_frames)

                    g_output_flow, g_output_frame = self.G(input_data_test)
                    gt_flow_esti_tensor = torch.cat([input_data_test, target_test], 1)
                    flow_gt_vis, flow_gt = flow_batch_estimate(self.F, gt_flow_esti_tensor,
                                                               self.normalize.param['val'],
                                                               output_format='Y',
                                                               optical_size=[self.cfg.DATASET.height,
                                                                             self.cfg.DATASET.width])
                    test_psnr = psnr_error(g_output_frame, target_test)
                    score, _, _ = amc_score(target_test, g_output_frame, flow_gt, g_output_flow, wf, wi)
                    test_psnr = test_psnr.tolist()
                    score = score.tolist()
                    frame_psnr.append(test_psnr)
                    frame_score.append(score)

                    j = j + 1
                smax = max(frame_score)
                normal_scores = np.array([np.divide(s, smax) for s in frame_score])
                normal_scores = np.clip(normal_scores, 0, None)

                video_psnr.append(frame_psnr)
                video_score.append(normal_scores)
                video_ids.append(frame_ids)

        assert len(video_psnr) == len(
            video_labels), f'Ground truth has {len(video_labels)} videos, BUT got {len(video_psnr)} detected videos!'

        # Calculate AUC scores
        export_dir = visualization_dir_path
        auc, fpr, tpr, thresholds = calculate_auc_scores(self.cfg, video_psnr, video_labels)

        # Export AUC graph
        self.logger.info(f'AUC: {auc * 100:.2f}%')
        auc_filepath = os.path.join(export_dir, 'AUC.png')
        draw_auc_manual(fpr, tpr, color='darkorange', filepath=auc_filepath)
        self.writer.flush()
