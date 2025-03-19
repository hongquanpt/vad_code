import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import exp
import sys
import os

sys.path.append('../')


# https://github.com/vt-le/astnet
# https://neptune.ai/blog/pytorch-loss-functions
class IntensityLoss(nn.Module):
    def __init__(self):
        super(IntensityLoss, self).__init__()

    def forward(self, prediction, target):
        # return torch.mean(torch.abs((prediction - target) ** 2))    # it's mine
        return torch.mean(torch.pow(torch.abs(prediction - target), 2), dim=[1, 2, 3], keepdim=True).squeeze()   # PyAnomaly
        # return torch.pow(torch.abs(prediction - target), 2)  # PyAnomaly


class GradientLoss(nn.Module):
    def __init__(self, channels, device):
        super(GradientLoss, self).__init__()

        pos = torch.from_numpy(np.identity(channels, dtype=np.float32))
        neg = -1 * pos

        # Note: when doing conv2d, the channel order is different from tensorflow, so do permutation.
        filter_x = torch.stack((neg, pos)).unsqueeze(0).permute(3, 2, 0, 1).to(device)
        filter_y = torch.stack((pos.unsqueeze(0), neg.unsqueeze(0))).permute(3, 2, 0, 1).to(device)

        # https://github.com/lv-tuan/semantic-segmentation/blob/b4fc685bb35d9b7547b805b1395c515876ec48db/sdcnet/models/sdc_net2d.py#L76
        self.register_buffer('filter_x', filter_x)
        self.register_buffer('filter_y', filter_y)

    def forward(self, prediction, target):
        # https://github.com/NVIDIA/vid2vid/blob/2e6d13755fc2e33200e7d4c0c44f2692d6ab0898/models/flownet.py#L27
        filter_x = self.filter_x
        filter_y = self.filter_y

        # Do padding to match the  result of the original tensorflow implementation
        gen_frames_x = nn.functional.pad(prediction, [0, 1, 0, 0])
        gen_frames_y = nn.functional.pad(prediction, [0, 0, 0, 1])
        gt_frames_x = nn.functional.pad(target, [0, 1, 0, 0])
        gt_frames_y = nn.functional.pad(target, [0, 0, 0, 1])

        gen_dx = torch.abs(nn.functional.conv2d(gen_frames_x, filter_x))
        gen_dy = torch.abs(nn.functional.conv2d(gen_frames_y, filter_y))
        gt_dx = torch.abs(nn.functional.conv2d(gt_frames_x, filter_x))
        gt_dy = torch.abs(nn.functional.conv2d(gt_frames_y, filter_y))

        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        return torch.mean(grad_diff_x + grad_diff_y, dim=[1, 2, 3], keepdim=True).squeeze()
        # return grad_diff_x + grad_diff_y


class L2Loss(nn.Module):
    """
    L2 loss is called mean square error nn.MSELoss
    https://pytorch.org/docs/master/generated/torch.nn.MSELoss.html?highlight=mse%20loss#torch.nn.MSELoss
    output = loss(prediction, target)
    """

    def __init__(self, eps=1e-8):  # 1 x 10^(-8) = 0.00000001
        super(L2Loss, self).__init__()
        self.eps = eps

    def forward(self, prediction, target):
        error = torch.mean(torch.pow((prediction - target), 2), dim=[1, 2, 3], keepdim=True).squeeze()
        # error = torch.pow((prediction - target), 2)
        error = torch.sqrt(error + self.eps)
        return error


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity 
    # cs = v1 / v2 # (anhle)
    cs = (cs + 1) / 2

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean() 
        # ret = ssim_map # (anhle)
        ret = (ret + 1) / 2
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
        ret = (ret + 1) / 2

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
# https://github.com/Po-Hsun-Su/pytorch-ssim
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)


# 2022- Attention-based residual autoencoder for video anomaly detection
# https://github.com/vt-le/astnet
class MultiLossFunction_ASTNet(nn.Module):
    def __init__(self, channels, device):
        super(MultiLossFunction_ASTNet, self).__init__()
        self.intensity_loss = IntensityLoss()
        self.gradient_loss = GradientLoss(channels, device)
        self.msssim_loss = MSSSIM()
        self.l2_loss = L2Loss()

    def forward(self, prediction, target):
        inte_loss = self.intensity_loss(prediction, target)
        grad_loss = self.gradient_loss(prediction, target)
        msssim = (1 - self.msssim_loss(prediction, target)) / 2
        l2_loss = self.l2_loss(prediction, target)

        return inte_loss, grad_loss, msssim, l2_loss


# ==============================================================

def get_outnorm(x: torch.Tensor, out_norm: str = '') -> torch.Tensor:
    """ Common function to get a loss normalization value. Can
        normalize by either the batch size ('b'), the number of
        channels ('c'), the image size ('i') or combinations
        ('bi', 'bci', etc)
    """
    # b, c, h, w = x.size()
    img_shape = x.shape

    if not out_norm:
        return 1

    norm = 1
    if 'b' in out_norm:
        # normalize by batch size
        # norm /= b
        norm /= img_shape[0]
    if 'c' in out_norm:
        # normalize by the number of channels
        # norm /= c
        norm /= img_shape[-3]
    if 'i' in out_norm:
        # normalize by image/map size
        # norm /= h*w
        norm /= img_shape[-1] * img_shape[-2]

    return norm


# 2017- Fast and accurate image super-resolution with deep laplacian pyramid networks
# https://github.com/victorca25/traiNNer/blob/master/codes/models/modules/loss.py
class Charbonnier_Loss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6, out_norm: str = 'bci'):
        super(Charbonnier_Loss, self).__init__()
        self.eps = eps
        self.out_norm = out_norm

    def forward(self, x, y):
        norm = get_outnorm(x, self.out_norm)
        loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps ** 2))
        return loss * norm  # sum + norm == torch.mean()?


# ==============================================================
class L1_Loss(nn.Module):
    def __init__(self):
        super(L1_Loss, self).__init__()

    def forward(self, prediction, target):
        mae_loss = nn.L1Loss()
        error = mae_loss(prediction, target)
        return error


# 2022- FastAno: Fast Anomaly Detection via Spatio-temporal Patch Transformation
class MultiLossFunction_FastAno(nn.Module):
    def __init__(self):
        super(MultiLossFunction_FastAno, self).__init__()
        self.l1_loss = L1_Loss()
        # self.ssim_loss = SSIM()

    def forward(self, prediction, target):
        l1_loss = self.l1_loss(prediction, target)
        # ssim = self.ssim_loss(prediction, target)
        '''
        The SSIM loss is computed as (1 - ssim_value), where ssim_value is the SSIM index between img1 and img2. This is because we want to minimize the loss (i.e., maximize the SSIM index) during training1.
        '''
        ssim_loss = 1 - ssim(prediction, target)
        return l1_loss, ssim_loss


def extract_lite_flow_net(device, input_last, predict_frame, target_frame):
    """
    param device: cuda or cpu
    param input_last (bz, 3, 256, 256) in range of (0,1)
    param predict_frame (bz, 3, 256, 256) in range of (0,1)
    param target_frame (bz, 3, 256, 256) in range of (0,1)
    """
    from libs.liteFlownet.lite_flownet import Network
    flow_net = Network().to(device)

    pretrained_filepath = os.path.join(os.getcwd(), 'libs', 'liteFlownet', 'network-default.pytorch')
    flow_net.load_state_dict(torch.load(pretrained_filepath))

    flow_net.eval()  # Use flow_net to generate optic flows, so set to eval mode.

    pred_flow_input = torch.cat([input_last, predict_frame], 1)
    gt_flow_input = torch.cat([input_last, target_frame], 1)

    # No need to train flow_net, use .detach() to cut off gradients.
    flow_pred = flow_net.batch_estimate(pred_flow_input, flow_net).detach()
    flow_gt = flow_net.batch_estimate(gt_flow_input, flow_net).detach()

    return flow_pred, flow_gt


def extract_flow_net2(device, input_last, predict_frame, target_frame):
    """
    param device: cuda or cpu
    param input_last (bz, 3, 256, 256) in range of (0,1)
    param predict_frame (bz, 3, 256, 256) in range of (0,1)
    param target_frame (bz, 3, 256, 256) in range of (0,1)
    """
    from libs.flownet2.models import FlowNet2SD
    flow_net = FlowNet2SD().to(device)
    pretrained_filepath = os.path.join(os.getcwd(), 'libs', 'flownet2', 'FlowNet2-SD.pth')
    flow_net.load_state_dict(torch.load(pretrained_filepath)['state_dict'])

    flow_net.eval()  # Use flow_net to generate optic flows, so set to eval mode.

    pred_flow_input = torch.cat([input_last.unsqueeze(2), predict_frame.unsqueeze(2)], 2)
    gt_flow_input = torch.cat([input_last.unsqueeze(2), target_frame.unsqueeze(2)], 2)

    # Input for flownet2sd is in (0, 255).
    flow_pred = (flow_net(pred_flow_input * 255.) / 255.).detach()
    flow_gt = (flow_net(gt_flow_input * 255.) / 255.).detach()

    return flow_pred, flow_gt


def centralize(img1, img2):
    b, c, h, w = img1.shape
    rgb_mean = torch.cat([img1, img2], dim=2).view(b, c, -1).mean(2).view(b, c, 1, 1)
    return img1 - rgb_mean, img2 - rgb_mean, rgb_mean


def extract_fast_flow_net(device, input_last, predict_frame, target_frame):
    """
    param device: cuda or cpu
    param input_last (bz, 3, 256, 256) in range of (0,1)
    param predict_frame (bz, 3, 256, 256) in range of (0,1)
    param target_frame (bz, 3, 256, 256) in range of (0,1)
    """
    from libs.FastFlowNet.models.FastFlowNet import FastFlowNet
    flow_net = FastFlowNet().to(device)
    pretrained_filepath = os.path.join(os.getcwd(), 'libs', 'FastFlowNet', 'checkpoints', 'fastflownet_ft_mix.pth')
    flow_net.load_state_dict(torch.load(pretrained_filepath))

    flow_net.eval()  # Use flow_net to generate optic flows, so set to eval mode.

    # C1
    img1, img2, _ = centralize(input_last, predict_frame)
    pred_flow_input = torch.cat([img1, img2], 1).to(device)
    flow_pred = flow_net(pred_flow_input).data
    print('flow_pred.shape = ', flow_pred.shape)

    # C2
    img3, img4, _ = centralize(input_last, target_frame)
    gt_flow_input = torch.cat([img3, img4], 1).to(device)
    flow_gt = flow_net(gt_flow_input).data
    print('flow_gt.shape = ', flow_gt.shape)

    return flow_pred, flow_gt


class Flow_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gen_flows, gt_flows):
        return torch.mean(torch.abs(gen_flows - gt_flows), dim=[1, 2, 3], keepdim=True).squeeze()
        # return torch.abs(gen_flows - gt_flows)


class SF_MultiLoss(nn.Module):
    def __init__(self, channels, device):
        super(SF_MultiLoss, self).__init__()
        self.intensity_loss = IntensityLoss().to(device)
        self.gradient_loss = GradientLoss(channels, device).to(device)
        self.msssim_loss = MSSSIM().to(device)
        self.l2_loss = L2Loss().to(device)
        self.flow_loss = Flow_Loss().to(device)
        self.device = device
        print("Finish: SF_MultiLoss()")

    def forward(self, input_last, predict_frame, target_frame):
        """
        param input_last (I_t)
        param predict_frame (I^_t+1)
        param target_frame (I_t+1)
        """

        intensity_loss = self.intensity_loss(predict_frame, target_frame)
        grad_loss = self.gradient_loss(predict_frame, target_frame)
        msssim = (1 - self.msssim_loss(predict_frame, target_frame)) / 2
        l2_loss = self.l2_loss(predict_frame, target_frame)

        # extract_lite_flow_net, extract_flow_net2, extract_fast_flow_net
        flow_pred, flow_gt = extract_fast_flow_net(self.device, input_last, predict_frame, target_frame)
        flow_loss = self.flow_loss(flow_pred, flow_gt)
        return intensity_loss, grad_loss, msssim, l2_loss, flow_loss

class SF_KF_MultiLoss(nn.Module):
    def __init__(self, channels, device):
        super(SF_KF_MultiLoss, self).__init__()
        self.intensity_loss = IntensityLoss().to(device)
        self.gradient_loss = GradientLoss(channels, device).to(device)
        self.msssim_loss = MSSSIM().to(device)
        self.l2_loss = L2Loss().to(device)
        print("Finish: SF_KF_MultiLoss()")

    def forward(self, predict_frame, target_frame):
        """
        param predict_frame (I^_t+1)
        param target_frame (I_t+1)
        """

        intensity_loss = self.intensity_loss(predict_frame, target_frame)
        grad_loss = self.gradient_loss(predict_frame, target_frame)
        msssim = 0.0 # (1 - self.msssim_loss(predict_frame, target_frame)) / 2
        l2_loss = self.l2_loss(predict_frame, target_frame)

        return intensity_loss, grad_loss, msssim, l2_loss

