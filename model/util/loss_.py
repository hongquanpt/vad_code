from .loss_ssim import *


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, prediction, target):
        mae_loss = nn.L1Loss()
        error = mae_loss(prediction, target)
        return error


class MultiLossFunction_FastAno(nn.Module):
    def __init__(self):
        super(MultiLossFunction_FastAno, self).__init__()
        self.l1_loss = L1Loss()
        # self.ssim_loss = SSIM()

    def forward(self, prediction, target):
        l1_loss = self.l1_loss(prediction, target)
        # ssim = self.ssim_loss(prediction, target)
        ssim_loss = 1 - ssim(prediction, target)
        return l1_loss, ssim_loss
