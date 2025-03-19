import torch
from .pytorch_msssim import msssim
from torch.nn import functional as F
from math import log10

def loss_function(recon_x, x): 
    msssim_loss = ((1-msssim(x,recon_x)))/2
    f1 =  F.l1_loss(recon_x, x)
    #psnr_error=(10 * log10( 65025/ ((torch.abs(torch.sum(x) - torch.sum(recon_batch))))))
    psnr_error=(10 * log10( 65025/ ((torch.abs(torch.sum(x) - torch.sum(recon_x)))))) # are these the same?
    return msssim_loss, f1, psnr_error
