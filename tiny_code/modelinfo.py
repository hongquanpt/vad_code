import os

from system import *
from dataset import *
from model import *

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchinfo import summary

from model.MNAD.PAMAE_Pred import PAMAE_Pred
from model.MNAD.PAMAE_Recon import PAMAE_Recon
from model.MNAD.PAMAE_woMem_Recon import PAMAE_woMem_Recon

from model.STEAL.STEAL_AE import STEAL_AE

from model.ConvAE.ConvVAE3D_Recon import Conv3D_AE_Recon
from model.ConvAE.ConvVAE_Recon import ConvVAE_Recon
from model.ConvAE.ConvAE_SVM import ConvAE_SVM

from model.ConvLSTM_AE.ConvLSTM_AE import ConvLSTM_AE

from model.ConvVRNN.VRNN import VRNN
from model.Diffusion2.modules import UNet
from model.FastAno.FastAnoAE import FastAnoAE
from model.GMM_DAE.AE_model import SI_DAE, DI_DAE


def print_pamae_Recon(device):
    print("===========>>> print_pamae_Recon(device) >>>=============")
    # input: 1 frames, 3 channels  [1, 3, 256, 256]
    # output: 1 frame, 3 channels  [1, 3, 256, 256]
    bz = 1
    num_channels = 3
    num_frames = 2

    model = PAMAE_Recon(num_channels=num_channels, num_frames=num_frames, device=device)
    m_items = F.normalize(torch.rand((10, 512), dtype=torch.float), dim=1)
    x = torch.rand(bz, num_channels * num_frames, 256, 256)
    summary_str = summary(model.to(device),
                          input_data=x.to(device),
                          keys=m_items.to(device),
                          verbose=0)
    print(summary_str)

    '''
    Total params: 14,872,515
    Trainable params: 14,872,515
    Non-trainable params: 0
    Total mult-adds (G): 49.79
    
    Input size (MB): 0.79
    Forward/backward pass size (MB): 618.14
    Params size (MB): 59.49
    Estimated Total Size (MB): 678.41
    '''


def print_pamae_Recon_wo_memory(device):
    print("===========>>> print_pamae_Recon_wo_memory(device) >>>=============")
    # input: 1 frames, 3 channels  [1, 3, 256, 256]
    # output: 1 frame, 3 channels  [1, 3, 256, 256]
    bz = 1
    num_channels = 3
    num_frames = 2

    model = PAMAE_woMem_Recon(num_channels=num_channels, num_frames=num_frames)
    x = torch.rand(bz, num_channels * num_frames, 256, 256)
    summary_str = summary(model.to(device),
                          input_data=x.to(device),
                          verbose=0)
    print(summary_str)
    '''
    Total params: 12,513,219
    Trainable params: 12,513,219
    Non-trainable params: 0
    Total mult-adds (G): 47.38
   
    Input size (MB): 0.79
    Forward/backward pass size (MB): 618.14
    Params size (MB): 50.05
    Estimated Total Size (MB): 668.97
    '''


def print_pamae_Pred(device):  # Predict the 5th frame from the 4 successive frames
    # input: 5 frames, 3 channels  [1, 3*4, 256, 256]
    # output: 1 frame, 3 channels  [1, 3, 256, 256]
    bz = 1
    num_channels = 3
    num_frames = 5

    model = PAMAE_Pred(num_channels=num_channels, num_frames=num_frames, device=device)
    x = torch.rand(bz, num_channels * (num_frames - 1), 256, 256)
    m_items = F.normalize(torch.rand((10, 512), dtype=torch.float), dim=1)
    summary_str = summary(model.to(device),
                          input_data=x.to(device),
                          keys=m_items.to(device),
                          verbose=0)
    print(summary_str)
    '''
    Total params: 15,651,843
    Trainable params: 15,651,843
    Non-trainable params: 0
    Total mult-adds (G): 229.52
    
    Input size (MB): 12.58
    Forward/backward pass size (MB): 2472.54
    Params size (MB): 62.60
    Estimated Total Size (MB): 2547.73
    '''


def print_VRNN_Pred():  # Predict the 4th frame from the 3 successive frames
    model = VRNN(input_size=(16, 16),
                 input_dim=1024,
                 hidden_dim=[512, 256, 128, 64, 32],
                 kernel_size=(3, 3),
                 num_layers=5)
    step = 3  # step=channel=3
    seq_len = (4 - 1) * step
    bz = 1

    summary_str = summary(model,
                          input_size=(bz, 9, 256, 256),
                          seq_len=seq_len, step=step,
                          verbose=0)
    print(summary_str)
    '''
    Total params: 93,402,478
    Trainable params: 93,402,478
    Non-trainable params: 0
    Total mult-adds (G): 275.69
   
    Input size (MB): 2.36
    Forward/backward pass size (MB): 1498.64
    Params size (MB): 373.61
    Estimated Total Size (MB): 1874.61
    '''


def print_UNet_input_size():
    model = UNet()
    bz = 1
    x = torch.rand(bz, 3, 64, 64)
    t = torch.randint(low=1, high=1000, size=(bz,))
    model_stats = summary(model,
                          input_size=(bz, 3, 64, 64),
                          t=t, verbose=0)
    print(model_stats)


def print_UNet_input_data(device):
    print("===========>>> print_UNet_input_data(device) >>>=============")
    model = UNet(device=device)
    bz = 1
    x = torch.rand(bz, 3, 64, 64)
    t = torch.randint(low=1, high=1000, size=(bz,))
    model_stats = summary(model.to(device),
                          input_data=x.to(device),
                          t=t.to(device),
                          verbose=0)
    print(model_stats)
    # output:  [1, 3, 64, 64]


def print_Conv3D_AE_Recon(device):
    print("===========>>> print_Conv3D_AE_Recon(device) >>>=============")
    model = Conv3D_AE_Recon().to(device)
    bz = 1
    summary_str = summary(model,
                          input_size=(bz, 1, 16, 256, 256),
                          # input_size=(bz, 1, 4, 256, 256),
                          verbose=0
                          )
    print(summary_str)


def print_ConvVAE_Recon():
    model = ConvVAE_Recon(num_frames=2)
    bz = 1
    summary_str = summary(model,
                          input_size=(bz, 3, 256, 256),
                          verbose=0)
    print(summary_str)
    '''
    ========================
    1) only AE:
    ========================
    Total params: 12,513,219
    Trainable params: 12,513,219
    Non-trainable params: 0
    Total mult-adds (G): 47.38
   
    Input size (MB): 0.79
    Forward/backward pass size (MB): 618.14
    Params size (MB): 50.05
    Estimated Total Size (MB): 668.97
    
    ========================
    2) VAE, latent_size=16
    ========================
    Total params: 38,203,363
    Trainable params: 38,203,363
    Non-trainable params: 0
    Total mult-adds (T): 4.72

    Input size (MB): 0.79
    Forward/backward pass size (MB): 622.33
    Params size (MB): 152.81
    Estimated Total Size (MB): 775.93
    
    ========================
    3) VAE, latent_size=32
    ========================
    Total params: 63,369,219
    Trainable params: 63,369,219
    Non-trainable params: 0
    Total mult-adds (T): 9.12
    
    Input size (MB): 0.79
    Forward/backward pass size (MB): 622.33
    Params size (MB): 253.47
    Estimated Total Size (MB): 876.59
    
    ========================
    4) VAE, latent_size=64
    ========================
    Total params: 113,700,931
    Trainable params: 113,700,931
    Non-trainable params: 0
    Total mult-adds (T): 17.92

    Input size (MB): 0.79
    Forward/backward pass size (MB): 622.33
    Params size (MB): 454.80
    Estimated Total Size (MB): 1077.92
    '''


def print_ConvLSTM_AE():
    model = ConvLSTM_AE()
    bz = 64
    summary_str = summary(model,
                          input_size=(bz, 1, 1, 227, 227),
                          verbose=0
                          )
    print(summary_str)
    '''
    Total params: 1,068,225
    Trainable params: 1,068,225
    Non-trainable params: 0
    Total mult-adds (G): 20.28
    ==========================================================================================
    Input size (MB): 2.06
    Forward/backward pass size (MB): 138.76
    Params size (MB): 4.27
    Estimated Total Size (MB): 145.09
    '''


def print_ConvAE_SVM():
    model = ConvAE_SVM(n_channel=3, num_frames=1)
    bz = 1
    summary_str = summary(model,
                          input_size=(bz, 3, 256, 256),
                          verbose=0)
    print(summary_str)


def print_FastAno(device):
    print("===========>>> print_FastAno(device) >>>=============")
    # input: 5 frames, 3 channels  [1, 3, 5, 256, 256]
    # output: 1 frame, 3 channels  [1, 3, 1, 256, 256]
    bz = 1
    num_channels = 3
    num_frames = 5

    model = FastAnoAE(batch_size=bz, channels=num_channels)
    x = torch.rand(bz, num_channels, num_frames, 256, 256)
    summary_str = summary(model.to(device),
                          input_data=x.to(device),
                          verbose=0)
    print(summary_str)
    '''
    bz = 8
    Total params: 2,163,011
    Trainable params: 2,163,011
    Non-trainable params: 0
    Total mult-adds (G): 351.08
    ==========================================================================================
    Input size (MB): 31.46
    Forward/backward pass size (MB): 918.55
    Params size (MB): 8.65
    Estimated Total Size (MB): 958.66
    '''


def print_SI_DAE():
    model = SI_DAE()
    bz = 1
    summary_str = summary(model,
                          input_size=(bz, 1, 64, 64),
                          verbose=0)
    print(summary_str)
    '''
    Total params: 109,289
    Trainable params: 109,289
    Non-trainable params: 0
    Total mult-adds (M): 73.37
    ==========================================================================================
    Input size (MB): 0.02
    Forward/backward pass size (MB): 5.87
    Params size (MB): 0.44
    Estimated Total Size (MB): 6.32
    '''


def print_DI_DAE():
    model = DI_DAE()
    bz = 1
    summary_str = summary(model,
                          input_size=(bz, 1, 64, 64),
                          verbose=0)
    print(summary_str)
    '''
    Total params: 109,289
    Trainable params: 109,289
    Non-trainable params: 0
    Total mult-adds (M): 73.37
    ==========================================================================================
    Input size (MB): 0.02
    Forward/backward pass size (MB): 5.87
    Params size (MB): 0.44
    Estimated Total Size (MB): 6.32
    '''


def print_STEAL(device):
    print("===========>>> print_STEAL(device) >>>=============")
    # input: 16 frames, 1 channels  [1, 1, 16, 256, 256]
    # output: 16 frame, 1 channels  [1, 1, 16, 256, 256]
    bz = 1
    num_frames = 16
    num_channels = 1

    model = STEAL_AE()
    x = torch.rand(bz, num_channels, num_frames, 256, 256)
    summary_str = summary(model.to(device),
                          input_data=x.to(device),
                          verbose=0)
    print(summary_str)


def setup_device(mydevice="cuda:1"):
    """
    param mydevice: "cuda:0"; "cuda:1"; "cpu"
    """
    global device
    num_gpus_available = torch.cuda.device_count()

    if num_gpus_available > 1:
        print(f"The number of GPUs available = {num_gpus_available}")
        cudnn.benchmark = True
        cudnn.determinstic = False
        cudnn.enabled = True

        device = torch.device(mydevice)  # "cuda:0"; "cuda:1"; "cpu"
        print(f'device = {device}')
    else:
        print("Single GPU or no GPU available")
    return device


def main():
    """
    verbose = 0 (quiet): No output
    verbose = 1 (default): Print model summary
    verbose = 2 (verbose): Show weight and bias layers in full detail
    """
    device = setup_device(mydevice="cuda:1")

    # print_Conv3D_AE_Recon(device) # ok
    # print_pamae_Recon(device) # ok
    # print_pamae_Recon_wo_memory(device) # ok
    # print_pamae_Pred(device) # ok

    # print_VRNN_Pred()
    # print_ConvVAE_Recon()
    # print_ConvLSTM_AE()
    # print_ConvAE_SVM()

    # print_SI_DAE()
    # print_DI_DAE()

    # print_FastAno(device) # ok
    print_STEAL(device)  # ok

    # print_UNet_input_data(device) # ok


# call in terminal: python modelinfo.py
if __name__ == '__main__':
    main()
