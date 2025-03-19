# This package is cloned from HF2VAD: 
- A Hybrid Video Anomaly Detection Framework via Memory-Augmented Flow Reconstruction and Flow-Guided Frame Prediction
- https://github.com/LiUzHiAn/hf2vad
# But we can reference to Nvidia source: 
- https://github.com/NVIDIA/flownet2-pytorch

# Before using flownet, we must build it:
```commandline
chmod u+x libs/flownet_networks/install.sh
libs/flownet_networks/install.sh
```
- Build it by calling the file "install.sh" in terminal
- Download pre-trained model (FlowNet2_checkpoint.pth.tar) by calling the file "/weights/download.py" in terminal
- Reference: muc 2.2.4 va muc 2.3.1 trong link sau:
    - https://www.notion.so/anhle-lqdtu/2-Install-e1f2e075953441679e1da449589865d0#0f6ee441a7b940d9ade18eb5c24a628e

## 2.3. Pretrain Models

### 2.3.1. Optical Flow

This package will use `Flownet2` and `Liteflownet`, so please follow the instructions on their Github to get the models and users should change the location of these models in configuration.

- [Flownet2](https://github.com/NVIDIA/flownet2-pytorch)
    - download pre-trained model file `FlowNet2_checkpoint.pth.tar`:
        
        https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view
        
- [LiteFlowNet](https://github.com/sniklaus/pytorch-liteflownet)
    - download pre-trained model file `network-sintel.pytorch`:
        
        https://gitcode.net/u011622208/pytorch-liteflownet3?from_codechina=yes => https://drive.google.com/file/d/1vUSEIxXGZa9d2PQ82SG_gbbIUWLNfH50/view