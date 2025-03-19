## Dependencies
* Python 3.6
    ```commandline
    conda create --name mnad python=3.6
    conda activate mnad
    ```
* PyTorch 1.1.0
  * CUDA 10.0 (https://pytorch.org/get-started/previous-versions/)
    ```commandline
      conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0
      conda install -c conda-forge cudatoolkit=9.0
    ```
* Numpy
* Sklearn
* matplotlib
* cv2
    ```
      pip install numpy==1.16.6
      pip install scikit-learn==0.20.4
      pip install matplotlib==3.0.3
      pip install opencv-python==4.5.3.56
    ```

## pre-trained models (ped2, avenue)
```commandline
  https://drive.google.com/file/d/1w9YpyL9Nl2pNhPo4DVFPJ57hNuorb2pA/view?usp=sharing
  gdown 1w9YpyL9Nl2pNhPo4DVFPJ57hNuorb2pA
  https://drive.google.com/file/d/1w9YpyL9Nl2pNhPo4DVFPJ57hNuorb2pA/view?usp=sharing
  https://drive.google.com/file/d/1w9YpyL9Nl2pNhPo4DVFPJ57hNuorb2pA/view?usp=sharing
```

## docker
* pytorch docker (tham khảo)
  * https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html
  * https://hub.docker.com/r/pytorch/pytorch  (Official - ấn tab Tag để hiển thị các phiên bản)
  * https://github.com/cnstark/pytorch-docker
  * 
* Dockerfile
  ```commandline
  FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel
  FROM python:3.6-slim
  
  RUN pip install \
      numpy==1.16.6 \
      scikit-learn==0.20.4 \
      matplotlib==3.0.3 \
      opencv-python==4.5.3.56
      
  WORKDIR /VAD
  ```
