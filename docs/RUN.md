# 1. Install anaconda
- Download file using command in Linux
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
```

- Run Anaconda Installation Script
```bash
chmod +x Anaconda3-2023.09-0-Linux-x86_64.sh
```

- Run file 
```bash
./Anaconda3-2023.09-0-Linux-x86_64.sh
```

- Activate and Test Installation
```bash
export PATH=~/anaconda3/bin
```

# 2. Create virtual environment
- We provide a file called `pya.yml` which can be used to install virtual environment 
- We exported it by command
```bash
    conda env export > pya.yml
```
- Then you can install it by command
```bash
    conda env create -f pya.yml
```
- Check GPU if available:
```bash
    >> python
    import torch
    print(torch.cuda.is_available()) # true
```
- Conda environment
```bash
    # list all packages in virtual environment
    conda list
    
    # list all virtual environments
    conda info --envs
```
- 
# 3. Run train or test phase of a method
- You can call train or test phase of a method from Terminal
```bash
python main.py --model 'PAMAE' --method 'PAMAE_KF_EN' --phase 'train' --dataset 'ped2'
```

- You can change config files which are located in directories `system/model/method`
- For example, the `MNAD` method can be trained on 3 datasets `ped2`, `avenue`, `shanghaitech` which are configured in 3 yaml files:
    - `system/PAMAE/PAMAE_KF_EN/ped2.yaml`

# 4 Generate Video from frames
- Install ffmpeg in Linux
```bash
Command 'ffmpeg' not found, but can be installed with:
sudo snap install ffmpeg  # version 4.3.1, or
sudo apt  install ffmpeg  # version 7:4.4.1-3ubuntu5
```

- Move to the folder which contain exported image, then generate `video_01.mp4`:
  - Command in Linux:
    ```bash
    ffmpeg -framerate 10 -i frame_01_%04d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p video_01.mp4
    ```
    ```bash
    ffmpeg -framerate 10 -i frame_01_%04d.png -c:v libx265 -profile:v high -crf 20 -pix_fmt yuv420p video_01.mp4
    ```
  - Command in Windows:
  ```bash
  ffmpeg -framerate 10 -i frame_01_%04d.png -c:v h264_nvenc -profile:v high -crf 20 -pix_fmt yuv420p video_01.mp4
  ```

# 5. The number of Cores and Threads in CPU, and status of GPU
- The Number of CPU Cores: 
  - Intel XEON GOLD 6138 2.0 up 3.7GHz: 20 Cores, 02 Threads per Core
  ```bash
  lscpu | grep "Core(s) per socket" 
  ```
- The Number of CPU Threads:
  ```bash
  lscpu | grep "Thread(s) per core"
  ```
- The Maximum Number of Workers: 
  ```bash
  Maximum workers = Number of CPU cores * Number of threads per core
  ```
- Process viewer: 
  - `htop`: displays a list of processes running on the system, including their PID (Process ID), user, CPU usage, memory usage, and more.
  ```bash
  htop
  ```
- GPU viewer:
  - `nvtop`:  a real-time overview of GPU usage, including information about GPU utilization, memory usage, temperature, power consumption, and more.
  ```bash
  nvtop
  ```
  - or
  ```bash
  gpustat
  ```
