# Kiểm tra phiên bản conda
```bash
conda -V
```

# Update conda version
```bash
conda update conda
```

# 1. Tạo môi trường ảo mới
- python=3.7 phiên bản python được cài
- anaconda: bao gồm danh sách các gói khác như: 
- ipykernel, ipython, h5py, matplotlib, pandas, pillow, numpy, tqdm, yaml
```bash
conda create -n vpttoan python=3.7 anaconda
conda create -n huongct python=3.7 anaconda
```

# 2. Initialize your shell by running 'conda init'
```bash
conda init bash
```

# 3. Restart your shell after running 'conda init'
- To see the list of all the available environments use command 
```bash
conda info -e
```

# 3. activate môi trường ảo vừa tạo
```bash
conda activate yourenvname
conda activate vpttoan
```

# 4. Register virtual environment
- Register virtual environment, để môi trường mới lên được jupyter!!!
```bash
python -m ipykernel install --user --name vpttoan --display-name "Python 3.7 (vpttoan)"
python -m ipykernel install --user --name huongct --display-name "Python 3.7 (huongct)"
```

- cài đặt pytorch CUDA
```bash
https://pytorch.org/get-started/locally/
https://pytorch.org/get-started/previous-versions/#v182-with-lts-support
```

- RTX 3060
```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

- liệt kê các packages đã cài trong conda
```bash
conda list
```

- cài đặt package mới trong môi trường ảo của bạn
```bash
conda install -n yourenvname packagename
conda install -n virtualpt packagename
```

- deactive môi trường ảo
```bash
conda deactivate
```

- delete môi trường ảo
```bash
conda remove -n yourenvname --all
```
# 5. Clone virtual environment
- To clone a virtual environment in PyTorch, you can use the conda command or virtualenv depending on how you created the environment.
```bash
    conda create --name new_env_name --clone old_env_name
```