'''
In Terminal: 
    conda activate pyanomaly
    python libs/flownet_networks/weights/download.py
'''
import subprocess
"""
URL = "https://drive.google.com/u/0/uc?id=1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da&export=download"
=> ID URL = 1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da
"""

sys_cmd = "gdown --id 1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da --output libs/flownet_networks/weights/FlowNet2_checkpoint.pth.tar"
child = subprocess.Popen(sys_cmd, shell = True)
child.wait()