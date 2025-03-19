#!/bin/bash

# https://stackoverflow.com/questions/47034061/executing-a-python-script-multiple-times-with-different-option-flags

'''
# run a file bash shell: 
step 0: add #!/bin/bash on top of this file
step 1: chmod u+x system/Diffusion2/train.sh
step 2: ./system/Diffusion2/train.sh
'''
# Before running terminal command, you must change values in file .yaml 
# For example: TRAIN (begin_epoch, end_epoch, resume, checkpoint_filepath, skip_frames_prob)
# ddpm_conditional.yaml # ddpm_unconditional.yaml
config_filepath='system/Diffusion2/ddpm_conditional.yaml' 
#loop=(2 3 4 5)
loop=(1)

# to get the path to file conda.sh run command: conda info | grep -i 'base environment'
source /home/anhnam/anaconda3/etc/profile.d/conda.sh
conda init bash
conda activate pyanomaly

for id in "${loop[@]}"; do
    echo ">>> main_Diffusion.py >>>>"
    python main_Diffusion2.py --cfgfile "$config_filepath" SYSTEM.phase 'train' SYSTEM.loop_index "$id"
done
conda deactivate