#!/bin/bash

# https://stackoverflow.com/questions/47034061/executing-a-python-script-multiple-times-with-different-option-flags

# run a file bash shell: 
# step 0: add #!/bin/bash on top of this file
# step 1: chmod u+x system/GMM_DAE/preprocess_training.sh
# step 2: ./system/GMM_DAE/preprocess_training.sh

# Before running terminal command, you must change values in file .yaml 
# For example: TRAIN (begin_epoch, end_epoch, resume, checkpoint_filepath, skip_frames_prob)
config_filepath='system/GMM_DAE/ped2.yaml' # avenue.yaml, shanghaitech.yaml
#loop=(1 2 3 4 5)
loop=(1)

# to get the path to file conda.sh run command: conda info | grep -i 'base environment'
source /home/asus/anaconda3/etc/profile.d/conda.sh
conda init bash
conda activate pyanomaly

for id in "${loop[@]}"; do
    echo ">>> main_GMM_DAE.py >>>>"
    python main_GMM_DAE.py --cfgfile "$config_filepath" SYSTEM.phase 'preprocess_training' SYSTEM.device "cuda:0" SYSTEM.loop_index "$id"
done
conda deactivate