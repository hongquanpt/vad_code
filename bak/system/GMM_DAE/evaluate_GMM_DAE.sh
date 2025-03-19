#!/bin/bash
# https://stackoverflow.com/questions/47034061/executing-a-python-script-multiple-times-with-different-option-flags

# run a file bash shell: 
# step 0: add #!/bin/bash on top of this file
# step 1: chmod u+x system/GMM_DAE/evaluate_GMM_DAE.sh
# step 2: ./system/GMM_DAE/evaluate_GMM_DAE.sh

# Before running terminal command, you must change values in file .yaml 
# For example: TEST (checkpoint_filepaths, ano_score_alpha, update_threshold)
config_filepath='system/GMM_DAE/ped2.yaml'
                
# to get the path to file conda.sh run command: conda info | grep -i 'base environment'
source /home/asus/anaconda3/etc/profile.d/conda.sh
conda init bash
conda activate pyanomaly

echo ">>> main_GMM_DAE.py >>>>"
python main_GMM_DAE.py --cfgfile "$config_filepath" SYSTEM.phase 'evaluate' SYSTEM.device "cuda:0"

conda deactivate