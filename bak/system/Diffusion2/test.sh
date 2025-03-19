#!/bin/bash
# https://stackoverflow.com/questions/47034061/executing-a-python-script-multiple-times-with-different-option-flags
'''
# run a file bash shell: 
step 0: add #!/bin/bash on top of this file
step 1: chmod u+x system/Diffusion2/test.sh
step 2: ./system/Diffusion2/test.sh
'''
# Before running terminal command, you must change values in file .yaml 
# For example: TEST (checkpoint_filepaths, ano_score_alpha, update_threshold)
# ddpm_conditional.yaml # ddpm_unconditional.yaml
config_filepath='system/Diffusion2/ddpm_conditional.yaml'
                
# to get the path to file conda.sh run command: conda info | grep -i 'base environment'
source /home/anhnam/anaconda3/etc/profile.d/conda.sh
conda init bash
conda activate pyanomaly

echo ">>> main_Diffusion.py >>>>"
python main_diffusion2.py --cfgfile "$config_filepath" SYSTEM.phase 'test'

conda deactivate