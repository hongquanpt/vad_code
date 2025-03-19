# 1. Clone project from github.com
```commandline
    cd DATA
    git clone -b master git@github.com:lenhoanh/VAD.git
```

# 2. download files in terminal 
### using wget command
```bash
    # install wget
    sudo apt install wget
    
    # download a file from URL
    wget URL
    
    # download multiple files: 
    # save their URLs in a text file
    wget -i download_files.txt
```

# 3. Download file from Google Drive 
### using gdown
```bash
    pip install gdown
    
    # a file is stored on Gooogle Drive: 
    # ped1: https://drive.google.com/file/d/1fh7A9HtHmggT2VKYedR6yxyf3_Yll75D/view?usp=drive_link
    # ped2: https://drive.google.com/file/d/18hStjIEDbNcQESoPOSs7XYXmBQr08T-S/view?usp=sharing
    # ped2_training_Entropy: https://drive.google.com/file/d/1i4TqCgHNrZ0ow4uSpMjXrB-YKGtVBmO4/view?usp=drive_link
    # ped2_training_OBJ: https://drive.google.com/file/d/1ehfwhSxO_e2CNH5_A7pcmqAr4HwVwK63/view?usp=sharing
    
    # avenue: https://drive.google.com/file/d/1qUp50gPBZF4PfeTfYsE-yHHS2N0vvblG/view?usp=drive_link
    # avenue_training_Entropy: https://drive.google.com/file/d/1F9Uq6uVJ-Wfo5Db9mpPfWjYq2aOPcTtj/view?usp=drive_link
    # avenue_training_22: https://drive.google.com/file/d/1WF4cgQ_Zl60KC2HbX0sTG_UthRfOeSnC/view?usp=sharing
    # avenue_training_KF_EN_22: https://drive.google.com/file/d/1XaauL1tcKfHuljs84bIcU-qiFT6WBhaD/view?usp=sharing
    # avenue_training_OBJ: https://drive.google.com/file/d/11QhGNI2rgttl4LPL8vWuAXZsklaVRrsU/view?usp=sharing
    
    # shanghaitech: https://drive.google.com/file/d/1-HzukKeQnaijs06BsTV-EZ-3aOKeDwa-/view?usp=drive_link
    # shanghaitech_training_KF_EM: https://drive.google.com/file/d/1l9htn1E_abMpzbiVn2am0UfxW9fD2BJ8/view?usp=drive_link
    # shanghaitech_training_KF_EN: https://drive.google.com/file/d/1zvJckipsRKnhn6C98o44ScHsrabHUmBB/view?usp=drive_link
    # shanghaitech_training_KF_ES_30: https://drive.google.com/file/d/1T9IQgAvIxyqn2slCWXYUZQL9k2QFVEcb/view?usp=drive_link
    # shanghaitech_training_KF_OF_30: https://drive.google.com/file/d/1r6_C0P5ghvZNl5sh50Gvkytn9tgskwiX/view?usp=sharing
    # shanghaitech_test_frame_mask: https://drive.google.com/file/d/1G1DePyw-R7PYvQfWPTwHecYH6aB52i68/view?usp=drive_link
    
    # the id URL parameter 1RhEkVRUB7XTbSX6PDswPVaMON9THCKDU in the link
    gdown 1fh7A9HtHmggT2VKYedR6yxyf3_Yll75D --output ped1.zip
    gdown 18hStjIEDbNcQESoPOSs7XYXmBQr08T-S --output ped2.zip
    gdown 1i4TqCgHNrZ0ow4uSpMjXrB-YKGtVBmO4 --output ped2_training_Entropy.zip
    gdown 1ehfwhSxO_e2CNH5_A7pcmqAr4HwVwK63 --output ped2_training_OBJ.zip
    
    gdown 1qUp50gPBZF4PfeTfYsE-yHHS2N0vvblG --output avenue.zip
    gdown 1F9Uq6uVJ-Wfo5Db9mpPfWjYq2aOPcTtj --output avenue_training_Entropy.zip
    gdown 1WF4cgQ_Zl60KC2HbX0sTG_UthRfOeSnC --output avenue_training_22.zip
    gdown 1XaauL1tcKfHuljs84bIcU-qiFT6WBhaD --output avenue_training_KF_EN_22.zip
    gdown 11QhGNI2rgttl4LPL8vWuAXZsklaVRrsU --output avenue_training_OBJ.zip
    
    gdown 1-HzukKeQnaijs06BsTV-EZ-3aOKeDwa- --output shanghaitech.zip
    gdown 1l9htn1E_abMpzbiVn2am0UfxW9fD2BJ8 --output shanghaitech_training_KF_EM.zip
    gdown 1zvJckipsRKnhn6C98o44ScHsrabHUmBB --output shanghaitech_training_KF_EN.zip
    gdown 1T9IQgAvIxyqn2slCWXYUZQL9k2QFVEcb --output shanghaitech_training_KF_ES_30.zip
    gdown 1r6_C0P5ghvZNl5sh50Gvkytn9tgskwiX --output shanghaitech_training_KF_OF_30.zip
    gdown 1G1DePyw-R7PYvQfWPTwHecYH6aB52i68 --output shanghaitech_test_frame_mask.zip
    
```

# 4. unzip, 7zip
### using unzip:
```bash
    pip install unzip
    unzip file.zip -d destination_folder
    unzip ped2.zip
    
    unzip avenue.zip
    unzip avenue_training_Entropy.zip
    
    unzip shanghaitech.zip
    unzip shanghaitech_training_KF_EM.zip
    unzip shanghaitech_training_KF_EN.zip
    unzip shanghaitech_training_KF_ES_30.zip
    unzip shanghaitech_training_KF_OF_30.zip
    unzip shanghaitech_test_frame_mask.zip
```


### using 7zip:
```bash
    sudo apt install p7zip-full
    
    # list the content of the zip
    7z l zipfile.zip
    
    # extract the contents of the zip
    7z x zipfile.zip
```


