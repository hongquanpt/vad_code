import os, subprocess
import natsort

#==========================================================
# Yolo5: https://github.com/ultralytics/yolov5/releases
# DeepSORT: https://github.com/fengxingxiang/Yolov5_DeepSort_Pytorch/tree/master
#==========================================================
def detect_Objects_yolov5_training(dataset_path = 'dataset/ped2/training', 
                                   detect_file = 'libs/pytorch_yolov5/detect_training.py',
                                   weights = 'libs/pytorch_yolov5/weights/yolov5s.pt'):
    dataset_path = dataset_path 
    parent_path = os.path.dirname(dataset_path) # 'home/anhnam/Data/VAD/dataset/ped2'
    current_dirname = os.path.basename(dataset_path) # 'training'
        
    dirs = [d for d in os.listdir(dataset_path) if not d.startswith('.')]
    videos_names = natsort.natsorted(dirs)
    #print(f'videos_names = {videos_names}')  # ['01', '02',.., '15', '16']
    for video_name in videos_names:
        # source (yolov5) = video_path = 'home/anhnam/Data/VAD/dataset/ped2'/tranining/01'
        video_path = os.path.join(dataset_path, video_name)  
        crop_path_Img = os.path.join(parent_path, 
                                    current_dirname + '_Crop_Img', video_name) # 'training_Crop_Img/01
        crop_path_Dimg = os.path.join(parent_path, 
                                    current_dirname + '_Crop_Dimg', video_name) # 'training_Crop_Dimg/01
        detect_path = os.path.join(parent_path, current_dirname + '_Detect')
            
        sys_cmd = "python {} --weights {} --img {} --conf {} \
                        --source {} --crop_path_Img {} --crop_path_Dimg {} --save_crop \
                        --project {} --name {}".format(detect_file,
                                                        weights, 
                                                        640, 0.25,
                                                        video_path,
                                                        crop_path_Img,
                                                        crop_path_Dimg,
                                                        detect_path,
                                                        video_name)
        child = subprocess.Popen(sys_cmd, shell = True)
        child.wait()