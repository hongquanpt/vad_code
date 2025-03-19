import argparse,sys,os,torch,cv2,pickle,time
from PIL import Image
import numpy as np
from torchvision import transforms as T
from torchvision import utils
from sklearn.metrics import roc_auc_score
import natsort
import torch.nn as nn

from .AE_model import SI_DAE, DI_DAE
from libs.pytorch_yolov5.detect_testing import run

def load_checkpoint(logger, model, checkpoint_filepath):
    try:
        checkpoint = torch.load(checkpoint_filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f'Success: load_checkpoint: {checkpoint_filepath} \n')
        return model
    except:
        logger.info(f'Error: load_checkpoint: {checkpoint_filepath} !!!\n')
        return None

def moving_avg(X, belta=0.25):
    Y = X.copy()
    for i in range(X.shape[0]):
        if i >= 1:
            Y[i] = belta*Y[i] + (1-belta)*Y[i-1]
        else:
            pass
    return Y

def calculate_AUROC(MSE_FEA_List, score_lambda, ground_th, use_slide_avg=False):
    prediction=[]
    for (psnr1, psnr2, likelihood1, likelihood2) in MSE_FEA_List:
        max_score = -50
        for i in range(len(psnr1)):
            score = 100-np.dot(np.array([psnr1[i], psnr2[i], likelihood1[i], likelihood2[i]]), 
                               score_lambda)
            if score > max_score:
                max_score = score
        prediction.append(max_score)
    prediction = np.array(prediction)
    if use_slide_avg:
        prediction = moving_avg(prediction)
    prediction -= np.min(prediction)
    prediction /= max(1e-5,np.max(prediction))
    
    return roc_auc_score(ground_th,prediction), prediction

def grid_search(MSE_FEA_List,ground_th,search_step_size=0.1):
    max_lambda = None
    max_roc = -50
    prediction = None
    for i in range(int(1/search_step_size)+1):
        lambda1 = search_step_size*i
        for j in range(int((1-lambda1)/search_step_size)+1):
            lambda2 = search_step_size*j
            for k in range(int((1-lambda1-lambda2)/search_step_size)+1):
                lambda3 = search_step_size*k
                lambda4 = 1-lambda1-lambda2-lambda3
                temp = np.array([lambda1,lambda2,lambda3,lambda4])
                roc, prediction = calculate_AUROC(MSE_FEA_List,temp,ground_th)
                if max_roc < roc:
                    max_roc = roc
                    max_lambda = temp
    return max_roc, max_lambda, prediction

def evaluate(cfg, device, logger, visualization_dirpath):
    root = os.path.join(os.getcwd(), 'dataset') # '/home/dataset'
    dataset_path = os.path.join(root, cfg.DATASET.name, cfg.DATASET.test.testset)
    label_path = os.path.join(root, cfg.DATASET.name, cfg.DATASET.test.label_filename)
    
    parent_path = os.path.dirname(dataset_path) # 'home/anhnam/Data/VAD/dataset/ped2'
    current_dirname = os.path.basename(dataset_path) # 'testing'
        
    dirs = [d for d in os.listdir(dataset_path) if not d.startswith('.')]
    videos_names = natsort.natsorted(dirs)
    
    #================================================
    # Initialize
    #================================================
    # load Static Image DAE model
    simg_dae_model = SI_DAE()
    #simg_dae_model = nn.DataParallel(simg_dae_model, device_ids=gpus).cuda()
    simg_dae_model = simg_dae_model.to(device)
    simg_dae_model = load_checkpoint(logger, simg_dae_model, cfg.TEST.checkpoint_DAE_Img)
    simg_dae_model.eval()

    # load Dynamic Image DAE model
    dimg_dae_model = DI_DAE()
    #dimg_dae_model = nn.DataParallel(dimg_dae_model, device_ids=gpus).cuda()
    dimg_dae_model = dimg_dae_model.to(device)
    dimg_dae_model = load_checkpoint(logger, dimg_dae_model, cfg.TEST.checkpoint_DAE_Dimg)
    dimg_dae_model.eval()

    # load GMM Image theta
    with open(cfg.TEST.checkpoint_GMM_Img, "rb") as f:    
        img_theta=pickle.load(f)
    # load GMM Dimage theta
    with open(cfg.TEST.checkpoint_GMM_Dimg, "rb") as f:    
        dimg_theta=pickle.load(f)
        
    #================================================
    #================================================
    MSE_FEA_List = []
    SI_feature_List = []
    DI_feature_List = []
    #logger.info(f'videos_names = {videos_names}')  # ['01', '02',.., '15', '16']
    for video_name in videos_names:
        # source (yolov5) = video_path = 'home/anhnam/Data/VAD/dataset/ped2'/testing/01'
        video_path = os.path.join(dataset_path, video_name)  
        
        #  call function run() in file detect_testing.py
        MSE_FEA,SI_feature,DI_feature = run(weights = cfg.DATASET.weights_yolov5,
                                            simg_dae_model=simg_dae_model,
                                            dimg_dae_model=dimg_dae_model,
                                            img_theta=img_theta,
                                            dimg_theta=dimg_theta,
                                            source = video_path,
                                            project = visualization_dirpath)
    
        MSE_FEA_List = MSE_FEA_List + MSE_FEA
        SI_feature_List = SI_feature_List + SI_feature
        DI_feature_List = DI_feature_List + DI_feature
    
    # save to file
    with open(os.path.join(visualization_dirpath,"MSE_FEA_List.pkl"), "wb") as f:   
        pickle.dump(MSE_FEA_List, f)
    with open(os.path.join(visualization_dirpath,"SI_feature_List.pkl"), "wb") as f:   
        pickle.dump(SI_feature_List, f)
    with open(os.path.join(visualization_dirpath,"DI_feature_List.pkl"), "wb") as f:   
        pickle.dump(DI_feature_List, f)
        
    # load label of ped2
    logger.info(f'len(MSE_FEA_List) = {len(MSE_FEA_List)}')
    arr = np.load(label_path)
    ground_th = arr[0][0:len(MSE_FEA_List)]
    
    # AUC of 4 cases
    logger.info(f"SI+DAE AUROC: {calculate_AUROC(MSE_FEA_List, np.array([1,0,0,0]), ground_th)[0]}")
    logger.info(f"DI+DAE AUROC: {calculate_AUROC(MSE_FEA_List, np.array([0,1,0,0]), ground_th)[0]}")
    logger.info(f"SI+GMM AUROC: {calculate_AUROC(MSE_FEA_List, np.array([0,0,1,0]), ground_th)[0]}")
    logger.info(f"DI+GMM AUROC: {calculate_AUROC(MSE_FEA_List, np.array([0,0,0,1]), ground_th)[0]}")

    # grid search to find the 4 best lamdas
    max_roc,best_lambda,prediction=grid_search(MSE_FEA_List, ground_th)
    logger.info(f"max AUROC:{max_roc}")
    logger.info(f"best lambda: {best_lambda[0]}, {best_lambda[1]}, {best_lambda[2]}, {best_lambda[3]}") 
    logger.info(f"max AUROC after filter: {calculate_AUROC(MSE_FEA_List, best_lambda, ground_th, use_slide_avg=True)[0]}")
    '''
    (*) ped2_tiny:
        2023-11-09-22:21:len(MSE_FEA_List) = 360
        2023-11-09-22:21:SI+DAE AUROC: 0.9305888286470811
        2023-11-09-22:21:DI+DAE AUROC: 0.864298323036187
        2023-11-09-22:21:SI+GMM AUROC: 0.8464254192409532
        2023-11-09-22:21:DI+GMM AUROC: 0.9178697516076156
        2023-11-09-22:21:max AUROC:0.9445530197957382
        2023-11-09-22:21:best lambda: 0.8, 0.0, 0.0, 0.19999999999999996
        2023-11-09-22:21:max AUROC after filter: 0.9662400706090027
    '''
    '''
    (*) ped2:
    2023-11-10-00:32:len(MSE_FEA_List) = 2010
    2023-11-10-00:32:SI+DAE AUROC: 0.8438757174274526
    2023-11-10-00:32:DI+DAE AUROC: 0.8470530158772729
    2023-11-10-00:32:SI+GMM AUROC: 0.7417194121117846
    2023-11-10-00:32:DI+GMM AUROC: 0.8497391782438449
    2023-11-10-00:32:max AUROC:0.8635496567076114
    2023-11-10-00:32:best lambda: 0.4, 0.1, 0.0, 0.5
    2023-11-10-00:32:max AUROC after filter: 0.901202864345867
    '''
