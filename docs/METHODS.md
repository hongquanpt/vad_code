# 1. MNAD
- Paper: 2020- Learning Memory-guided Normality for Anomaly Detection (MNAD)
- Link: 
  - [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Park_Learning_Memory-Guided_Normality_for_Anomaly_Detection_CVPR_2020_paper.pdf)
  - [Code](https://github.com/cvlab-yonsei/MNAD)
- Model: `model/MNAD/MNADModel.py` 
    - Model: MNAD = AE + Memory (Prediction)
    - train_set: only Normal samples
      - get_train_dataset(cfg) with sample_type = 'clips'
      - clip_mode = 'cat'
      - clip_dim = 0
    - test_set: get_test_dataset(cfg) with sample_type = 'clips_by_video'
    - Input: 4 frames ([bz=1, 4*3, 256, 256])
    - Output: to predict the 5th frame ([bz=1, 3, 256, 256])
    - Trạng thái: đã trained + tested 
# 2. PAMAE
- Paper: 2022- An integration of Pseudo Anomalies and Memory Augmented Autoencoder for Video Anomaly Detection (PA-MAE) 
- Link: 
  - [Paper](https://dl.acm.org/doi/abs/10.1145/3568562.3568642)
- Model: `model/MNAD/PAMAEModel.py`
    - Model: PAMAE_Pred = AE + Memory (Prediction)
    - train_set: Normal samples + Pseudo Anomalies (skip_frames)
    - test_set: get_test_dataset(cfg) with sample_type = 'videos'
      - get_train_dataset(cfg) with sample_type = 'clips'
      - clip_mode = 'cat'
      - clip_dim = 0
    - Input: 4 frames ([bz=1, 4*3, 256, 256])
    - Output: to predict the 5th frame ([bz=1, 3, 256, 256])
    - Trạng thái: đã trained + tested

- PAMAEModel2.py:
    - Model: PAMAE_Pred = AE + Memory (Prediction)
    - train_set: Normal samples + Pseudo Anomalies (skip_frames, patch)
      - get_train_dataset(cfg) with sample_type = 'clips'
      - clip_mode = 'cat'
      - clip_dim = 0
    - test_set: get_test_dataset(cfg) with sample_type = 'videos'
    - Input: 4 frames ([bz=1, 4*3, 256, 256])
    - Output: to predict the 5th frame ([bz=1, 3, 256, 256])
    - main_PAMAE2.py:
          train_dataset, train_dataloader = get_train_dataset(cfg)
    - Trạng thái: đã trained + tested
    
- PAMAEModel3.py:
    - Model: PAMAE_Pred = AE + Memory (Prediction)
    - train_set: Normal samples + Pseudo Anomalies (list_augtype)
      - get_train_dataset(cfg) with sample_type = 'clips'
      - clip_mode = 'cat'
      - clip_dim = 0
    - test_set: get_test_dataset(cfg) with sample_type = 'videos'
    - Input: 4 frames ([bz=1, 4*3, 256, 256])
    - Output: to predict the 5th frame ([bz=1, 3, 256, 256])
    - main_PAMAE3.py:
          train_dataset, train_dataloader = get_train_dataset_aug(cfg)
        - augmode: 'normal_only' # 'normal_only', 'random', 'probability'
          augmode = 'normal_only' => augtype = 'normal_only', nghĩa là lấy các frame ra theo thứ tự liên tiếp
          augmode = 'random'  => augtype được lấy ngẫu nhiên từ list_augtype.
           augmode = 'probability'  => augtype được lấy theo tỷ lệ xác suất trong list_prob_weights.
        - list_augtype: ['normal_only', 'TMT', 'SRT', 'gaussian_noise', 'simplex_noise']
        - list_prob_weights: [0.0, 0.5, 0.5, 0.0, 0.0]
      => video.py
    - Trạng thái: đã trained + tested

- PAMAEModel4.py: giống PAMAEModel.py
    - Model: PAMAE_Pred = AE + Memory (Prediction)
    - train_set: Normal samples + Pseudo Anomalies (skip_frames)
      - get_train_dataset(cfg) with sample_type = 'clips'
      - clip_mode = 'cat'
      - clip_dim = 0
    - test_set: get_test_dataset(cfg) with sample_type = 'clips_by_video'
      Hàm train(): giống STEALModel khi thêm Pseudo Anomalies, khác PAMAEModel, PAMAEModel2, PAMAEModel3.
    - Input: 4 frames ([bz=1, 4*3, 256, 256])
    - Output: to predict the 5th frame ([bz=1, 3, 256, 256])
      Hàm test(): giống STEALModel.py, khác PAMAEModel, PAMAEModel2, PAMAEModel3.

- PAMAEModel5.py: giống PAMAEModel4.py
    - Model: PAMAE_Pred = AE + Memory (Prediction)
    - train_set: Normal samples + Pseudo Anomalies (keyframes) extracted by DeepKmean
      - get_train_dataset(cfg) with sample_type = 'clips'
      - clip_mode = 'cat'
      - clip_dim = 0
    - test_set: get_test_dataset(cfg) with sample_type = 'clips_by_video'
      Hàm train(): giống PAMAEModel4 khi thêm Pseudo Anomalies, khác nhưng sử dụng keyframes_iter.
    - Thử nghiệm với tỷ lệ keyframes là 20%, 30%, 40%.
# 3. ASTNet
- 2022- Attention-based residual autoencoder for video anomaly detection
- Link:
  - [Paper](https://link.springer.com/article/10.1007/s10489-022-03613-1)
  - [Code](https://github.com/vt-le/astnet)
- ASTNetModel.py
    - Thuật toán ASTNet = AE + Attention (Prediction)
    - train_set: 
      - get_train_dataset(cfg) with sample_type = 'clips'
      - clip_mode = 'list'
      - clip_dim = 0
    - test_set: get_test_dataset(cfg) with sample_type = 'clips_by_video'
    - Input: a list of 4 frames ([bz=1, num_channels=3, 256, 256])
    - Output: to predict the 5th frame ([bz=1, num_channels=3, 256, 256])
    - Đặc điểm:
        - frame_steps = 2: => số mẫu trong tập train giảm 1 nửa
        - epoch = 120, gấp đôi PAMAE
        - frame_step2 = 2 thì traininng time/epoch = 5 phút, 120 epochs = 11 giờ
    - Trạng thái: đã trained + tested (1 epoch)

# 4. FastAno
- 2022- FastAno_Fast_Anomaly_Detection_via_Spatio-Temporal_Patch_Transformation
- FastAno_Model.py
    - FastAno = Data Augmentation (TMT, SRT) + AE (Prediction)
    - train_set:
      - get_train_dataset(cfg) with sample_type = 'clips'
      - clip_mode = 'stack'
      - clip_dim = 0
    - test_set: get_test_dataset(cfg) with sample_type = clips_by_video'
    - Input: a tensor of stacked 5 frames, 1 channels
          ([bz, num_frames=5, num_channels=1, 256, 256])
    - Output: to predict the 6th frame
          ([bz, num_frames=1, num_channels=1, 256, 256])
      Hàm test() mới, khác ASTNetModel, PAMAEModel, PAMAEModel2,PAMAEModel3.
    - Trạng thái: ok
 
# 5. STEAL
- 2021- Synthetic Temporal Anomaly Guided End-to-End Video Anomaly Detection
- STEALModel.py: 
    - Model: STEAL = AE + Pseudo Abnormal Module (Reconstruction)
    - train_set:
      - get_train_dataset(cfg) with sample_type = 'clips'
      - clip_mode = 'stack'
      - clip_dim = 1
    - test_set: get_test_dataset(cfg) with sample_type = 'clips_by_video'
    - Input: a tensor of stacked 16 frames, 1 channels
          ([bz=1, num_channels=1, num_frames=16, 256, 256])
    - Output: 16 frame, 1 channels
          ([bz=1, num_channels=1, num_frames=16, 256, 256])
      Hàm test() mới, khác ASTNetModel, PAMAEModel, PAMAEModel2,PAMAEModel3.
    - Trạng thái: đã trained + tested
 
 
# 6. GMM_DAE
- 2020-Video Anomaly Detection by Estimating Likelihood of Representations
- Bước 1: Tiền xử lý, sử dụng yolov5 (pretrained file: yolov5s.pt) để trích xuất các đối tượng người trong các frames từ 2 tập trainset và testset.
    - main_GMM_DAE.py: def preprocess_training(cfg)
    - main_GMM_DAE.py: def preprocess_testing(cfg)
- Bước 2: Huấn luyện mạng DAE (denoising AE): model/GMM_DAE/DAEModel.py
- Bước 3: Huấn luyện mạng GMM (Gaussian Mixture Model): model/GMM_DAE/GMMModel.py
- Bước 4: Phát hiện frame bất thường trong tập testset:  model/GMM_DAE/evaluate.py

# 7. AMC
- 2019- Anomaly Detection in Video Sequence with Appearance-Motion Correspondence
- Not implement

# 8. Diffusion2
- Source: https://www.youtube.com/watch?v=a4Yfz2FxXiY
- DiffusionModel2.py
    - Test thử mô hình Diffusion (dựa trên tutorial trên Youtube)
    - Trạng thái: Chưa implement model này!
    - Method: Reconstruction
    - Dự định thử nghiệm trên tập: cifar10-64, ped2

# 9. VideoVAE
- 2018-ECCV-Probabilistic Video Generation using Holistic Attribute Control
- VideoVAEModel.py
    - áp dụng cho bài toán Video Generation
    - Method: Reconstruction
    - Trạng thái: mới chỉ cài đặt để chạy được phần training: Reconstruction

# 10. ConvVRNN
- 2019-Future Frame Prediction Using Convolutional VRNN for Anomaly Detection
- ConvVRNNModel.py
    - ConvVRNN = 
    - Method: Prediction
    - Trạng thái: đã train và test
    - Kết quả: AUC = 23.6%

# 11. ConvAE
- ConvVAEModel2.py: cfg.MODEL.name = ConvVAE_Recon
    - Trạng thái: Chưa implement model này!

- ConvAE_SVMModel.py: cfg.MODEL.name = ConvAE_SVMModel
    - Trạng thái: đã train và test
    - Dataset: ped2
    - Method: Reconstruction
    - Input: 1 frame (3, 256, 256)
    - Training time: 97.38901424407959 seconds/epoch
    - Trained model: final.pth = 150 MB
    - Trained OCSVM: ocsvm_0.1.pkl = 1.1 GB, ocsvm_0.5.pkl (too large > 4 GB, can not save)
    - Kết quả: F1 Score = 0.119; auc_score = 0.4474886686692056

# 12. ConvLSTM_AE
- 2017-Abnormal Event Detection in Videos using Spatiotemporal Autoencoder
- ConvLSTM_AEModel.py
    - ConvLSTM_AE = ConvLSTM + AE (Reconstruction)
    - Kết quả rất tệ.
    - Input: (10 grayscale frames ([bz=1, num_frames=10, num_channels=1, 227, 227]))
    - xem file init_util.py: cách khởi tạo trọng số cho mạng
    - tham khảo github:
        - video_anomaly_detection_pytorch-master
        - spatio-temporal-autoencoder-for-videos-main
        
