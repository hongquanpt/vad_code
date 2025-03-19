# 1. flownet
- flownet là mô hình xấp xỉ (dự đoán) các đặc trưng optialflow
- flownet sử dụng một mạng học sâu với các lớp tích chập và kết hợp các lớp fully connected.
- Ưu điểm: Mô hình này tập trung vào việc dự đoán optical flow từ các cặp ảnh đầu vào.
- Nhược điểm: không giữ lại thông tin về thời gian.
- It is cloned from HF2VAD: A Hybrid Video Anomaly Detection Framework via Memory-Augmented Flow Reconstruction and Flow-Guided Frame Prediction
- Link download: 
  - https://github.com/LiUzHiAn/hf2vad (Pytorch)
    - `pre_process/flownet_networks`: sử dụng trong file `keyframes_op.py: extract_optical_flow()` 
  - https://github.com/feiyuhuahuo/Anomaly_Prediction (Pytorch)
    - `models/flownet2`: sử dụng trong file `loss.py: extract_flow_net2()`
    - `models/liteFlownet`: đang bị lỗi file `correlation.py`
  - https://github.com/open-mmlab/mmflow
  - https://github.com/sniklaus/pytorch-liteflownet
  - https://github.com/twhui/LiteFlowNet (Tensorflow) a series of 3 versions of LiteFlowNet of author
  - https://github.com/ltkong218/FastFlowNet (Pytorch)
    

# 2. RAFT (Recurrent All-Pairs Field Transforms)
- RAFT cũng giống flownet, nó là mô hình xấp xỉ (dự đoán) các đặc trưng optialflow
- RAFT sử dụng mô hình học sâu với các lớp tích chập và kết hợp mô hình RNN (recurrent neural network), sử dụng cấu trúc đặc biệt để xử lý thời gian.
- Ưu điểm: Có khả năng mô hình hóa các hiện tượng phức tạp hơn, mô hình hóa tốt trên cả thời gian và không gian.
- Nhược điểm: yêu cầu tài nguyên tính toán cao.
- Link download: https://github.com/princeton-vl/RAFT

# 3. pytorch_yolov5
- Yolov5 là mô hình object detection một bước, có tốc độ xử lý thời gian thực.
- Link download: https://github.com/ultralytics/yolov5/releases/tag/v7.0
- Có thể tải về các pretrained model như: yolov5s.pt từ link trên.

# 4. Neptune.ai
- Getting started: https://docs.neptune.ai/usage/quickstart/
- What you can log: https://docs.neptune.ai/logging/what_you_can_log/
  