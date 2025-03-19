'''
In terminal:
    conda activate pyanomaly
    python libs/pytorch_yolov5/weights/download.py
'''
# https://github.com/ultralytics/yolov5
import wget
"""
URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt"
response = wget.download(URL, "yolov5x.pt")
"""
URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt"
response = wget.download(URL, "yolov5l.pt")