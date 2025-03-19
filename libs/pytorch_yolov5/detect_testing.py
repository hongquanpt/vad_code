# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""
import torch
import cv2
import os
import sys
import numpy as np
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, increment_path,
                           non_max_suppression,
                           scale_boxes)
from utils.plots import Annotator
from utils.torch_utils import select_device
from PIL import Image
from torchvision import transforms as transforms
from scipy.stats import multivariate_normal


def ae_preprocess(img_cv2):
    """
    Preprocess images in format of DAE (64x64)
    """
    img_PIL = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    img = img_PIL.convert('L')
    temp = int((img.size[1] - img.size[0]) / 2)
    arround = 0
    pad = (temp + arround, 0 + arround) if temp >= 0 else (0 + arround, arround - temp)
    # Image.ANTIALIAS was removed in Pillow 10.0.0 
    # Now you need to use: PIL.Image.LANCZOS or PIL.Image.Resampling.LANCZOS. 
    transform = transforms.Compose([transforms.Pad(pad, fill=0), transforms.Resize((64, 64), Image.LANCZOS),
                                    transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])
    img_PIL = transform(img)
    img_tensor = img_PIL[np.newaxis, :]
    return img_tensor.to('cuda')


def calcu_proba(theta, test_x):
    k = theta["pi"].shape[0]
    q = np.zeros((k, 1))
    for i in range(k):
        q[i, :] = theta['pi'][i] * multivariate_normal.pdf(test_x,
                                                           mean=theta['miu'][i],
                                                           cov=theta['sigma'][i, ...])
    return np.log(1e-5 + q.sum())


def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        simg_dae_model=None,
        dimg_dae_model=None,
        img_theta=None,
        dimg_theta=None,
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        min_boxsize=10,
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(480, 480),  # inference size (height, width)
        conf_thres=0.3,  # confidence threshold
        iou_thres=0.3,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=True,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    # Run inference
    MSE_FEA = []
    SI_feature = []
    DI_feature = []

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            # source = 'home/anhnam/Data/VAD/dataset/ped2/testing/01'
            # Dimg_path = 'home/anhnam/Data/VAD/dataset/ped2/testing_DI/01/000.jpg'
            Dimg_path = os.path.join(os.path.dirname(source) + "_DI", os.path.basename(source), p.name)
            Dimg = cv2.imread(Dimg_path)

            # LOGGER.info(f'save_crop = {save_crop}')
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                patch_index = 0
                psnr1 = []
                psnr2 = []
                likelihood1 = []
                likelihood2 = []
                for *xyxy, conf, cls in reversed(det):  # for each detected object
                    patch_index += 1
                    xmin, ymin, xmax, ymax = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                    if (xmax - xmin) * (ymax - ymin) > min_boxsize:
                        img_in = ae_preprocess(im0[ymin:ymax, xmin:xmax, :])
                        dimg_in = ae_preprocess(Dimg[ymin:ymax, xmin:xmax, :])

                        # static img DAE model, dynamic image DAE model
                        recon1, fea1, mse1, _ = simg_dae_model(img_in)
                        recon2, fea2, mse2, _ = dimg_dae_model(dimg_in)

                        # calculate PSNR1, PSNR2 from MSE1, MSE2
                        psnr1.append(10 * np.log10(255 / mse1.item()))
                        psnr2.append(10 * np.log10(255 / mse2.item()))

                        # calcuate Likelihood1, Likelihood2 from (GMM) img_theta, (GMM) dimg_theta
                        fea1 = fea1.flatten().cpu().detach().numpy()
                        fea2 = fea2.flatten().cpu().detach().numpy()
                        likelihood1.append(calcu_proba(img_theta, fea1))
                        likelihood2.append(calcu_proba(dimg_theta, fea2))

                        # static image feature, dynamic image feature
                        SI_feature.append(fea1)
                        DI_feature.append(fea2)
                MSE_FEA.append([psnr1, psnr2, likelihood1, likelihood2])
            else:
                MSE_FEA.append([[], None, None, None])

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
    return MSE_FEA, SI_feature, DI_feature
