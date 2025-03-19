import os, sys
import cv2, csv, natsort
import numpy as np
# ================================================
# for method_1
import peakutils  # pip install PeakUtils
from dataset.image_reader import CV2_imread
from keyframes_utils import convert_frame_to_grayscale, plot_metrics
from dataset.util import get_videos, delete_file, delete_files_in_dir
# from os_lib import delete_file, delete_files_in_dir
from dataset.csv_lib import writefile_csv_dic

# ================================================
# for method_DeepKmean
from feature_extractor import extract_features_CNN
from sklearn.cluster import KMeans
import shutil
import torch


# ================================================
# https://github.com/joelibaceta/video-keyframe-detector/blob/master/KeyFrameDetector/key_frame_detector.py
def extract_keyframes(source, dest, Thres=0.3, plotMetrics=False):
    """
    param source: full path to dataset (i.e: 'home/dataset/ped2_tiny/training')
    param dest: '/home/dataset/ped2_tiny/keyframes_0.3'
    """
    # 1. Tạo thư mục mới: 'home/dataset/ped2_tiny/keyframes_0.3'
    parent_path = os.path.dirname(source)  # 'home/dataset/ped2_tiny'
    current_dir_name = os.path.basename(source)  # 'training'
    dataset_name = os.path.basename(parent_path)  # 'ped2_tiny'
    dest = os.path.join(dest, 'keyframes_{:.1f}'.format(Thres))
    os.makedirs(dest, exist_ok=True)
    # print('parent_path = ', parent_path)
    # print('current_dir_name = ', current_dir_name)

    # 2. Tạo danh sách: self.videos chứa đường dẫn đến các frames của các video
    # self.videos là list các videos
    # self.videos[i] là list các đường dẫn đến các frames của video thứ i.
    videos, frame_count = get_videos(source)
    # print('len(videos) = ', len(videos))

    # 3. Duyệt các video và các frames trong 1 video
    for i in range(len(videos)):
        frames = videos[i]
        video_name = frames[0].split('/')[-2]

        print('video_name =', video_name)
        frame_id = -1
        lstfrm = []  # list of frame_id
        lstdiffMag = []  # list of  #non-zero pixel

        images = []  # list of grayframe
        full_color = []  # list of colorframe
        lastFrame = None
        for frame in frames:
            frame, H, W = CV2_imread(frame)
            grayframe, blur_gray = convert_frame_to_grayscale(frame)

            images.append(grayframe)
            full_color.append(frame)

            frame_id += 1
            lstfrm.append(frame_id)
            if frame_id == 0:
                lastFrame = blur_gray

            diff = cv2.subtract(blur_gray, lastFrame)
            diffMag = cv2.countNonZero(diff)  # đếm số lượng pixel khác không
            lstdiffMag.append(diffMag)

            lastFrame = blur_gray

        y = np.array(lstdiffMag)
        base = peakutils.baseline(y, 2)
        indices = peakutils.indexes(y - base, Thres, min_dist=1)

        if (plotMetrics):
            plot_metrics(indices, lstfrm, lstdiffMag)

        # make keyframe dir
        keyframe_dir = os.path.join(dest, video_name)  # '../01', '../02'
        os.makedirs(keyframe_dir, exist_ok=True)
        print('keyframe_dir = ', keyframe_dir)

        # make .csv files
        path2file = os.path.join(dest, '{}_{}_{:.1f}.csv'.format(dataset_name, video_name, Thres))

        delete_files_in_dir(keyframe_dir)
        delete_file(path2file)
        for id, x in enumerate(indices):  # list of keyframe indices
            cv2.imwrite(os.path.join(keyframe_dir, '{:03d}.jpg'.format(x)), full_color[x])
            fieldnames = ['id', 'name']
            name = '{:03d}.jpg'.format(x)
            row = {'id': str(id), 'name': name}
            if id == 0:
                writefile_csv_dic(path2file, 'w', fieldnames, row)
            else:
                writefile_csv_dic(path2file, 'a', fieldnames, row)


def extract_DeepKmean(source, dest, frame_percent=0.1):
    """
    param source: full path to dataset (i.e: 'home/dataset/ped2_tiny/training')
    param dest: '/home/asus/DATA/VAD/dataset/ped2_tiny'
    param keyframe_percent: the percent of frames to extract
    """
    if frame_percent >= 1.0 or frame_percent <= 0.0:
        print('frame_percent =', frame_percent)
        print('Warning: frame_perent must be in range of (0.0, 1.0) !!!')
        return
    # 1. Tạo thư mục mới: 'home/dataset/ped2_tiny/keyframes_FramePercent'
    parent_path = os.path.dir_name(source)  # 'home/dataset/ped2_tiny'
    current_dir_name = os.path.basename(source)  # 'training'
    dataset_name = os.path.basename(parent_path)  # 'ped2_tiny'

    dest = os.path.join(dest, current_dir_name + '_keyframes_DeepKmean_{:.02f}'.format(frame_percent))
    os.makedirs(dest, exist_ok=True)
    # print('parent_path = ', parent_path)
    # print('current_dir_name = ', current_dir_name)

    # 2. Tạo danh sách self.videos chứa đường dẫn đến các frames của các video
    # self.videos là list các videos
    # self.videos[i] là list các đường dẫn đến các frames của video thứ i.
    videos, frame_count = get_videos(source)
    # print('len(videos) = ', len(videos))

    # 3. Duyệt các video, các frames trong video
    for i in range(len(videos)):
        frames = videos[i]
        keyframe_count = int(len(frames) * frame_percent)
        video_name = frames[0].split('/')[-2]

        # make keyframe dir
        keyframe_dir = os.path.join(dest, video_name)  # '../01', '../02'
        os.makedirs(keyframe_dir, exist_ok=True)

        # 3.1. Extract features from frames (full_path) in a video
        frame_features = []
        for frame in frames:
            features = extract_features_CNN(frame, 'efficientnet_b0')  # torch.Size([1, xxxx])
            frame_features.append(features)

        # 3.2. Convert frame features to a NumPy array
        frame_features = torch.cat(frame_features, dim=0)
        print(f'frame_features.shape = {frame_features.shape}')

        # Use K-Means clustering to select keyframes
        kmeans = KMeans(n_clusters=keyframe_count, random_state=0).fit(frame_features)

        # An array of cluster labels for each data point
        # skipframe: (1, 3, 5, 7, 9) sf = 2
        # keyframe: (0, 3, 7, 9, 11)
        labels = kmeans.labels_  # ([0], 0, 0, [1], 1, 1, 1, [0], 0, [2], 2, 2, 2)
        print(f'labels = {labels}')

        # Get the indices of keyframes
        key_labels = [0]
        pre_label = labels[0]
        keyframes = [frames[0]]  # Initialize keyframes list
        for i, label in enumerate(labels):
            if label == pre_label:
                continue
            else:
                key_labels.append(i)
                pre_label = label
                keyframes.append(frames[i])

        print(f'key_labels = {key_labels}')

        # make .csv files
        path2file = os.path.join(dest, '{}_{}_{:03d}.csv'.format(dataset_name, video_name, keyframe_count))

        delete_files_in_dir(keyframe_dir)
        delete_file(path2file)

        for id, keyframe in enumerate(keyframes):
            file_name = keyframe.split('/')[-1]
            dest_file = os.path.join(keyframe_dir, file_name)
            shutil.copy(keyframe, dest_file)

            fieldnames = ['id', 'name']
            row = {'id': str(id), 'name': file_name}
            if id == 0:
                writefile_csv_dic(path2file, 'w', fieldnames, row)
            else:
                writefile_csv_dic(path2file, 'a', fieldnames, row)


import argparse


def main_extract_keyframes():
    parser = argparse.ArgumentParser()

    parser.add_argument('--source', type=str, default='/home/dataset/ped2_tiny/training', help='source file')
    parser.add_argument('--dest', type=str, default='/home/dataset/ped2_tiny', help='destination folder')
    parser.add_argument('--Thres', type=float, default=0.5, help='Threshold of the image difference')

    args = parser.parse_args()
    extract_keyframes(args.source, args.dest, float(args.Thres))


def main_extract_DeepKmean():
    parser = argparse.ArgumentParser()

    parser.add_argument('--source', type=str, default='/home/dataset/shanghaitech/training', help='source file')
    parser.add_argument('--dest', type=str, default='/home/dataset/shanghaitech', help='destination folder')
    parser.add_argument('--frame_percent', type=float, default=0.3, help=' The percent of frames to extract')

    args = parser.parse_args()
    extract_DeepKmean(args.source, args.dest, args.frame_percent)


# call this in terminal: python dataset/KF_KMean_CNN.py
if __name__ == '__main__':
    # main_extract_keyframes()
    main_extract_DeepKmean()
