import os
import cv2
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
# from feature_extractor import extract_features_CNN
from sklearn.cluster import KMeans
import shutil
import torch
from skimage.metrics import structural_similarity as ssim
import argparse
from sklearn.metrics.cluster import entropy


def extract_Entropy(dataset, percent=0.3):
    source = os.path.join(os.getcwd(), 'dataset', dataset, 'training')
    dest = os.path.join(os.getcwd(), 'dataset', dataset)

    parent_path = os.path.dirname(source)  # 'home/dataset/ped2_tiny'
    current_dirname = os.path.basename(source)  # 'training'
    dataset_name = os.path.basename(parent_path)  # 'ped2_tiny'

    print('parent_path = ', parent_path)
    print('current_dirname = ', current_dirname)
    print('dataset_name = ', dataset_name)

    dest_path = os.path.join(dest, current_dirname + '_KF_KMean_{:.02f}'.format(percent))
    os.makedirs(dest_path, exist_ok=True)

    # 2. Tạo danh sách: self.videos chứa đường dẫn đến các frames của các video
    # self.videos là list các videos
    # self.videos[i] là list các đường dẫn đến các frames của video thứ i.
    videos, frame_count = get_videos(source)
    print('len(videos) = ', len(videos))

    # for each video
    for i in range(0, len(videos)):
        frames = videos[i]
        keyframe_count = int(len(frames) * percent)
        split_char = '/'
        num = len(frames[0].split('/'))
        if num == 1:
            split_char = '\\'

        video_name = frames[0].split(split_char)[-2]

        # make keyframe dir
        keyframe_dir = os.path.join(dest_path, video_name)  # '../01', '../02'
        os.makedirs(keyframe_dir, exist_ok=True)

        # for each frame (full_path) in a video
        num_frames = len(frames)
        list_entropy = []  # a list of Matrices of a video
        # for j in range(0, num_frames - 1):
        for j in range(0, num_frames):
            frame1 = frames[j]

            curr_frame_name = frame1.split(split_char)[-1]  # '000.jpg'

            # (H,W,C)
            im1 = CV2_imread(frame1, flag=0)
            # frame2 = frames[j + 1]
            # im2 = CV2_imread(frame2, flag=0)
            # etp = ssim(im1, im2)

            etp = entropy(im1)
            list_entropy.append(etp)

        x = np.array(list_entropy).reshape(-1, 1)

        # Use K-Means clustering to select keyframes
        kmeans = KMeans(n_clusters=keyframe_count, random_state=0).fit(x)

        labels = kmeans.labels_  # ([0], 0, 0, [1], 1, 1, 1, [0], 0, [2], 2, 2, 2)
        print(f'labels = {labels}')

        # Get the indices of keyframes
        key_labels = [0]
        pre_label = labels[0]
        keyframes = [frames[0]]  # Initialize keyframes list
        for j, label in enumerate(labels):
            if label == pre_label:
                continue
            else:
                key_labels.append(j)
                pre_label = label
                keyframes.append(frames[j])

        print(f'key_labels = {key_labels}')

        # make .csv files
        path2file = os.path.join(dest, '{}_{}_{:03d}.csv'.format(dataset_name, video_name, keyframe_count))

        delete_files_in_dir(keyframe_dir)
        delete_file(path2file)

        for j, keyframe in enumerate(keyframes):
            file_name = keyframe.split(split_char)[-1]
            dest_file = os.path.join(keyframe_dir, file_name)
            shutil.copy(keyframe, dest_file)

            fieldnames = ['id', 'name']
            row = {'id': str(j), 'name': file_name}
            if j == 0:
                writefile_csv_dic(path2file, 'w', fieldnames, row)
            else:
                writefile_csv_dic(path2file, 'a', fieldnames, row)


def main_extract_Entropy():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='ped2', help='dataset name')
    parser.add_argument('--percent', type=float, default=0.3, help=' The percent of frames to extract')

    # parser.add_argument('--source', type=str, default='/home/dataset/ped2/training', help='source file')
    # parser.add_argument('--dest', type=str, default='/home/dataset/ped2', help='destination folder')
    # parser.add_argument('--percent', type=float, default=0.3, help=' The percent of frames to extract')

    args = parser.parse_args()
    # extract_Entropy(args.source, args.dest, args.percent)
    extract_Entropy(args.dataset, args.percent)


# in terminal: python dataset/KeyFrame_AnhLe/KF_KMean_Entropy.py
# in debug: dataset/KeyFrame_AnhLe/KF_KMean_Entropy.py
if __name__ == '__main__':
    main_extract_Entropy()
