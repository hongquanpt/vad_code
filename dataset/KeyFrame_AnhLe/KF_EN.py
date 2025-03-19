import os
import shutil

import random
import cv2
import math
import imageio
import torch
from sklearn.metrics.cluster import entropy
from dataset.util import get_videos, delete_file, delete_files_in_dir
from dataset.csv_lib import writefile_csv_dic
from skimage.metrics import structural_similarity as ssim
import argparse

# MAX_NUMBER_OF_FRAMES = 100 # numFrames

# Sub Entropy.
SubEn = []
medium = 0


def getSSIM2Image(source, i, j):
    im1 = cv2.imread(os.path.join(source, '{:04d}.jpg'.format(i + 1)), 0)
    im2 = cv2.imread(os.path.join(source, '{:04d}.jpg'.format(j + 1)), 0)
    ED = ssim(im1, im2)
    return ED


# Calculate Average Entropy Difference for a chromosome.
def getSSIM(source, numFrames):
    distance_sum = 0
    global SubEn
    SubEn[:] = []
    for i in range(0, numFrames - 1):
        distance = getSSIM2Image(source, i, i + 1)
        distance_sum += distance
        SubEn.append(distance)

    # print('sub=', SubEn)
    global medium
    medium = distance_sum / len(SubEn)
    # print('Medium=', medium)


# def extract_keyframes(source, dest):
def extract_keyframes(dataset):
    source = os.path.join(os.getcwd(), 'dataset', dataset, 'training')
    dest = os.path.join(os.getcwd(), 'dataset', dataset)

    parent_path = os.path.dirname(source)  # 'home/dataset/ped2_tiny'
    current_dirname = os.path.basename(source)  # 'training'
    dataset_name = os.path.basename(parent_path)  # 'ped2_tiny'

    dest = os.path.join(dest, current_dirname + '_keyframes_no_adjacent_medium_SSIM')
    os.makedirs(dest, exist_ok=True)

    global medium

    videos, frame_count = get_videos(source)
    for i in range(len(videos)):
        frames = videos[i]
        numFrames = len(frames)

        split_char = '/'
        num = len(frames[0].split('/'))
        if num == 1:
            split_char = '\\'
        video_name = frames[0].split(split_char)[-2]

        print('video_name =', video_name)
        print('Number of frame =', numFrames)

        # make keyframe dir
        keyframe_dir = os.path.join(dest, video_name)  # '../01', '../02'
        os.makedirs(keyframe_dir, exist_ok=True)
        print('keyframe_dir = ', keyframe_dir)

        # make .csv files
        path2file = os.path.join(dest, '{}_{}.csv'.format(dataset_name, video_name))
        print('path2file = ', path2file)
        delete_files_in_dir(keyframe_dir)
        delete_file(path2file)
        sourceEvo = os.path.join(source, video_name)
        print('sourceEvo=', sourceEvo)

        # start
        # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # starter.record()
        getSSIM(sourceEvo, numFrames)
        print('Medium Co=', medium)

        frame_start = 0
        frame_end = frame_start + 1

        while (frame_start < (numFrames - 1)) and (frame_end < numFrames):
            if getSSIM2Image(sourceEvo, frame_start, frame_end) <= medium:
                # print('KC=', getAED2Image(sourceEvo, frame_start, frame_end))
                keyframe = os.path.join(sourceEvo, '{:04d}.jpg'.format(frame_start + 1))
                file_name = keyframe.split(split_char)[-1]
                dest_file = os.path.join(keyframe_dir, file_name)
                shutil.copy(keyframe, dest_file)
                fieldnames = ['id', 'name']
                row = {'id': str(frame_start), 'name': file_name}
                if frame_start == 0:
                    writefile_csv_dic(path2file, 'w', fieldnames, row)
                else:
                    writefile_csv_dic(path2file, 'a', fieldnames, row)

                frame_start = frame_end
                frame_end = frame_start + 1
            else:
                frame_end = frame_end + 1

        # imageio.mimsave(keyframe_dir + '{}_{}_{:03d}.gif'.format(dataset_name, video_name, numKeyFrames),
        #                 images_for_gif)
    # ender.record()

    # torch.cuda.synchronize()  # # Waits for everything to finish running
    # inference_time = starter.elapsed_time(ender) * 1e-3  # milisecond to second
    # print(f'Inference time of a video: {inference_time} seconds')


def main_extract_keyframes():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='shanghaitech', help='dataset name')
    args = parser.parse_args()
    extract_keyframes(args.dataset)


# in terminal: python KF_EN.py
if __name__ == '__main__':
    main_extract_keyframes()
