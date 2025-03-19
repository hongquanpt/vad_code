import os
import sys
import shutil

import random
import cv2
import math
import imageio
import torch
from sklearn.metrics.cluster import entropy
from dataset.csv_lib import writefile_csv_dic
from dataset.util import get_videos, delete_file, delete_files_in_dir
from dataset.util import is_windows_path, is_linux_path

import argparse

# MAX_NUMBER_OF_FRAMES = 100 # numFrames


# Entropy matrix.
En = []

medium = 0


# Calculate Average Entropy Difference for a chromosome.
def getEntropy(source, numFrames):
    entropy_sum = 0
    global En
    En[:] = []

    for i in range(0, numFrames):
        im1 = cv2.imread(os.path.join(source, '{:03d}.jpg'.format(i)), 0)
        En.append(entropy(im1))
    # print('i=', i, 'En=', En(i))
    # print('En=', En)
    print('Len=', len(En))
    for i in range(len(En) - 1):
        entropy_sum += abs(En[i] - En[i + 1])
    global medium
    medium = entropy_sum / (len(En) - 1)
    print('Medium=', medium)


# print('Before=', SubEn)
# print('After=', SortedSub)

# Calculate Average Entropy Difference for a chromosome.
def getEntropy4Num(source, numFrames):
    entropy_sum = 0
    global En
    En[:] = []

    for i in range(0, numFrames):
        im1 = cv2.imread(os.path.join(source, '{:04d}.jpg'.format(i)), 0)
        En.append(entropy(im1))
    # print('i=', i, 'En=', En(i))
    # print('En=', En)
    print('Len=', len(En))
    for i in range(len(En) - 1):
        entropy_sum += abs(En[i] - En[i + 1])
    global medium
    medium = entropy_sum / (len(En) - 1)
    print('Medium=', medium)


def extract_keyframesAvenue(source, dest):
    parent_path = os.path.dirname(source)  # 'home/dataset/avenue'
    current_dirname = os.path.basename(source)  # 'training'
    dataset_name = os.path.basename(parent_path)  # 'avenue'

    # dest = os.path.join(dest, current_dirname + '_keyframes_no_adjacent_medium_Entropy')
    os.makedirs(dest, exist_ok=True)
    global medium
    print(f'source = ', source)
    videos, frame_count = get_videos(source)
    print('len(videos) = ', len(videos))
    # print('frame count = ', frame_count)
    for i in range(len(videos)):
        frames = videos[i]
        numFrames = len(frames)
        if is_windows_path(frames[0]):
            print("Windows path")
            video_name = frames[0].split('\\')[-2]
        elif is_linux_path(frames[0]):
            print("Linux path")
            video_name = frames[0].split('/')[-2]
        else:
            print('Invalid file path')
            return

        print('video_name =', video_name)
        print('Number of frame =', numFrames)

        # make keyframe dir
        keyframe_dir = os.path.join(dest, video_name)  # '../01', '../02'
        os.makedirs(keyframe_dir, exist_ok=True)
        print('keyframe_dir = ', keyframe_dir)

        if video_name in {'05', '11', '12', '13', '14', '15', '16', '22'}:
            # make .csv files
            path2file = os.path.join(dest, '{}_{}.csv'.format(dataset_name, video_name))
            print('path2file = ', path2file)
            delete_files_in_dir(keyframe_dir)
            delete_file(path2file)
            sourceEvo = os.path.join(source, video_name)
            print('sourceEvo=', sourceEvo)

            # start
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            getEntropy(sourceEvo, numFrames)

            print('medium=', medium)

            images_for_gif = []
            frame_start = 0
            frame_end = frame_start + 1

            while (frame_start < (numFrames - 1)) and (frame_end < numFrames):
                if abs(En[frame_end] - En[frame_start]) >= medium:
                    keyframe = os.path.join(sourceEvo, '{:03d}.jpg'.format(frame_start))
                    # images_for_gif.append(imageio.v2.imread(keyframe))

                    if is_windows_path(keyframe):
                        # print("Windows path")
                        file_name = keyframe.split('\\')[-1]
                    elif is_linux_path(frames[0]):
                        # print("Linux path")
                        file_name = keyframe.split('/')[-1]
                    else:
                        print('Invalid file path')
                        return

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

            # imageio.mimsave(keyframe_dir + '{}_{}.gif'.format(dataset_name, video_name), images_for_gif)
            ender.record()
        elif video_name in {'01', '02', '03', '04', '06', '07', '08', '09', '10', '17', '18', '19', '20', '21'}:
            # make .csv files
            path2file = os.path.join(dest, '{}_{}.csv'.format(dataset_name, video_name))
            print('path2file = ', path2file)
            delete_files_in_dir(keyframe_dir)
            delete_file(path2file)
            sourceEvo = os.path.join(source, video_name)
            print('sourceEvo=', sourceEvo)

            # start
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            getEntropy4Num(sourceEvo, numFrames)

            print('medium=', medium)

            images_for_gif = []
            frame_start = 0
            frame_end = frame_start + 1

            while (frame_start < (numFrames - 1)) and (frame_end < numFrames):
                if abs(En[frame_end] - En[frame_start]) >= medium:
                    keyframe = os.path.join(sourceEvo, '{:04d}.jpg'.format(frame_start))
                    # images_for_gif.append(imageio.v2.imread(keyframe))

                    if is_windows_path(keyframe):
                        # print("Windows path")
                        file_name = keyframe.split('\\')[-1]
                    elif is_linux_path(frames[0]):
                        # print("Linux path")
                        file_name = keyframe.split('/')[-1]
                    else:
                        print('Invalid file path')
                        return

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

            # imageio.mimsave(keyframe_dir + '{}_{}.gif'.format(dataset_name, video_name), images_for_gif)
            ender.record()
        else:
            print('Not in=', video_name)

        torch.cuda.synchronize()  # # Waits for everything to finish running
        inference_time = starter.elapsed_time(ender) * 1e-3  # milisecond to second
        print(f'Inference time of a video: {inference_time} seconds')


def main_extract_keyframesAvenue():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='avenue', help='dataset name')

    args = parser.parse_args()
    source_dir = os.path.join(os.getcwd(), 'dataset', args.dataset, 'training_22')
    dest_dir = os.path.join(os.getcwd(), 'dataset', args.dataset, 'training_KF_EN_22')

    extract_keyframesAvenue(source_dir, dest_dir)


def change_file_names(directory='path/to/image/directory'):
    # Lấy danh sách các file trong thư mục
    files = os.listdir(directory)

    # Lặp qua các file và đổi tên
    for i, file in enumerate(files):
        if file.endswith('.jpg') or file.endswith('.png'):
            new_filename = f"{i:03d}.{file.split('.')[-1]}"
            os.rename(os.path.join(directory, file), os.path.join(directory, new_filename))
            print('new_filename = ', new_filename)


# in terminal: python KF_NP_Entropy_avenue.py
if __name__ == '__main__':
    main_extract_keyframesAvenue()
    # source_dir = os.path.join(os.getcwd(), 'dataset', 'avenue', 'training_22', '22')
    # change_file_names(source_dir)
