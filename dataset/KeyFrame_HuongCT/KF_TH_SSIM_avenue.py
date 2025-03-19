import argparse
import os
import shutil

import cv2
import torch

from dataset.csv_lib import writefile_csv_dic
from dataset.util import get_videos, delete_file, delete_files_in_dir

from skimage.metrics import structural_similarity as ssim

# MAX_NUMBER_OF_FRAMES = 100 # numFrames

# Sub Entropy.
SubEn = []
SortedSub = []


def getSSIM2Image(source, i, j):
    im1 = cv2.imread(os.path.join(source, '{:03d}.jpg'.format(i)), 0)
    im2 = cv2.imread(os.path.join(source, '{:03d}.jpg'.format(j)), 0)
    ED = ssim(im1, im2)
    return ED

def getSSIM2Image4Num(source, i, j):
    im1 = cv2.imread(os.path.join(source, '{:04d}.jpg'.format(i)), 0)
    im2 = cv2.imread(os.path.join(source, '{:04d}.jpg'.format(j)), 0)
    ED = ssim(im1, im2)
    return ED
# Calculate Average Entropy Difference for a chromosome.
def getSSMIM(source, numFrames):
    global SubEn
    SubEn[:] = []
    for i in range(0, numFrames - 1):
        SubEn.append(getSSIM2Image(source, i, i + 1))

    global SortedSub
    SortedSub = []

    SortedSub = SubEn.copy()
    SortedSub.sort()
    #print('sub=', SubEn)
    #print("Sorted=", SortedSub)

# Calculate Average Entropy Difference for a chromosome.
def getSSMIM4Num(source, numFrames):
    global SubEn
    SubEn[:] = []
    for i in range(0, numFrames - 1):
        SubEn.append(getSSIM2Image4Num(source, i, i + 1))

    global SortedSub
    SortedSub = []

    SortedSub = SubEn.copy()
    SortedSub.sort()
   # print('sub=', SubEn)
   # print("Sorted=", SortedSub)


def extract_keyframes(source, dest, frame_percent=0.3):
    if frame_percent >= 1.0 or frame_percent <= 0.0:
        print('frame_percent =', frame_percent)
        print('Warning: frame_perent must be in range of (0.0, 1.0) !!!')
        return
    parent_path = os.path.dirname(source)  # 'home/dataset/ped2_tiny'
    current_dirname = os.path.basename(source)  # 'training'
    dataset_name = os.path.basename(parent_path)  # 'ped2_tiny'

    dest = os.path.join(dest, current_dirname + '_keyframes_Sorted_SSIM_{:.02f}'.format(frame_percent))
    os.makedirs(dest, exist_ok=True)
    global SortedSub
    print(f'source = ', source)
    videos, frame_count = get_videos(source)
    print('len(videos) = ', len(videos))
    # print('frame count = ', frame_count)
    for i in range(len(videos)):
        frames = videos[i]
        numFrames = len(frames)
        numKeyFrames = int(numFrames * frame_percent)
        video_name = frames[0].split('/')[-2]

        print('video_name =', video_name)
        print('Number of frame =', numFrames)
        print('numKeyFrames =', numKeyFrames)
        # make keyframe dir
        keyframe_dir = os.path.join(dest, video_name)  # '../01', '../02'
        os.makedirs(keyframe_dir, exist_ok=True)
        print('keyframe_dir = ', keyframe_dir)

        if video_name in {'05', '11', '12', '13', '14', '15', '16'}:
            # make .csv files
            path2file = os.path.join(dest, '{}_{}_{:03d}.csv'.format(dataset_name, video_name, numKeyFrames))
            print('path2file = ', path2file)
            delete_files_in_dir(keyframe_dir)
            delete_file(path2file)
            sourceEvo = os.path.join(source,  video_name)
            print('sourceEvo=', sourceEvo)

            # start
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            getSSMIM(sourceEvo, numFrames)

            #   print('SortedSub =', SortedSub)
            #   print('len=', len(SortedSub))
            threshold = SortedSub[int(numKeyFrames * 0.5)]
            print('threshold position=', int(numKeyFrames * 0.5))
            print('th=', threshold)

            # images_for_gif = []
            for frame_number in range(len(SubEn)):
                if SubEn[frame_number] <= threshold:
                    keyframe = os.path.join(sourceEvo, '{:03d}.jpg'.format(frame_number))
                    #   images_for_gif.append(imageio.v2.imread(keyframe))
                    file_name = keyframe.split('/')[-1]
                    dest_file = os.path.join(keyframe_dir, file_name)
                    shutil.copy(keyframe, dest_file)
                    fieldnames = ['id', 'name']
                    row = {'id': str(frame_number), 'name': file_name}
                    if frame_number == 0:
                        writefile_csv_dic(path2file, 'w', fieldnames, row)
                    else:
                        writefile_csv_dic(path2file, 'a', fieldnames, row)

                    keyframe = os.path.join(sourceEvo, '{:03d}.jpg'.format(frame_number + 1))
                    # images_for_gif.append(imageio.v2.imread(keyframe))
                    file_name = keyframe.split('/')[-1]
                    dest_file = os.path.join(keyframe_dir, file_name)
                    shutil.copy(keyframe, dest_file)
                    fieldnames = ['id', 'name']
                    row = {'id': str(frame_number + 1), 'name': file_name}
                    writefile_csv_dic(path2file, 'a', fieldnames, row)

            # imageio.mimsave(keyframe_dir + '{}_{}_{:03d}.gif'.format(dataset_name, video_name, numKeyFrames),
            #                 images_for_gif)
            ender.record()
        elif video_name in {'01', '02', '03', '04', '06', '07', '08', '09', '10'}:
            # make .csv files
            path2file = os.path.join(dest, '{}_{}_{:03d}.csv'.format(dataset_name, video_name, numKeyFrames))
            print('path2file = ', path2file)
            delete_files_in_dir(keyframe_dir)
            delete_file(path2file)
            sourceEvo = os.path.join(source,  video_name)
            print('sourceEvo=', sourceEvo)

            # start
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            getSSMIM4Num(sourceEvo, numFrames)

            #   print('SortedSub =', SortedSub)
            #   print('len=', len(SortedSub))
            threshold = SortedSub[int(numKeyFrames * 0.5)]
            print('threshold position=', int(numKeyFrames * 0.5))
            print('th=', threshold)

            # images_for_gif = []
            for frame_number in range(len(SubEn)):
                if SubEn[frame_number] <= threshold:
                    keyframe = os.path.join(sourceEvo, '{:04d}.jpg'.format(frame_number))
                    #   images_for_gif.append(imageio.v2.imread(keyframe))
                    file_name = keyframe.split('/')[-1]
                    dest_file = os.path.join(keyframe_dir, file_name)
                    shutil.copy(keyframe, dest_file)
                    fieldnames = ['id', 'name']
                    row = {'id': str(frame_number), 'name': file_name}
                    if frame_number == 0:
                        writefile_csv_dic(path2file, 'w', fieldnames, row)
                    else:
                        writefile_csv_dic(path2file, 'a', fieldnames, row)

                    keyframe = os.path.join(sourceEvo, '{:04d}.jpg'.format(frame_number + 1))
                    # images_for_gif.append(imageio.v2.imread(keyframe))
                    file_name = keyframe.split('/')[-1]
                    dest_file = os.path.join(keyframe_dir, file_name)
                    shutil.copy(keyframe, dest_file)
                    fieldnames = ['id', 'name']
                    row = {'id': str(frame_number + 1), 'name': file_name}
                    writefile_csv_dic(path2file, 'a', fieldnames, row)

            # imageio.mimsave(keyframe_dir + '{}_{}_{:03d}.gif'.format(dataset_name, video_name, numKeyFrames),
            #                 images_for_gif)
            ender.record()
        else:
            print('Not in=', video_name)

        torch.cuda.synchronize()  # # Waits for everything to finish running
        inference_time = starter.elapsed_time(ender) * 1e-3  # milisecond to second
        print(f'Inference time of a video: {inference_time} seconds')


def main_extract_keyframes():
    parser = argparse.ArgumentParser()

    source_dir = os.path.join('avenue', 'training')
    print('Source=', source_dir)
    dest_dir = os.path.join('avenue')
    
    parser.add_argument('--source', type=str, default=source_dir, help='source file')
    parser.add_argument('--dest', type=str, default=dest_dir, help='destination folder')

    parser.add_argument('--frame_percent', type=float, default=0.3, help=' The percent of frames to extract')

    args = parser.parse_args()
    extract_keyframes(args.source, args.dest, args.frame_percent)


if __name__ == '__main__':
    main_extract_keyframes()
