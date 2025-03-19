from dataset.util import get_videos, delete_file, delete_files_in_dir
from dataset.util import is_windows_path, is_linux_path
from dataset.KeyFrame_AnhLe.DE_Entropy3 import evolution

import os
import argparse
import torch


def extract_keyframes(source, dest, numGeneration, popSize, frame_percent=0.3):
    if frame_percent >= 1.0 or frame_percent <= 0.0:
        print('frame_percent =', frame_percent)
        print('Warning: frame_parent must be in range of (0.0, 1.0) !!!')
        return

    parent_path = os.path.dirname(source)  # 'home/dataset/ped2_tiny'
    current_dirname = os.path.basename(source)  # 'training'
    dataset_name = os.path.basename(parent_path)  # 'ped2_tiny'
    print('current_dirname = ', current_dirname)

    # 1. Tạo thư mục mới: để lưu trữ kết quả keyframes
    dest = os.path.join(dest, current_dirname + '_keyframes_Entropy_{:.02f}'.format(frame_percent))
    os.makedirs(dest, exist_ok=True)
    print('dest = ', dest)

    # 2. Tạo danh sách: self.videos chứa đường dẫn đến các frames của các video
    # self.videos là list các videos
    # self.videos[i] là list các đường dẫn đến các frames của video thứ i.
    print(f'source = ', source)
    videos, frame_count = get_videos(source)
    print('len(videos) = ', len(videos))

    # 3. For each video
    for i in range(len(videos)):
        frames = videos[i]
        numFrames = len(frames)
        print('numFrames = ', numFrames)
        numKeyFrames = int(numFrames * frame_percent)
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
        print('numKeyFrames =', numKeyFrames)

        # Make keyframe dir for video_name
        keyframe_dir = os.path.join(dest, video_name)  # '../01', '../02'
        os.makedirs(keyframe_dir, exist_ok=True)
        print('keyframe_dir = ', keyframe_dir)

        # Make .csv files
        path2file = os.path.join(dest, '{}_{}_{:03d}.csv'.format(dataset_name, video_name, numKeyFrames))
        print('path2file = ', path2file)
        delete_files_in_dir(keyframe_dir)
        delete_file(path2file)

        # Source of video_name for Evolution
        sourceEvo = os.path.join(source, video_name)
        print('sourceEvo=', sourceEvo)

        # start time
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()

        # run Evolution
        evolution(sourceEvo, numGeneration, popSize, numFrames, numKeyFrames, keyframe_dir, path2file)

        # end time
        ender.record()

        torch.cuda.synchronize()  # # Waits for everything to finish running
        inference_time = starter.elapsed_time(ender) * 1e-3  # milisecond to second
        print(f'Inference time of a video: {inference_time} seconds')


def main_extract_keyframes():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='ped2', help='dataset name')
    parser.add_argument('--numGeneration', type=int, default=10, help='Max number of generation')
    parser.add_argument('--popSize', type=int, default=10, help='number of candidates')
    parser.add_argument('--frame_percent', type=float, default=0.3, help=' The percent of frames to extract')

    args = parser.parse_args()
    source_dir = os.path.join(os.getcwd(), 'dataset', args.dataset, 'training')
    dest_dir = os.path.join(os.getcwd(), 'dataset', args.dataset)
    extract_keyframes(source_dir, dest_dir, args.numGeneration, args.popSize, args.frame_percent)


# in terminal: python KF_DE_Entropy_ped2.py --dataset ped2
if __name__ == '__main__':
    main_extract_keyframes()
