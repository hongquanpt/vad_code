import cv2
import os
import shutil
import sys
import argparse
import numpy as np
import torch

sys.path.append('../../')

from dataset.image_reader import CV2_imread
from dataset.util import get_videos, delete_file, delete_files_in_dir, is_exists_file
from dataset.csv_lib import writefile_csv_dic
from libs.flownet_networks.flownet2_models import FlowNet2

FLOWNET_INPUT_WIDTH = {"ped1": 512 * 2, "ped2": 512 * 2, "avenue": 512 * 2, "shanghaitech": 1024}
FLOWNET_INPUT_HEIGHT = {"ped1": 384 * 2, "ped2": 384 * 2, "avenue": 384 * 2, "shanghaitech": 640}


def extract_optical_flow(source, flownet2_path):
    """
    1. Extract Optical Flow using flownet2
    2. Apply method in paper: Key frame and skeleton extraction for deep learning-based human action recognition
    param source: full path to dataset (i.e: 'home/dataset/ped2_tiny/training')
    param flownet2_path: '/home/asus/DATA/VAD/libs/flownet_networks/weights/FlowNet2_checkpoint.pth.tar'
    """

    parent_path = os.path.dirname(source)  # 'home/dataset/ped2_tiny'
    current_dirname = os.path.basename(source)  # 'training'
    dataset_name = os.path.basename(parent_path)  # 'ped2_tiny'

    print('parent_path = ', parent_path)
    print('current_dirname = ', current_dirname)
    print('dataset_name = ', dataset_name)

    # 2. Tạo danh sách: self.videos chứa đường dẫn đến các frames của các video
    # self.videos là list các videos
    # self.videos[i] là list các đường dẫn đến các frames của video thứ i.
    videos, frame_count = get_videos(source)
    print('len(videos) = ', len(videos))

    # 3. Load pre-trained flownet2 model
    # code in: HF2VAD/extract_flows.py
    WIDTH, HEIGHT = FLOWNET_INPUT_WIDTH[dataset_name], FLOWNET_INPUT_HEIGHT[dataset_name]
    flownet2 = FlowNet2()  # initialize structure of flownet2
    is_exists_file(flownet2_path)
    checkpoint = torch.load(flownet2_path)
    flownet2.load_state_dict(checkpoint['state_dict'])
    flownet2.cuda()
    flownet2.eval()
    with torch.no_grad():
        # 4. Duyệt các video và các frames trong 1 video
        M = []
        num_videos = len(videos)

        # for each video
        for i in range(0, num_videos):
            frames = videos[i]

            # for each frame (full_path) in a video
            num_frames = len(frames)
            M_i = []  # a list of Matrices of a video
            for j in range(0, num_frames - 1):
                frame1 = frames[j]
                frame2 = frames[j + 1]
                curr_frame_name = frame1.split('/')[-1]  # '000.jpg'

                # (H,W,C)
                im1 = CV2_imread(frame1)
                im2 = CV2_imread(frame2)

                H = im1.shape[0]
                W = im1.shape[1]
                # H = 240, W = 360 
                # print(f'H = {H}, W = {W} \n')

                # resize format (W',H')　　
                old_size = (W, H)  # W,H
                im1 = cv2.resize(im1, (WIDTH, HEIGHT))
                im2 = cv2.resize(im2, (WIDTH, HEIGHT))
                ims = np.array([im1, im2]).astype(np.float32)  # numpy[2,H,W,C]

                # convert ims to tensor, then add more dimension
                ims = torch.from_numpy(ims).unsqueeze(0)

                # tensor[bs,2,H,W,C] -> tensor[bs,C,2,H,W]
                ims = ims.permute(0, 4, 1, 2, 3).contiguous().cuda()

                # predict optical flow
                pred_flow = flownet2(ims).cpu().data
                # pred_flow.shape = torch.Size([1, 2, 768, 1024])
                # print('pred_flow.shape = ', pred_flow.shape)

                # convert to numpy[2,H,W], and to numpy[H,W,2]
                pred_flow = pred_flow[0].numpy().transpose((1, 2, 0))
                # resize pred_flow to old_size
                new_inputs = cv2.resize(pred_flow, old_size)

                # calcuate matrix M_t at time t (frame t)
                # old_size(W, H)
                M_t = 0
                for h in range(0, H):
                    for w in range(0, W):
                        M_t += np.abs(new_inputs[h, w, 0]) + np.abs(new_inputs[h, w, 1])

                M_i.append(M_t)
                # np.save(os.path.join(video_dir, curr_frame_name + '.npy'), new_inputs)
            print(f'M[i={i}] (num_elements={j})= \n')
            G_i = []
            for j in range(0, num_frames - 2):
                G_i.append(M_i[j + 1] - M_i[j])
            print(f'G[i={i}] (num_elements={j})= {G_i}\n')
            M.append(G_i)
        return videos, M


def extract_keyframes(videos, M, dest, frame_percent=0.3):
    """
    param videos: list các videos; videos[i] là list các đường dẫn đến các frames của video thứ i.
    param M: list các matrices; M[i] là list các matrices M_t của các frames thứ t trong video thứ i
    param dest: '/home/dataset/ped2_tiny'
    param num_keyframes: the number of keyframes have minima or maxima of motion
    """
    dataset_name = os.path.basename(dest)  # 'ped2_tiny'

    # 1. Tạo thư mục mới: 'home/dataset/ped2_tiny/keyframes_op_frame_percent'
    dest = os.path.join(dest, 'training_keyframes_op_{:.2f}'.format(frame_percent))
    os.makedirs(dest, exist_ok=True)

    # 2. Extract keyframes
    num_videos = len(videos)
    for i in range(0, num_videos):
        # make keyframe dir
        # '/home/dataset/ped2_tiny/keyframes_op_0.3/01', 
        # '/home/dataset/ped2_tiny/keyframes_op_0.3/02'
        video_name = videos[i][0].split('/')[-2]  # '01', '02'
        keyframe_dir = os.path.join(dest, video_name)
        os.makedirs(keyframe_dir, exist_ok=True)

        num_keyframes = int(len(M[i]) * frame_percent) // 2
        # Find the indices of the [num_keyframes] minimum elements using np.argpartition
        min_indices = np.argpartition(M[i], num_keyframes)[:num_keyframes]
        print("The indices of the minimum elements are:", min_indices)

        # Find the indices of the [num_keyframes] maximum elements using np.argpartition
        max_indices = np.argpartition(M[i], -num_keyframes)[-num_keyframes:]
        print("The indices of the maximum elements are:", max_indices)

        # Concatenate two numpy arrays
        keyframes_indices = np.concatenate((min_indices, max_indices))
        # Sort the array in ascending order
        keyframes_indices = np.sort(keyframes_indices)
        print("The indices of the keyframes elements are:", keyframes_indices)

        # Export keyframes to the directory
        keyframes = []
        num_keyframes = len(keyframes_indices)
        for j in range(0, num_keyframes):
            k = keyframes_indices[j]
            keyframes.append(videos[i][k])

        # make .csv files
        path2file = os.path.join(dest, '{}_{}_{:03d}.csv'.format(dataset_name, video_name, num_keyframes))
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


def parse_args():
    parser = argparse.ArgumentParser()
    source_dir = os.path.join(os.getcwd(), 'dataset', 'ped1', 'training')
    dest_dir = os.path.join(os.getcwd(), 'dataset', 'ped1')
    flownet2_path = os.path.join(os.getcwd(), 'libs', 'flownet_networks', 'weights', 'FlowNet2_checkpoint.pth.tar')
    parser.add_argument('--source', type=str,
                        default=source_dir,
                        help='source file')
    parser.add_argument('--dest', type=str,
                        default=dest_dir,
                        help='destination folder')
    parser.add_argument('--flownet2_path', type=str,
                        default=flownet2_path,
                        help='pre-trained flownet2 path')
    parser.add_argument('--frame_percent', type=float, default=0.3, help='frame_percent')

    args = parser.parse_args()
    return args


def main_extract_keyframes_op():
    args = parse_args()

    # 1. Extract optical flow
    videos, M = extract_optical_flow(args.source, args.flownet2_path)

    # 2. Extract keyframes
    extract_keyframes(videos, M, args.dest, args.frame_percent)


# call this in terminal: python dataset/KF_OF.py
if __name__ == '__main__':
    main_extract_keyframes_op()
