import argparse
import csv
import os

from dataset.label import LabelVideoDataset
from model.util.metrics import get_anomaly_rectanges


def parse_args():
    parser = argparse.ArgumentParser(description='VAD')

    parser.add_argument("--dataset", help='ped2, avenue, or shanghaitech',
                        default='ped2', type=str)

    # Read args from command line
    args = parser.parse_args()
    return args.dataset


def export_anomaly_ranges_to_csv(dataset_name="ped2", test_set="testing"):
    video_labels = LabelVideoDataset(dataset_name, test_set)()
    num_videos = len(video_labels)
    if num_videos == 0:
        print('WARNING: len(video_labels) == 0. Check PATH to label files')
        return
    anomaly_ranges_dir = os.path.join(os.getcwd(), 'dataset', dataset_name, 'anomaly_ranges')  # '/home/dataset'
    if not os.path.exists(anomaly_ranges_dir):
        os.makedirs(anomaly_ranges_dir)
    for idx in range(num_videos):
        file_path = os.path.join(anomaly_ranges_dir, 'video_{:02d}.csv'.format(idx + 1))

        frame_labels = video_labels[idx]
        starts, ends = get_anomaly_rectanges(frame_labels)

        # Export anomaly ranges to CSV
        with open(f"{file_path}", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["starts", "ends"])
            for start, end in zip(starts, ends):
                writer.writerow([start, end])


def main():
    dataset_name = parse_args()
    export_anomaly_ranges_to_csv(dataset_name)
    return 1


# call in terminal: python frame_labels.py --dataset ped2
if __name__ == '__main__':
    main()
