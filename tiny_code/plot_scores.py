import argparse
import csv
import os
from dataset.label import LabelVideoDataset
from model.util.metrics import get_anomaly_rectanges

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def read_data_from_csv(file_path):
    # Read anomaly scores from CSV
    frame_ids = []
    scores = []
    with open(f"{file_path}", "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            frame_ids.append(int(row[0]))
            scores.append(float(row[1]))

    return frame_ids, scores


def main():
    dataset_name = "ped2"
    models = ["MNAD", "PAMAE"]
    methods = ["MNAD", "PAMAE_KF_EN"]
    sessions_parent = ["session_2024_01_11_23_22", "session_2024_03_24_09_15"]
    sessions = ["session_2024_03_25_22_54", "session_2024_03_25_23_04"]
    alphas = ['0.9', '0.5']
    videos = ['01', '01']
    scores_files = ["scores_video_01.csv", "scores_video_01.csv"]
    scores_colors = ['red', 'blue']

    idx = 1
    test_set = "testing"
    export_dir = os.path.join(os.getcwd(), 'output', 'scores', dataset_name)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    video_labels = LabelVideoDataset(dataset_name, test_set)()
    num_videos = len(video_labels)
    if num_videos == 0:
        print('WARNING: len(video_labels) == 0. Check PATH to label files')
        return

    # Set the figure size
    plt.figure(figsize=(12, 4))  # Width: 8 inches, Height: 6 inches

    num_files = len(scores_files)
    for i in range(num_files):
        file_path = os.path.join(os.getcwd(), 'output', 'test', models[i], methods[i],
                                 dataset_name, sessions_parent[i], sessions[i], 'visualization',
                                 alphas[i], videos[i],
                                 scores_files[i])
        frame_ids, scores = read_data_from_csv(file_path)
        color = scores_colors[i]
        label = methods[i]

        # plotting the data
        plt.plot(frame_ids, scores, color=color, label=label)

    # Adding the title
    plt.title("Video {:02d}".format(idx))

    # Adding the labels
    plt.xlabel("Frame-ID")
    plt.ylabel("Anomaly Scores")

    frame_labels = video_labels[idx - 1]
    starts, ends = get_anomaly_rectanges(frame_labels)
    for rs, re in zip(starts, ends):
        current_axis = plt.gca()
        current_axis.add_patch(Rectangle((rs, -0.01), re - rs, 1.02, facecolor="pink"))

    plt.legend()
    scores_graph = os.path.join(export_dir, 'video_{:02d}.png'.format(idx))
    plt.savefig(scores_graph, dpi=100)
    plt.close()
    return 1


# call in terminal: python plot_scores.py
if __name__ == '__main__':
    main()
