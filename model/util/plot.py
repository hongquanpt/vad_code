# draw anomaly_score line of level-frame evaluation
# t-SNE
import os
import cv2
import csv
import numpy as np

from PIL import Image, ImageDraw, ImageFont
import imageio
import shutil

from .metrics import mse_error_frame, get_anomaly_rectanges
from .metrics import calculate_anomaly_scores_MNAD, calculate_anomaly_scores
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
import torch.nn as nn

# Plot the confusion matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix


def get_img_np(img_tensor_gpu):
    """
    param img_tensor_gpu: tensor image (shape of [1, 3, 256, 256]) in range of [-1, 1]
    :return: numpy.ndarray in range of [0, 255]
    """
    img_np = img_tensor_gpu.cpu().data.numpy()  # [1, 3, 256, 256]
    img_np = img_np.squeeze(0)  # shape = (3, 256,256)
    img_np = np.transpose(img_np, (1, 2, 0))  # shape = (256,256,3)
    img_np = ((img_np + 1) * 127.5).astype(np.uint8)  # convert images from [-1; 1] to [0; 255]
    return img_np


def export_gt_img_cv2(target, visualization_dir_path, video_id, frame_id):
    """
    param target: with shape of [1, 3, 256, 256] in range of [-1, 1]
                   remember: batch_size = 1, if not error will appear
    """

    gt_img = get_img_np(target)

    export_dir = os.path.join(visualization_dir_path, '{:02d}').format(video_id)
    filepath = os.path.join(export_dir, 'gt_{:04d}.png').format(frame_id)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)
    try:
        is_successed = cv2.imwrite(filepath, gt_img)
        print('export_gt_img_cv2:  ', is_successed)
    except:
        print('Error: export_gt_img_cv2 !!!')


def export_output_img_cv2(output, visualization_dir_path, video_id, frame_id):
    """
    param output: with shape of [1, 3, 256, 256] in range of [-1, 1]
                   remember: batch_size = 1, if not error will appear
    """
    output_img = get_img_np(output)

    export_dir = os.path.join(visualization_dir_path, '{:02d}').format(video_id)
    filepath = os.path.join(export_dir, 'out_{:04d}.png').format(frame_id)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)
    try:
        is_successed = cv2.imwrite(filepath, output_img)
        print('export_output_img_cv2:  ', is_successed)
    except:
        print('Error: export_output_img_cv2 !!!')


def get_color_error_fr(target, output):
    """
    param target: with shape of [1, 3, 256, 256] in range of [-1, 1]
    param output: with shape of [1, 3, 256, 256] in range of [-1, 1]
                   remember: batch_size = 1, if not error will appear
    """
    """
    img_np = img_tensor_gpu.cpu().data.numpy()  # [1, 3, 256, 256]
    img_np = img_np.squeeze(0)  # shape = (3, 256,256)
    img_np = np.transpose(img_np, (1, 2, 0))  # shape = (256,256,3)
    img_np = ((img_np + 1) * 127.5).astype(np.uint8)  # convert images from [-1; 1] to [0; 255]
    """
    # mse_error_frame(output, target)
    loss_func_mse = nn.MSELoss(reduction='none')
    error = loss_func_mse((output[0] + 1) / 2, (target[0] + 1) / 2)  # convert [-1,1] to [0,1],
    # print(f"error.shape = {error.shape}")  # = torch.Size([3, 256, 256])
    error_fr = error[0].cpu().data.detach().numpy()  # convert tensor gpu to cpu, then to numpy
    # print(f"error_fr.shape = {error_fr.shape}")  # = (256, 256)

    error_fr = error_fr[:, :, np.newaxis]
    error_fr = (error_fr - np.min(error_fr)) / (np.max(error_fr) - np.min(error_fr))  # normalized
    error_fr = error_fr * 255  # convert [0,1] to [0,255]
    error_fr = error_fr.astype(dtype=np.uint8)
    color_error_fr = cv2.applyColorMap(error_fr, cv2.COLORMAP_JET)

    # print(f"color_error_fr.shape = {color_error_fr.shape}")  # = (256, 256, 3)
    color_error_fr = color_error_fr.astype(np.uint8)[:, :, [2, 1, 0]]  # change RGB to BGR in Channel
    return color_error_fr


def export_mse_img_cv2(target, output, visualization_dir_path, video_id, frame_id):
    color_error_fr = get_color_error_fr(target, output)

    export_dir = os.path.join(visualization_dir_path, '{:02d}').format(video_id)
    filepath = os.path.join(export_dir, 'mse_{:04d}.png').format(frame_id)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)
    try:
        is_successed = cv2.imwrite(filepath, color_error_fr)
        print('export_mse_img_cv2:  ', is_successed)
    except:
        print('Error: export_mse_img_cv2 !!!')


def export_anomaly_scores_to_csv(export_dir, idx, frame_ids, scores):
    file_path = os.path.join(export_dir, 'scores_video_{:02d}.csv'.format(idx))
    print(f"export_anomaly_scores_to_csv: {file_path}")

    # Export anomaly ranges to CSV
    with open(f"{file_path}", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame_ids", "scores"])
        for frame_id, score in zip(frame_ids, scores):
            writer.writerow([frame_id, score])


def export_thresholds_to_csv(export_dir, idx, frame_ids, thresholds):
    file_path = os.path.join(export_dir, 'scores_video_{:02d}.csv'.format(idx))
    print(f"export_anomaly_scores_to_csv: {file_path}")

    # Export anomaly ranges to CSV
    with open(f"{file_path}", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame_ids", "thresholds"])
        for frame_id, score in zip(frame_ids, thresholds):
            writer.writerow([frame_id, score])


def plot_anomaly_scores_MNAD(idx, frame_ids, frame_labels,
                             frame_psnr, frame_feas_distance,
                             alpha, export_dir):
    """
    Plot the graph of anomaly scores of a video
    """
    # get rectangles of a video idx^th
    starts, ends = get_anomaly_rectanges(frame_labels)

    # calculate anomaly scores of video idx^th
    scores = calculate_anomaly_scores_MNAD(frame_psnr, frame_feas_distance, alpha)

    # plotting the data
    plt.plot(frame_ids, scores)

    # Adding the title
    plt.title("Video {:02d}".format(idx))

    # Adding the labels
    plt.xlabel("Frame-ID")
    plt.ylabel("Anomaly Scores")

    for rs, re in zip(starts, ends):
        current_axis = plt.gca()
        current_axis.add_patch(Rectangle((rs, -0.01), re - rs, 1.02, facecolor="pink"))
    scores_graph = os.path.join(export_dir, 'video_{:02d}.png'.format(idx))
    plt.savefig(scores_graph, dpi=100)
    plt.close()
    return scores, starts, ends, scores_graph


def plot_anomaly_scores(idx, frame_ids, frame_labels, frame_psnr, export_dir):
    """
    Plot the graph of anomaly scores of a video
    """
    # get rectangles of a video idx^th
    starts, ends = get_anomaly_rectanges(frame_labels)

    # calculate anomaly scores of video idx^th
    scores = calculate_anomaly_scores(frame_psnr)

    # plotting the data
    plt.plot(frame_ids, scores)

    # Adding the title
    plt.title("Video {:02d}".format(idx))

    # Adding the labels
    plt.xlabel("Frame-ID")
    plt.ylabel("Anomaly Scores")

    for rs, re in zip(starts, ends):
        current_axis = plt.gca()
        current_axis.add_patch(Rectangle((rs, -0.01), re - rs, 1.02, facecolor="pink"))
    scores_graph = os.path.join(export_dir, 'video_{:02d}.png'.format(idx))
    plt.savefig(scores_graph, dpi=100)
    plt.close()
    return scores, starts, ends, scores_graph


def export_images(images, visualization_dir_path):
    print(images.shape)
    plt.figure(figsize=(32, 32))

    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1), ], dim=-2).permute(1, 2, 0).cpu())

    scores_graph = os.path.join(visualization_dir_path, 'images.png')
    plt.savefig(scores_graph, dpi=100)
    plt.close()
    return 1


def add_text_to_image(image, text):
    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Define the font (you can specify the font file and size)
    font = ImageFont.load_default()  # Default font

    # Define the position and color for the text
    position = (10, 10)  # Adjust the position as needed
    text_color = (255, 255, 255)  # White color

    # Add the text to the image
    draw.text(position, text, fill=text_color, font=font)

    return image


# Create a custom colormap similar to "Jet"
def custom_jet_colormap(num_colors=256):
    custom_colormap = np.zeros((num_colors, 3), dtype=np.uint8)
    for i in range(num_colors):
        r, g, b = plt.cm.jet(i)[:3]
        custom_colormap[i] = (int(r * 255), int(g * 255), int(b * 255))
    return custom_colormap


def plot_gif_(idx, frame_targets, frame_outputs,
              frame_ids, scores, starts, ends,
              export_dir, gif_width=400, gif_height=300, dpi=100
              ):
    """
    Export results in a gif image:
    1. 03 images: GT, Output, ErrorMap
    2. A graph of Anomaly Scores
    param idx: index of video
    param frame_targets: a list of [1, 3, 256, 256] (bz=1, C, H, W)
    param frame_outputs: a list of [1, 3, 256, 256] (bz=1, C, H, W)
    param frame_ids: a list of indices of frames of video idx^th
    param scores: a list of anomaly scores of corresponding frames
    param starts: get rectangles of a video idx^th
    param ends: get rectangles of a video idx^th
    param export_dir: directory of export
    param gif_width: width of gif image
    param gif_height: height of gif image
    param dpi: quality of gif image
    """
    output_gif_path = os.path.join(export_dir, 'video_{:02d}.gif'.format(idx))
    # Create a list to store frames for the GIF
    frames = []
    border_width = 1

    # Loop through the variables and create frames
    num_frames = len(frame_ids)
    scores_temp = np.zeros_like(scores)
    for i in range(num_frames):
        # Assuming frame_targets and frame_outputs contain tensors [1, 3, 256, 256] in the range [-1, 1]
        # Convert tensors to numpy arrays and normalize them to the range [0, 255]
        target_image = get_img_np(frame_targets[i])
        output_image = get_img_np(frame_outputs[i])
        mse_image = get_color_error_fr(frame_targets[i], frame_outputs[i])

        # Create a PIL image from the numpy arrays
        target_image = Image.fromarray(target_image)
        output_image = Image.fromarray(output_image)
        mse_image = Image.fromarray(mse_image)

        # Resize the images to the desired width and height
        target_image = target_image.resize((gif_width // 3 - border_width, gif_height // 2 - border_width))
        output_image = output_image.resize((gif_width // 3 - border_width, gif_height // 2 - border_width))
        mse_image = mse_image.resize((gif_width // 3 - border_width, gif_height // 2 - border_width))

        # Add text to the images (e.g., frame ID)
        text = f"Frame:{frame_ids[i]}"
        target_image = add_text_to_image(target_image, text)
        output_image = add_text_to_image(output_image, text)
        mse_image = add_text_to_image(mse_image, text)

        # Create a line graph of anomaly_scores for the current frame
        fig, ax = plt.subplots(figsize=(6, 2))
        scores_temp[i] = scores[i]
        ax.plot(frame_ids, scores_temp, label='prediction', color='blue')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Anomaly Score')
        ax.set_title('Anomaly Scores over Frames')
        ax.grid(True)

        # Save the line graph as an image
        for rs, re in zip(starts, ends):
            current_axis = plt.gca()
            current_axis.add_patch(Rectangle((rs, -0.01), re - rs, 1.02, facecolor="pink"))
        graph_image_path = os.path.join(export_dir, f'graph_image_{i}.png')
        plt.savefig(graph_image_path, bbox_inches='tight', dpi=dpi)
        plt.close()

        # Load the graph image for the current frame
        graph_frame = Image.open(graph_image_path)
        graph_frame = graph_frame.resize((gif_width, gif_height // 2))
        # Create a combined image with the first row (images) and second row (graph)
        combined_image = Image.new('RGB', (gif_width, gif_height))

        # Paste the three images (first row) into the combined image
        combined_image.paste(target_image, (0, 0))
        combined_image.paste(output_image, (gif_width // 3 + border_width, 0))
        combined_image.paste(mse_image, (2 * gif_width // 3 + border_width, 0))

        # Paste the graph image (second row)
        combined_image.paste(graph_frame, (0, gif_height // 2))

        # Append the combined image to the frames list
        frames.append(combined_image)

        # Clean up the temporary graph image
        os.remove(graph_image_path)

    # Save frames as a GIF with the specified width and height
    # Adjust the duration as needed (in seconds)
    imageio.mimsave(output_gif_path, frames, duration=0.5, quantizer="nq")
    print(f'GIF saved as {output_gif_path}')


def plot_gif(idx, frame_targets, frame_outputs,
             frame_ids, scores, starts, ends,
             export_dir, gif_width=400, gif_height=300, dpi=100
             ):
    """
    Export results in a gif image:
    1. 03 images: GT, Output, ErrorMap
    2. A graph of Anomaly Scores
    param idx: index of video
    param frame_targets: a list of [1, 3, 256, 256] (bz=1, C, H, W)
    param frame_outputs: a list of [1, 3, 256, 256] (bz=1, C, H, W)
    param frame_ids: a list of indices of frames of video idx^th
    param scores: a list of anomaly scores of corresponding frames
    param starts: get rectangles of a video idx^th
    param ends: get rectangles of a video idx^th
    param export_dir: directory of export
    param gif_width: width of gif image
    param gif_height: height of gif image
    param dpi: quality of gif image
    """
    output_gif_path = os.path.join(export_dir, 'video_{:02d}.gif'.format(idx))
    # Create a list to store frames for the GIF
    frames = []

    # Loop through the variables and create frames
    num_frames = len(frame_ids)
    anomaly_score_dir = os.path.join(export_dir, 'anomaly_score')
    if not os.path.exists(anomaly_score_dir):
        os.makedirs(anomaly_score_dir)
    for i in range(num_frames):
        # Assuming frame_targets and frame_outputs contain tensors [1, 3, 256, 256] in the range [-1, 1]
        # Convert tensors to numpy arrays and normalize them to the range [0, 255]
        target_image = get_img_np(frame_targets[i])
        output_image = get_img_np(frame_outputs[i])
        mse_image = get_color_error_fr(frame_targets[i], frame_outputs[i])

        plt.axis('off')
        plt.subplot(231)
        plt.title('Ground truth', fontsize='small')
        plt.imshow(target_image)

        plt.axis('off')
        plt.subplot(232)
        plt.title('Predicted frame', fontsize='small')
        plt.axis('off')
        plt.imshow(output_image)

        plt.subplot(233)
        plt.title('Prediction error', fontsize='small')
        plt.axis('off')
        plt.imshow(mse_image)

        # anomaly score plot
        plt.subplot(212)
        plt.plot(range(i + 1), scores[0:i + 1], label='prediction', color='blue')
        plt.xlim(0, num_frames - 1)
        plt.xticks(fontsize='x-small')
        plt.xlabel('Frame ID', fontsize='x-small')
        plt.ylim(-0.01, 1.01)
        plt.ylabel('Anomaly Score', fontsize='x-small')
        plt.yticks(fontsize='x-small')
        plt.title('Anomaly Scores of video {}'.format(idx))

        # Save the line graph as an image
        for rs, re in zip(starts, ends):
            current_axis = plt.gca()
            current_axis.add_patch(Rectangle((rs, -0.01), re - rs, 1.02, facecolor="pink"))

        graph_image_path = os.path.join(anomaly_score_dir, 'frame_{:02d}_{:04d}.png').format(idx, i)
        plt.savefig(graph_image_path, dpi=dpi)
        plt.close()

        graph_frame = Image.open(graph_image_path)
        graph_frame = graph_frame.resize((gif_width, gif_height))
        combined_image = Image.new('RGB', (gif_width, gif_height))
        combined_image.paste(graph_frame, (0, 0))
        frames.append(combined_image)

        # Clean up the temporary graph image
        os.remove(graph_image_path)

    # Delete the folder and all its files
    shutil.rmtree(anomaly_score_dir)
    print(f"Folder: {anomaly_score_dir} and all its files deleted successfully!")

    # Save frames as a GIF with the specified width and height
    # Adjust the duration as needed (in seconds)
    imageio.mimsave(output_gif_path, frames, duration=0.5, quantizer="nq")
    print(f'GIF saved as {output_gif_path}')


def export_gt_output_mse(frame_id, frame_target, frame_output, export_dir):
    """
    param frame_id: frame_id of a video
    param frame_target: [1, 3, 256, 256] (bz=1, C, H, W)
    param frame_output: [1, 3, 256, 256] (bz=1, C, H, W)
    param export_dir: directory of export
    """
    # Assuming frame_targets and frame_outputs contain tensors [1, 3, 256, 256] in the range [-1, 1]
    # Convert tensors to numpy arrays and normalize them to the range [0, 255]
    target_image = get_img_np(frame_target)
    output_image = get_img_np(frame_output)
    mse_image = get_color_error_fr(frame_target, frame_output)

    # export_dir = os.path.join(visualization_dir_path, str(best_alpha))
    # export_dir = make_dir_path(export_dir, '{:02d}'.format(idx + 1))

    cv2.imwrite(os.path.join(export_dir, 'gt_{:04d}.png').format(frame_id), target_image)
    cv2.imwrite(os.path.join(export_dir, 'output_{:04d}.png').format(frame_id), output_image)
    cv2.imwrite(os.path.join(export_dir, 'mse_{:04d}.png').format(frame_id), mse_image)


def plot_video(idx, frame_targets, frame_outputs,
               frame_ids, scores, starts, ends,
               export_dir, dpi=300):
    """
    Export results in a gif image:
    1. 03 images: GT, Output, ErrorMap
    2. A graph of Anomaly Scores
    param idx: index of video
    param frame_targets: a list of [1, 3, 256, 256] (bz=1, C, H, W)
    param frame_outputs: a list of [1, 3, 256, 256] (bz=1, C, H, W)
    param frame_ids: a list of indices of frames of video idx^th
    param scores: a list of anomaly scores of corresponding frames
    param starts: get rectangles of a video idx^th
    param ends: get rectangles of a video idx^th
    param export_dir: directory of export
    param dpi: quality of video
    """

    # Loop through the variables and create frames
    num_frames = len(frame_ids)
    anomaly_score_dir = os.path.join(export_dir, 'anomaly_score')
    if not os.path.exists(anomaly_score_dir):
        os.makedirs(anomaly_score_dir)
    for i in range(num_frames):
        # Assuming frame_targets and frame_outputs contain tensors [1, 3, 256, 256] in the range [-1, 1]
        # Convert tensors to numpy arrays and normalize them to the range [0, 255]
        target_image = get_img_np(frame_targets[i])
        output_image = get_img_np(frame_outputs[i])
        mse_image = get_color_error_fr(frame_targets[i], frame_outputs[i])

        plt.axis('off')
        plt.subplot(231)
        plt.title('Ground truth', fontsize='small')
        plt.imshow(target_image)

        plt.axis('off')
        plt.subplot(232)
        plt.title('Predicted frame', fontsize='small')
        plt.axis('off')
        plt.imshow(output_image)

        plt.subplot(233)
        plt.title('Prediction error', fontsize='small')
        plt.axis('off')
        plt.imshow(mse_image)

        # anomaly score plot
        plt.subplot(212)
        plt.plot(range(i + 1), scores[0:i + 1], label='prediction', color='blue')
        plt.xlim(0, num_frames - 1)
        plt.xticks(fontsize='x-small')
        plt.xlabel('Frame ID', fontsize='x-small')
        plt.ylim(-0.01, 1.01)
        plt.ylabel('Anomaly Score', fontsize='x-small')
        plt.yticks(fontsize='x-small')
        plt.title('Anomaly Scores of video {}'.format(idx))

        # Save the line graph as an image
        for rs, re in zip(starts, ends):
            current_axis = plt.gca()
            current_axis.add_patch(Rectangle((rs, -0.01), re - rs, 1.02, facecolor="pink"))

        graph_image_path = os.path.join(anomaly_score_dir, 'frame_{:02d}_{:04d}.png').format(idx, i)
        plt.savefig(graph_image_path, dpi=dpi)
        plt.close()

    output_video_path = os.path.join(export_dir, 'video_{:02d}.mp4'.format(idx))
    images_to_video(image_folder=anomaly_score_dir,
                    output_video=output_video_path,
                    fps=20)


def images_to_video(image_folder, output_video, fps=30):
    """
    # Example usage
        images_to_video(image_folder='/path/to/images', output_video='output.mp4', fps=30)
    """
    # Get the list of image filenames in the folder
    image_files = sorted([os.path.join(image_folder, file) for file in os.listdir(image_folder) if
                          file.endswith(('png', 'jpg', 'jpeg'))])

    # Load the first image to get dimensions
    img = cv2.imread(image_files[0])
    height, width, _ = img.shape

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec based on file extension (e.g., 'XVID' for AVI)
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Iterate through each image and write to the video
    for image_file in image_files:
        img = cv2.imread(image_file)
        out.write(img)

    # Release the VideoWriter object
    out.release()
    print(f"Video = {output_video} created successfully!")

    """
    # Delete the image files in the folder
    for image_file in image_files:
        os.remove(image_file)
    print("Image files deleted successfully!")
    """

    # Delete the folder and all its files
    shutil.rmtree(image_folder)
    print(f"Folder = {image_folder} and all its files deleted successfully!")


def plot_confusion_matrix(y_true, y_score, optimal_threshold, filepath):
    # Apply the optimal threshold to get the predicted labels
    y_pred = (y_score >= optimal_threshold).astype(int)

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Normal", "Abnormal"],
                yticklabels=["Normal", "Abnormal"])
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    plt.savefig(filepath, dpi=100)
    plt.close()
