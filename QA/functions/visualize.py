import random
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from .utils import *

def plot_image_PIL(img_PIL):
    # Turn the image into an NUMPY array
    img_as_array = np.asarray(img_PIL)

    # Plot the image with matplotlib
    plt.imshow(img_as_array)
    plt.title(f"Image shape: {img_as_array.shape} -> [height, width, color_channels]")

def plot_image_cv2(img_cv2):
    # Convert BGR to RGB
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

    # Plot the image with matplotlib
    plt.imshow(img_cv2)
    plt.title(f"Image shape: {img_cv2.shape} -> [height, width, color_channels]")

def plot_clip_cv2(clip):
    plt.figure(figsize=(16, 8))
    j = 0
    channels = 3
    n = int(clip.shape[0] / channels)
    for i in range(0, clip.shape[0], channels):
        img = clip[i:i+3] # shape = (3,128,128)
        img_permuted = np.transpose(img, (1, 2, 0)) # shape = (128,128,3)
        j = j + 1
        plt.subplot(1, n, j)
        plt.imshow(img_permuted)
        plt.axis("off")
    
def plot_transformed_images(image_paths, transform, n=2, seed=42):
    """Plots a series of random images from image_paths.

    Will open n image paths from image_paths, transform them
    with transform and plot them side by side.

    Args:
        image_paths (list): List of target image paths. 
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator. Defaults to 42.
    """
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f) 
            ax[0].set_title(f"Original \nSize:{f.size}, {np.asarray(f).shape}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib 
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            # permute CHW to HWC
            transformed_image = transform(f).permute(1, 2, 0) 
            ax[1].imshow(transformed_image) 
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")
