from PIL import Image
import cv2

import numpy as np
import random
import matplotlib.pyplot as plt

import torchvision.transforms as transforms


def Tensor_convert_to_0_255_np(img_tensor):
    """
    Convert a tensor image on GPU from [-1, 1] to [0, 255]
    i.e, img_tensor.shape = torch.Size([1, 3, 256, 256])
    
    param img_tensor: tensor image in range of [-1, 1]; len(img_tensor.shape) == 4
    :return: numpy.ndarray in range of [0, 255]; shape = (256,256,3)
    """
    img_tensor = img_tensor.squeeze(0)  # torch.Size([1, 3, 256, 256]) => torch.Size([3, 256, 256])
    img_np = img_tensor.cpu().data.numpy()  # numpy.ndarray[3, 256, 256]
    img_np = ((img_np + 1) * 127.5).astype(np.uint8)  # convert images from [-1; 1] to [0; 255]
    img_np = np.transpose(img_np, (1, 2, 0))  # shape = (256,256,3)
    return img_np


# =========================================================
# =========================================================
"""
Cach 1: step by step
1) img_tensor [-1,1] => img_np [0,255] in CHW 
2) => np.transpose(img_np) in HWC 
3) => plt.imshow(img_np)
"""


def Plot_image_tensor_op1(img_tensor):
    """
    Plots a tensor image by converting it to a numpy array
    
    param img_tensor: a tensor image on gpu in range of [-1,1]
    """
    # Turn the image into an NUMPY array
    img_np = Tensor_convert_to_0_255_np(img_tensor)  # shape = (3,256,256)

    # Plot the image with matplotlib
    plt.imshow(img_np)
    plt.title(f"Image shape: {img_np.shape} -> [height, width, color_channels]")


"""
Cach 2: using transform
"""


def transform_tensor_to_PIL():
    """
    Returns a transform that convert a tensor (len(shape)=4) to a PIL image
    """
    transform = [
        transforms.Lambda(lambda t: t.squeeze(0)),
        transforms.Lambda(lambda t: (t + 1) / 2),  # tensor [-1, 1] to [0, 1]
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),  # tensor [0, 1] to tensor [0.0,255.0]
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),  # tensor[0.0,255.0] to numpy [0,255]
        transforms.ToPILImage(),  # numpy to PIL image object
    ]
    return transforms.Compose(transform)


def transform_tensor_to_np():
    """
    Returns a transform that convert a tensor (len(shape)=4) to a PIL image
    """
    transform = [
        transforms.Lambda(lambda t: t.squeeze(0)),
        transforms.Lambda(lambda t: (t + 1) / 2),  # tensor [-1, 1] to [0, 1]
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),  # tensor [0, 1] to tensor [0.0,255.0]
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),  # tensor[0.0,255.0] to numpy [0,255]
    ]
    return transforms.Compose(transform)


def Plot_image_tensor_op2(img_tensor):
    """
    Plots a tensor image by applying a transform which convert it to a numpy array
    
    param img_tensor: a tensor image on gpu in range of [-1,1]
    """
    tf = transform_tensor_to_np()
    img_np = tf(img_tensor)

    # Plot the image with matplotlib
    plt.imshow(img_np)
    plt.title(f"Image shape: {img_np.shape} -> [height, width, color_channels]")


def Plot_image_tensor_op3(img_tensor):
    """
    Plots a tensor image by applying a transform which convert it to a PIL image object
    
    param img_tensor: a tensor image on gpu in range of [-1,1]
    """
    tf = transform_tensor_to_PIL()
    img_PIL = tf(img_tensor)

    # Plot the image with matplotlib
    plt.imshow(img_PIL)
    plt.title(f"Image size: {img_PIL.size} -> [width, height]")


# =========================================================
# =========================================================
def Plot_image_PIL(img_PIL):
    """
    Plots a PIL image
    param img_PIL: a PIL image object.
    """
    # Turn the image into an NUMPY array
    img_np = np.asarray(img_PIL)

    # Plot the image with matplotlib
    plt.imshow(img_np)
    plt.title(f"Image shape: {img_np.shape} -> [height, width, color_channels]")


def Plot_image_CV2(img_CV2):
    """
    Plots a CV2 image
    param img_CV2: a CV2 image (a numpy array).
    """
    # Convert BGR to RGB
    img_CV2 = cv2.cvtColor(img_CV2, cv2.COLOR_BGR2RGB)

    # Plot the image with matplotlib
    plt.imshow(img_CV2)
    plt.title(f"Image shape: {img_CV2.shape} -> [Height, Width, Color_Channels]")


# =========================================================
# =========================================================
def Plot_clip_np(clip_np):
    """
    Plots all images in a numpy clip in a row
    
    param clip_np: is 'numpy.ndarray' which contains consecutive frames in range of [0,255]
    i.e, clip_np.shape = (128,128,4*3); len(clip_np.shape) = 3
    """
    # clip_np.shape[0] = num_of_frames * 3 (3 channels)
    plt.figure(figsize=(16, 8))
    j = 0
    channels = 3
    num_of_frames = int(clip_np.shape[2] / channels)
    for i in range(0, clip_np.shape[2], channels):  # i_th image
        img = clip_np[:, :, i:i + 3]  # shape = (128,128,3)
        j = j + 1
        plt.subplot(1, num_of_frames, j)
        plt.imshow(img)
        plt.axis("off")


def Plot_clip_tensor(clip_tensor):
    """
    Plots all images in a tensor clip in a row
    
    param clip_tensor: is 'numpy.ndarray' which contains consecutive frames in range of [-1,1]
    i.e, clip_tensor.shape = (1,3*4,128,128); len(clip_tensor.shape) == 4
    """
    clip_np = Tensor_convert_to_0_255_np(clip_tensor)  # i.e, clip_np.shape = (256,256,12)
    Plot_clip_np(clip_np)


def Plot_multi_images_in_rows_op1(imgs, max_samples=20, num_cols=4):
    """ 
    Plots some samples from a list of images in format of (row,column)
    
    param imgs: a tensor of images, i.e, torch.Size([1, 90, 256, 256]) (90= 30images*3channels)
    """
    imgs = imgs.squeeze(0)  # torch.Size([90, 256, 256])
    plt.figure(figsize=(25, 25))
    j = 0
    channels = 3
    total_frames = int(imgs.shape[0] / channels)  # 30 images = 90/3
    for i in range(0, imgs.shape[0], channels):  # i_th image
        j = j + 1
        if j > max_samples:
            break
        img = imgs[i:i + 3]  # torch.Size([3,256,256])
        plt.subplot(int(total_frames / num_cols) + 1, num_cols, j)
        tf = transform_tensor_to_np()
        img_np = tf(img)
        plt.imshow(img_np)


def Plot_multi_images_in_rows_op2(imgs, max_samples=20, num_cols=4):
    """ 
    Plots some samples from a list of images in format of (row,column)
    
    param imgs: a tensor of images, i.e, torch.Size([1, 90, 256, 256]) (90= 30images*3channels)
    """
    imgs_np = Tensor_convert_to_0_255_np(imgs)  # clip_np.shape = (256,256,90)
    plt.figure(figsize=(25, 25))
    j = 0
    channels = 3
    total_frames = int(imgs_np.shape[2] / channels)  # 30 images
    for i in range(0, imgs_np.shape[2], channels):  # i_th image
        j = j + 1
        if j > max_samples:
            break
        img_np = imgs_np[:, :, i:i + 3]  # shape = (256,256,3)
        plt.subplot(int(total_frames / num_cols) + 1, num_cols, j)
        plt.imshow(img_np)


# =========================================================
def Plot_image_paths(image_paths, transform, n=2, seed=42):
    """Plots a series of random images from image_paths.

    Will open n image paths from image_paths, transform them
    with transform and plot them side by side.

    Args:
        image_paths (list): List of target image paths. 
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 2.
        seed (int, optional): Random seed for the random generator. Defaults to 42.
    """
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:  # PIL Image
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
