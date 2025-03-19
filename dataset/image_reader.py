import cv2
import os
from PIL import Image
import numpy as np

"""
OpenCV tends to prefer 8-bit unsigned integers in range [0..255] 
CV2 color image:
- Three color channels: Blue-Green-Red (BGR): img_BGR = cv2.imread('data/src/lena.jpg')
- Height, Width, Color Channels (H, W, C=3) = img_BGR.shape
    H, W, C = img_BGR.shape[0], img_BGR.shape[1], img_BGR.shape[2]

CV2 gray image:
- img_gray = cv2.imread('data/src/lena.jpg', cv2.IMREAD_GRAYSCALE)
- Height, Width (H, W) = img_gray.shape
    H, W = img_gray.shape[0], img_gray.shape[1]
"""


def CV2_imread(file_path, flag=cv2.IMREAD_COLOR):
    """
    Read image using OpenCV's cv2.imread() 
    Returns: 
    - 'numpy.ndarray' where the third dimension represents the Color channels (HWC)
    - H, W
    Notes that: color channels are BGR

    param file_path: the full path of image
    param flag: cv2.IMREAD_COLOR (1), cv2.IMREAD_GRAYSCALE (0), cv2.IMREAD_UNCHANGED (-1)
    :return: numpy.ndarray 8-bit unsigned integers in range of [0,255] (Image shape: HWC)
    """
    # https://www.geeksforgeeks.org/python-opencv-cv2-imread-method/
    # flag = cv2.IMREAD_COLOR or 1 (default)
    # flag = cv2.IMREAD_GRAYSCALE or 0
    # flag = cv2.IMREAD_UNCHANGED or -1 (alpha channel)
    img_BGR_np = cv2.imread(file_path, flag)
    # H = img_BGR_np.shape[0]
    # W = img_BGR_np.shape[1]
    return img_BGR_np


def CV2_BGR_channels(file_path):
    """
    Read image using OpenCV's cv2.imread() 
    It returns three color channels (BGR).

    param file_path: the full path of image
    :return: Blue, Green, Red channel
    """
    flag = cv2.IMREAD_COLOR
    img_BGR_np = cv2.imread(file_path, flag)
    B = img_BGR_np[:, :, 0]  # get blue channel
    G = img_BGR_np[:, :, 1]  # get green channel
    R = img_BGR_np[:, :, 2]  # get red channel
    return B, G, R


def CV2_convert_to_RGB(img_BGR_np):
    img_RGB_np = cv2.cvtColor(img_BGR_np, cv2.COLOR_BGR2RGB)
    return img_RGB_np


def CV2_load_image(path, channel=3, resize_width=128, resize_height=128):
    """
    Load, resize image and convert it to numpy.ndarray. 
    Notes that:
    - color channels are BGR and 
    - color space is normalized from [0, 255] to [-1, 1].

    param path: the full path of image
    param channel: 1 (gray scale), 3 (color)
    param resize_width: resized width
    param resize_height: resized height
     interpolation: cv2.INTER_LINEAR (default), cv2.INTER_CUBIC, cv2.INTER_AREA
    :return: numpy.ndarray in range of [-1, 1]
    """
    if channel == 1:
        image_decoded = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        image_decoded = cv2.imread(path)

    img = cv2.resize(image_decoded, (resize_width, resize_height))
    img_np = img.astype(dtype=np.float32)
    img_np = (img_np / 127.5) - 1.0
    return img_np


def CV2_np_convert_to_1_1_np(img_np):
    """
    Convert numpy.ndarray (img_np) from [0, 255] to [-1, 1]
    
    param img_np: numpy.ndarray in range of [0, 255]
    :return: numpy.ndarray in range of [-1, 1]
    """
    img_np = (img_np / 127.5) - 1.0
    return img_np


# ======================================================
# ======================================================
# https://stackoverflow.com/questions/50966204/convert-images-from-1-1-to-0-255
def CV2_np_convert_to_0_255_np(img_np):
    """
    Convert numpy.ndarray (img) from [-1, 1] to [0, 255]
    
    param img_np: numpy.ndarray in range of [-1, 1]
    :return: numpy.ndarray in range of [0, 255]
    """
    img_np = cv2.normalize(img_np, None, alpha=0, beta=255,
                           norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img_np = img_np.astype(np.uint8)
    return img_np


def CV2_tensor_convert_to_0_255_np(img_tensor):
    # convert tensor (img) from [-1, 1] to [0, 255]
    """
    Convert tensor image from [-1, 1] to [0, 255]
    
    param img_tensor: tensor image in range of [-1, 1]
    :return: numpy.ndarray in range of [0, 255]
    """
    # 1. convert tensor to numpy.ndarray
    img_np = img_tensor.numpy()
    img_np = CV2_np_convert_to_0_255_np(img_np)
    return img_np


"""
PIL color image:
- Three color channels: Red-Green-Blue (RGB)
    img_PIL = Image.open('data/src/lena.jpg')
- Size: Width, Height (W, H)
    w, h = img_PIL.size
    w = img_PIL.width
    h = img_PIL.height

PIL gray image:
- im_gray = Image.open('data/src/lena.jpg').convert('L')
- Size: Width, Height (W, H)
    w, h = img_PIL.size
    w = img_PIL.width
    h = img_PIL.height
"""


def PIL_open(filepath):
    """
    Open image using PIL's Image.open()
    Returns: a PIL image object.
    Notes that:color channels are RGB

    param filepath: the full path of image
    :return: a PIL image object.
    """
    img_RGB = Image.open(filepath)
    # img_RGB.height, img_RGB.width
    return img_RGB


def load_image(path, channel=3, resize_width=128, resize_height=128, format="CV2"):
    """
    param path: full path to image
    param format: "CV2", "PIL"
    """
    if not os.path.exists(path):
        print("File does not exist.")
        return None
    if channel not in [1, 3]:
        raise ValueError("Invalid channel value. Channel must be either 1 or 3.")
    if not isinstance(resize_width, int) or not isinstance(resize_height,
                                                           int) or resize_width <= 0 or resize_height <= 0:
        raise ValueError("resize_width and resize_height must be positive integers")
    if format not in ['PIL', 'CV2']:
        raise ValueError("Invalid format parameter. Must be either 'PIL' or 'CV2'.")
    try:
        if format == "PIL":
            """
            Open image using PIL's Image.open()
            Returns: a PIL image object. Color channels are RGB
            """
            if channel == 1:
                raw_frame = Image.open(path).convert('L')
            else:
                raw_frame = Image.open(path).convert('RGB')
            raw_frame = raw_frame.resize((resize_width, resize_height))
        elif format == "CV2":
            """
            Load, resize image and convert it to numpy.ndarray. 
            - color channels are BGR 
            - color space is normalized from [0, 255] to [-1, 1].
            """
            if channel == 1:
                image_decoded = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            else:
                image_decoded = cv2.imread(path)

            img = cv2.resize(image_decoded, (resize_width, resize_height))
            img = img.astype(dtype=np.float32)
            raw_frame = (img / 127.5) - 1.0

            # Release memory after using unnecessary variables
            del image_decoded, img
        return raw_frame
    except Exception as e:
        raise e
