import os
import natsort
import cv2
import numpy as np

import torch
from libs.flownet_networks.flownet2_models import FlowNet2


# ==========================================================
class Folder_Capture:
    """
    Read all images in a image folder step by step
    For example: folder 'ped2/training/01'
    """

    def __init__(self, source) -> None:
        """
        param source: a full path to a image directory
        """
        super().__init__()
        # đường dẫn đến folder chứa các ảnh
        self.source = source

        # danh sách các ảnh trong folder
        self.img_List = natsort.natsorted(os.listdir(source))
        self.index = 0
        _, img = self.read()  # hàm đọc 1 ảnh theo index của ảnh đó
        self.index -= 1
        if img is not None:
            self.shape = img.shape

    def read(self):
        if self.index < len(self.img_List):
            img = cv2.imread(os.path.join(self.source, self.img_List[self.index]))
        try:
            img.shape
            ret = True
        except:
            ret = False
        self.index += 1
        return ret, img

    def isOpened(self):
        return self.index < len(self.img_List)

    def get(self, i=0):
        if i == 1:
            return self.index
        if i == 7:
            return len(self.img_List)
        if i == 4:
            return self.shape[0]
        if i == 3:
            return self.shape[1]

    def release(self):
        pass


# ==========================================================
# Some methods to extract difference between two frames
# ==========================================================
class Dynamic_Image:
    """
    A method to generate Dynamic Image (Motion Image/ Motion Information)
    """

    def __init__(self, cap, t, resize) -> None:
        super().__init__()
        self.t_Frames = []  # a sequence of frames
        self.cap = cap  # Folder_Capture object
        self.t = t  # num of frames
        self.resize = resize  # resize ratio
        self.shape = (cap.get(3), cap.get(4)) if self.resize == 1.0 else (
        int(self.resize * cap.get(3)), int(self.resize * cap.get(4)))

    def update(self):
        ret, frame = self.cap.read()  # read next frame
        dimg = None
        if ret:
            frame = frame if self.resize == 1.0 else cv2.resize(frame, self.shape)
            dimg = np.zeros(frame.shape)  # init a dynamic image
            self.t_Frames.append(frame)  # add frame to list
            T = min(self.t, len(self.t_Frames))  # compare t with the number of added frames
            for i in range(1, T + 1):
                temp = 0
                for j in range(i, T + 1):
                    temp += (2 * j - T - 1) / j

                dimg += temp * self.t_Frames[i - 1]

            if T > 1:
                dimg -= (dimg[np.unravel_index(dimg.argmin(), dimg.shape)])
                dimg /= (dimg[np.unravel_index(dimg.argmax(), dimg.shape)])
                dimg = 255 * dimg

            if len(self.t_Frames) > self.t:
                self.t_Frames.pop(0)
            return ret, dimg.astype(np.uint8)
        else:
            return ret, dimg


# ==========================================================
class Frame_Difference:
    def __init__(self, cap, t, resize) -> None:
        super().__init__()
        self.t_Frames = []
        self.cap = cap
        self.t = t
        self.resize = resize
        self.shape = (cap.get(3), cap.get(4)) if self.resize == 1.0 else (
        int(self.resize * cap.get(3)), int(self.resize * cap.get(4)))

    def update(self):
        ret, frame = self.cap.read()
        gradient = None
        if ret:
            frame = frame / 255.0
            frame = frame if self.resize == 1.0 else cv2.resize(frame, self.shape)
            gradient = np.zeros(frame.shape)
            self.t_Frames.append(frame)
            gradient = frame - self.t_Frames[0]
            if len(self.t_Frames) > 1:
                gradient -= (gradient[np.unravel_index(gradient.argmin(), gradient.shape)])
                gradient /= (gradient[np.unravel_index(gradient.argmax(), gradient.shape)])
                gradient = 255 * gradient

            if len(self.t_Frames) > self.t:
                self.t_Frames.pop(0)
            return ret, gradient.astype(np.uint8)
        else:
            return ret, gradient


# ==========================================================
# GMM_DAE: https://github.com/wufan-tb/gmm_dae
# ==========================================================
def generate_MotionImages(dataset_path='dataset/ped2/training',
                          motion_type='DI', num_frames=10, resize=1):
    parent_path = os.path.dirname(dataset_path)  # 'home/anhnam/Data/VAD/dataset/ped2'
    current_dirname = os.path.basename(dataset_path)  # 'training'

    # MotionImageGenerator: is a dictionary of methods to generate motion images
    MotionImageGenerator = {'DI': Dynamic_Image,
                            'FD': Frame_Difference
                            }

    dirs = [d for d in os.listdir(dataset_path) if not d.startswith('.')]
    videos_names = natsort.natsorted(dirs)  # ['01', '02',.., '15', '16']
    # print(f'videos_names = {videos_names}')
    for video_name in videos_names:
        # '/ped2/tranining/01'
        video_path = os.path.join(dataset_path, video_name)
        cap = Folder_Capture(video_path)
        process = MotionImageGenerator[motion_type](cap, num_frames, resize)

        # 'training_DI/01'
        motion_dir = os.path.join(parent_path,
                                  current_dirname + '_' + motion_type, video_name)
        os.makedirs(motion_dir, exist_ok=True)

        # export motion_img (motion images)
        index = 1
        while cap.isOpened():
            ret, motion_img = process.update()
            if ret:
                cv2.imwrite(os.path.join(motion_dir, '{:03d}.jpg'.format(index - 1)), motion_img)
                index += 1
            else:
                break
        cap.release()
        print(f"New Motion Directory: {motion_dir}!")


# ==========================================================
# Flownet2.0: extract Optical Flow
# ==========================================================
def generate_OpticalFlow_Flownet2(dataset_path='dataset/ped2/training',
                                  model_path='libs/flownet_networks/weights/FlowNet2_checkpoint.pth.tar'):
    motion_type = 'OF_FL2'  # OpticalFlow Flownet2.0
    parent_path = os.path.dirname(dataset_path)  # 'home/anhnam/Data/VAD/dataset/ped2'
    current_dirname = os.path.basename(dataset_path)  # 'training'
    dirs = [d for d in os.listdir(dataset_path) if not d.startswith('.')]
    videos_names = natsort.natsorted(dirs)  # ['01', '02',.., '15', '16']
    # print(f'videos_names = {videos_names}')

    # Initialize the FlowNet2 model
    flownet2 = FlowNet2()
    pretrained_dict = torch.load(model_path)['state_dict']
    model_dict = flownet2.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    flownet2.load_state_dict(model_dict)
    flownet2.cuda()
    flownet2.eval()
    WIDTH = 512
    HEIGHT = 384
    for video_name in videos_names:
        # Create an output folder to save the optical flow images
        # 'training_OF_FL2/01' 
        motion_dir = os.path.join(parent_path,
                                  current_dirname + '_' + motion_type, video_name)
        os.makedirs(motion_dir, exist_ok=True)

        # Get a list of image files in the folder
        # '/ped2/tranining/01'
        video_path = os.path.join(dataset_path, video_name)
        include_ext = [".png", ".jpg", "jpeg", ".bmp"]
        image_files = []
        for el in natsort.natsorted(os.listdir(video_path)):
            if os.path.isfile(os.path.join(video_path, el)) \
                    and not el.startswith('.') \
                    and any([el.endswith(ext) for ext in include_ext]):
                file_path = os.path.join(video_path, el)
                image_files.append(file_path)

        # Iterate through the image pairs and compute optical flow
        for i in range(len(image_files) - 1):
            # Load the current and next frames
            im1 = cv2.imread(image_files[i])  # numpy (H, W, C=3)
            im2 = cv2.imread(image_files[i + 1])

            im1 = cv2.resize(im1, (WIDTH, HEIGHT))
            im2 = cv2.resize(im2, (WIDTH, HEIGHT))
            ims = np.array([im1, im2]).astype(np.float32)  # numpy[2,h',w',3]

            # convert ims to tensor, then add more dimension
            ims = torch.from_numpy(ims).unsqueeze(0)

            # tensor[bs,2,H,W,3] -> tensor[bs,3,2,H,W]
            ims = ims.permute(0, 4, 1, 2, 3).contiguous().cuda()
            # print(f'ims.shape = {ims.shape}') #  torch.Size([1, 3, 2, 384, 512])
            # print('ims = ', ims)

            # Compute optical flow
            # ims: [0.0 - 255.0]
            with torch.no_grad():
                flow = flownet2(ims).cpu().data
            print('flow1 = ', flow)
            # Convert optical flow tensor to numpy array [h,w,2]
            flow = flow[0].cpu().numpy().transpose(1, 2, 0)

            # ========================================
            # Save the optical flow to numpy file
            # ========================================
            output_filename = os.path.join(motion_dir, f'{i:03d}.npy')

            try:
                np.save(output_filename, flow)
                print(f'Save the optical flow to numpy file: {output_filename}')
            except:
                print(f'Cannot save the optical flow to numpy file: {output_filename}')
                return 0

            # ========================================
            # Save the optical flow as an image
            # ========================================
            # Fill third channel with zeros
            flow = np.concatenate((flow, np.zeros((HEIGHT, WIDTH, 1))), axis=2)
            # print('flow2 = ', flow) # maybe: [-1,1]
            flow_image = (flow + 1) * 127.5  # [-1,1] to [0,255]
            flow_image = flow_image.astype(np.uint8)
            # print('flow_image = ', flow_image)

            # print(f'flow_image.shape = {flow_image.shape}') # (384, 512, 3)
            flow_image = cv2.cvtColor(flow_image, cv2.COLOR_RGB2BGR)
            output_filename = os.path.join(motion_dir, f'{i:03d}.png')
            try:
                cv2.imwrite(output_filename, flow_image)
                print(f'Save the optical flow to image file: {output_filename}')
            except:
                print(f'Cannot save the optical flow to image file: {output_filename}')
                return 0
