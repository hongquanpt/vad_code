import ipywidgets as widgets
from IPython.display import display
import cv2

def CV2_imread(file_path, flag= cv2.IMREAD_COLOR):
    # https://www.geeksforgeeks.org/python-opencv-cv2-imread-method/
    # flag = cv2.IMREAD_COLOR or 1 (default)
    # flag = cv2.IMREAD_GRAYSCALE or 0
    # flag = cv2.IMREAD_UNCHANGED or -1 (alpha channel)
    img_BGR_np = cv2.imread(file_path, flag)
    return img_BGR_np
    
def save_video(frames, fps=30):
    '''
    param frames: a list of numpy array
    '''
    # Create a widget to display the video
    out = widgets.Output(layout={'border': '1px solid black'})
    #display(out)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width, _ = frames[0].shape
    video_writer = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    # Display each frame and write it to the video
    for frame in frames:
        with out:
            display(widgets.Image(value=cv2.imencode('.jpg', frame)[1].tobytes()))
            video_writer.write(frame)

    # Release the VideoWriter
    video_writer.release()

def check_fourcc_codec(video_path = '01_001.mp4' ):

    # Open the video file for reading
    # output: FourCC Codec (String): hevc; FourCC Codec (Integer): 1668703592
    #video_path = '01_001.mp4'  
    
    # output: cv2.VideoWriter_fourcc(*'mp4v') FourCC Codec (String): FMP4
    #video_path = 'output.mp4' 
    
    # output: cv2.VideoWriter_fourcc(*'xvid') FourCC Codec (String): FMP4
    #video_path = 'output.avi' 
    
    video_capture = cv2.VideoCapture(video_path)
    
    # Get the FourCC codec of the video
    fourcc = int(video_capture.get(cv2.CAP_PROP_FOURCC))
    
    # Convert the FourCC code to a string
    fourcc_str = chr(fourcc & 255) + chr((fourcc >> 8) & 255) + chr((fourcc >> 16) & 255) + chr((fourcc >> 24) & 255)
    
    # Release the video capture object
    video_capture.release()
    
    # Print the FourCC codec as a string and as an integer
    print(f"FourCC Codec (String): {fourcc_str}")
    print(f"FourCC Codec (Integer): {fourcc}")

def main():
    # Path to the directory containing your frames
    frames_directory = '/home/dataset/ped2/training/01'
    
    # Get the list of frame files in the directory
    frame_files = [f for f in sorted(os.listdir(frames_directory)) if f.endswith('.jpg')]
    
    # 1. Load your frames into a list (frames_list)
    frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(frames_directory, frame_file)
        img = CV2_imread(frame_path)
        frames.append(img)
    
    # 2. Display the frames as a video
    save_video(frames, fps=30)

if __name__ == '__main__':
    main()