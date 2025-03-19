import os
import time
import logging
# SummaryWriter: key object for writing information to TensorBoard.
# from torch.utils.tensorboard import SummaryWriter


def make_output_dir(cfg):
    time_str = time.strftime('%Y_%m_%d_%H_%M_%S')  # 2023_08_07_10_34

    # Get the current working directory ("/home/asus/DATA/VAD")
    output_dir = os.path.join(os.getcwd(), 'output', cfg.SYSTEM.phase,
                              cfg.MODEL.name, cfg.MODEL.method, cfg.DATASET.name,
                              f'session_{time_str}')
    return output_dir


'''
Nguồn Tham khảo:
- Đọc kỹ bài này: 
    - https://realpython.com/python-logging/#:~:text=Logger%20%3A%20This%20is%20the%20class,%2C%20the%20message%2C%20and%20more.
- Tham khảo:
    - https://machinelearningmastery.com/logging-in-python/
    - https://www.geeksforgeeks.org/logging-in-python/

Định nghĩa: 
- Logging is a very useful tool in a programmer’s toolbox. It can help you develop a better understanding of the flow of a program and discover scenarios that you might not even have thought of while developing.
- Python provides a logging system as a part of its standard library, so you can quickly add logging to your application.
- The logging module provides you with a default logger that allows you to get started without needing to do much configuration. 

How to use:
    import logging

1) By default, there are 5 standard levels indicating the severity of events. The defined levels, in order of increasing severity, are the following: DEBUG; INFO; WARNING; ERROR; CRITICAL:

    logging.debug('This is a debug message')
    logging.info('This is an info message')

    logging.warning('This is a warning message')
    logging.error('This is an error message')
    logging.critical('This is a critical message')

    Output in shell: (in format: the level, name, and message separated by a colon (:)
    WARNING:root:This is a warning message
    ERROR:root:This is an error message
    CRITICAL:root:This is a critical message
    root is the name the logging module gives to its default logger.

2) basicConfig(**kwargs) method to configure the logging. Some of the commonly used parameters for basicConfig() are the following:
    level: The root logger will be set to the specified severity level.
    filename: This specifies the file.
    filemode: If filename is given, the file is opened in this mode. The default is a, which means 'append'.
    format: This is the format of the log message.
'''


def make_logger(output_dir: str, logger_dir_name: str):
    # 1. create output log directory
    log_dir_path = make_dir_path(output_dir, logger_dir_name)

    # 2. create log file and set up the basic of the logger
    log_file = os.path.join(log_dir_path, 'logger.log')
    logging.basicConfig(level=logging.INFO, filename=log_file)
    print(f'=> Creating the [log_file]: {log_file}')

    # logger
    '''
    logger = logging.getLogger()
    fmt = '%(asctime)-15s:%(message)s' # format logger
    datefmt = '%Y-%m-%d-%H:%M'
    formatter = logging.Formatter(fmt=fmt,datefmt=datefmt)
    level=logging.DEBUG
    logger.setLevel(level)
    '''
    logger = logging.getLogger('PIL')
    fmt = '%(asctime)-15s:%(message)s'  # format logger
    datefmt = '%Y-%m-%d-%H:%M'
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    level = logging.INFO
    logger.setLevel(level)

    # console
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)

    # file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # addHandlder: console, file_handler
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger, log_file


'''
Nguồn Tham khảo:
- https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
- https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
- https://www.learnpytorch.io/07_pytorch_experiment_tracking/
- https://tek4.vn/tensorboard-voi-pytorch-lap-trinh-neural-network-voi-pytorch

Định nghĩa:
- TensorBoard: Bộ công cụ trực quan hóa của TensorFlow
- TensorBoard cung cấp hình ảnh và công cụ cần thiết cho các thử nghiệm học máy:
    + Tracking và visualizing các chỉ số như mất mát và độ chính xác.
    + Visualizing the model graph (ops and layers) 
    + Xem biểu đồ trọng số, độ lệch hoặc các yếu tố khác khi chúng thay đổi theo thời gian
    + Chiếu các bản nhúng vào không gian có chiều thấp hơn
    + Hiển thị dữ liệu hình ảnh, văn bản và âm thanh
    + Lập hồ sơ chương trình TensorFlow
    + Và nhiều hơn nữa.
    
Install:
    pip install tensorboard
    
How to use:
1) Import and add
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=tensorboard_dir_path)
    
    https://pytorch.org/docs/stable/tensorboard.html
    writer.add_scalars(main_tag, tag_scalar_dict), where:
     - main_tag (string) - the name for the scalars being tracked (e.g. "Loss")
     - tag_scalar_dict (dict) - a dictionary of the values being tracked (e.g. {"train_loss": 0.3454})
    
    writer.add_graph() which tracks the computation graph of your model.

    For example,
        # Add loss results to SummaryWriter
        writer.add_scalars(tag="Loss", 
                           scalar_value={"train_loss": train_loss,
                                            "test_loss": test_loss},
                           global_step=epoch)
                           
        # Track the PyTorch model architecture
        writer.add_graph(model=model, 
                         # Pass in an example input
                         input_to_model=torch.randn(32, 3, 224, 224).to(device))
        
        # Close the writer
        writer.close()

2) Start Tensorboard by command: tensorboard --logdir=runs
- specifying the root log directory you used above. Argument 'logdir' points to directory where TensorBoard will look to find event files that it can display. 
- TensorBoard will recursively walk the directory structure rooted at logdir, looking for .*tfevents.* files.

3) Share TensorBoard dashboards: https://tensorboard.dev/

4) Tensorboard has 4 tabs: SCALARS, IMAGES, GRAPHS, PROJECTOR
- https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-tensorboard-with-pytorch.md
- https://pytorch.org/docs/stable/tensorboard.html
5) Example,
def train_model(iter):
    for epoch in range(iter):
        y1 = model(x)
        loss = criterion(y1, y)
        writer.add_scalar(tag="Loss/train", 
                            scalar_value=loss, 
                            global_step=epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

train_model(10)
writer.flush() # to make sure that all pending events have been written to disk.
writer.close() # If you do not need the summary writer anymore, call close() method.

'''


def make_writer(output_dir: str, tesorboard_dir_name: str):
    # 1. create output tensorboard directory 
    tensorboard_dir_path = make_dir_path(output_dir, tesorboard_dir_name)

    # 2. Create a tensorboard writer and tensorboard folder to track our model/data
    writer = SummaryWriter(log_dir=tensorboard_dir_path)
    return writer


def make_dir_path(root_dir: str, dir_name: str):
    dir_path = os.path.join(root_dir, dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        print(f'=> Creating a new directory: {dir_path}')
    else:
        print(f"- The directory [{dir_path}] existed!!!")

    if not os.path.exists(dir_path):
        raise Exception('Something wrong in creating dir_path: {}'.format(dir_path))
    return dir_path
