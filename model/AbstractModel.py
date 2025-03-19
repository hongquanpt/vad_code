from abc import ABC, abstractmethod
import os
import torch
import torch.nn as nn
from model.util.optimizer import get_optimizer, get_scheduler


# https://python-course.eu/oop/the-abc-of-abstract-base-classes.php
class AbstractModel(ABC):
    def __init__(self, cfg, device, logger, run):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.run = run
        self.name = cfg.MODEL.name
        self.type = cfg.MODEL.type

        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None
        self.epoch = 1
        self.loss_epoch = 0.0
        self.logger.info('Call: AbstractModel.__init__()')

    '''
    https://nttuan8.com/bai-6-luu-va-load-model-trong-pytorch/
    Cách 1: Lưu state_dict của model (nên chọn cách này)
    - Cú pháp:
        + save: 
            torch.save(model.state_dict(), PATH)
        + load: Khi load model thì mình cần dựng lại kiến trúc của model trước, 
                sau đó sẽ gọi hàm để load state_dict vào model.
            model = Net()
            model.load_state_dict(torch.load(PATH))
    Cách 2: Lưu cả model (không khuyến khích)
    - Cú pháp: 
        + save: 
            torch.save(mode, PATH)
        + load: 
            model = torch.load(PATH)
    - Khi mọi người lưu cả model thì Pytorch sẽ dùng pickle module của Python để lưu. 
    - Tuy nhiên, pickle thì không lưu trực tiếp model class 
      (class định nghĩa model, dùng để định nghĩa kiến trúc model) 
      mà lưu đường dẫn tới file chứa model class. 
      Thế nên khi load model, nếu mình refactor code và đường dẫn đến file chứa model class thay đổi thì code sẽ lỗi và không load model lên được.
    '''

    @abstractmethod
    def save_model(self, model_filepath):
        self.logger.info('call: save_model() in AbstractModel')
        try:
            torch.save(self.model, model_filepath)
            self.logger.info('Success: Model saved')
        except:
            self.logger.info('Error: Model not saved!!!')
            return 0
        return 1

    @abstractmethod
    def load_model(self, model_filepath):
        self.logger.info('call: load_model() in AbstractModel')
        self.model = torch.load(model_filepath)
        return self.model

    @abstractmethod
    def save_model_state_dict(self, model_filepath):
        self.logger.info('call: save_model_state_dict() in AbstractModel')
        try:
            torch.save(self.model.state_dict(), model_filepath)
            self.logger.info('Success: Model_state_dict saved')
        except:
            self.logger.info('Error: Model_state_dict not saved!!!')
            return 0
        return 1

    '''
    Khi load model:
    - Đầu tiên, cần dựng lại kiến trúc của model trước.
    - Sau đó, gọi hàm để load state_dict vào model.
    '''

    @abstractmethod
    def load_model_state_dict(self, model_filepath):
        self.logger.info('call: load_model_state_dict() in AbstractModel')
        state_dict = torch.load(model_filepath)
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']
            self.model.load_state_dict(state_dict)
        else:
            self.model.module.load_state_dict(state_dict)
        return self.model

    @abstractmethod
    def save_checkpoint(self, checkpoint_filepath):
        self.logger.info('call: save_checkpoint() in AbstractModel')
        self.logger.info(f'save_checkpoint: {checkpoint_filepath}')
        try:
            torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'loss': self.loss_epoch
                        }, checkpoint_filepath)
            self.logger.info(f'Success: save_checkpoint: {checkpoint_filepath}\n')
        except FileNotFoundError:
            self.logger.info(f"Error: Checkpoint file not found at {checkpoint_filepath}")
        except Exception as e:
            self.logger.info(f'Error: save_checkpoint: {checkpoint_filepath} !!!\n')
            self.logger.info(f"Error reason: {str(e)}")
        return 1

    @abstractmethod
    def load_checkpoint(self, checkpoint_filepath):
        self.logger.info('call: load_checkpoint() in AbstractModel')
        try:
            # The map_location parameter specifies where the tensors will be loaded, 
            # particularly useful when loading models saved on a different device (e.g., CUDA GPU)
            # onto a different device (e.g., CPU or a different GPU).
            checkpoint = torch.load(checkpoint_filepath,
                                    map_location=self.cfg.SYSTEM.device)
            self.epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # check if optimizer_state_dict exists or not
            self.optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', {}))
            # Try moving optimizer state to the GPU memory manually after loading it from the checkpoint.
            # torch.is_tensor(v): check if a variable is a PyTorch tensor and not any of its subclasses
            # isinstance(v, torch.Tensor):  If include subclass instances as well, 
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

            # check if scheduler_state_dict exists or not
            self.scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', {}))
            self.loss_epoch = checkpoint['loss']
            self.logger.info(f'Success: load_checkpoint: {checkpoint_filepath} \n')
            return self.epoch
        except FileNotFoundError:
            self.logger.info(f"Error: Checkpoint file not found at {checkpoint_filepath}")
            return 0
        except Exception as e:
            self.logger.info(f'Error: load_checkpoint: {checkpoint_filepath} !!!\n')
            self.logger.info(f"Error reason: {str(e)}")
            return -1

    def train_epoch(self, epoch, train_dataloader):
        self.logger.info('call: train_epoch() in AbstractModel')
        # =========================
        # Activate train mode
        # =========================
        self.epoch = epoch + 1
        self.model = self.model.to(self.device)
        self.model.train()
        loss_epoch = 0
        return loss_epoch

    def train(self, train_dataloader, checkpoint_filepath, checkpoint_dir_path):
        self.logger.info('call: train() in AbstractModel')
        begin_epoch = self.cfg.TRAIN.begin_epoch
        end_epoch = self.cfg.TRAIN.end_epoch
        if self.cfg.TRAIN.resume:
            # ==================================
            self.logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ')
            self.logger.info(">>>>>>>> RESUME TRAINING: >>>>>>>>>>>")
            self.logger.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ')
            # ==================================
            epoch = self.load_checkpoint(checkpoint_filepath)
            if epoch == -1:
                return
            begin_epoch = epoch
        # ==================================
        # Training process
        # ==================================
        # if torch.cuda.device_count() > 1:
        #    self.model = nn.DataParallel(self.model, device_ids=self.cfg.SYSTEM.gpus)
        self.model = self.model.to(self.device)
        self.model.train()
        on_timer = True
        starter_train = 0
        ender_train = 0
        if on_timer:
            starter_train, ender_train = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter_train.record()
        for epoch in range(begin_epoch, end_epoch):
            self.logger.info(f'epoch[{epoch + 1}/{end_epoch}]')
            print(f'epoch[{epoch + 1}/{end_epoch}]')
            starter_epoch = 0
            ender_epoch = 0
            if on_timer:
                starter_epoch, ender_epoch = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                starter_epoch.record()

            # self.loss_epoch = self.train_epoch(epoch, train_dataloader)
            self.train_epoch(epoch, train_dataloader)
            # self.logger.info('Loss Epoch {:.9f}'.format(self.loss_epoch))

            if on_timer:
                ender_epoch.record()
                torch.cuda.synchronize()  # Waits for everything to finish running
                total_seconds_epoch = starter_epoch.elapsed_time(ender_epoch) * 1e-3  # millisecond to second

                # Calculate minutes and remaining seconds
                hours = total_seconds_epoch // 3600
                minutes = (total_seconds_epoch % 3600) // 60
                seconds = total_seconds_epoch % 60
                self.logger.info(f'Time for epoch: {hours} hours, {minutes} minutes and {seconds} seconds')
                print(f'Time for epoch: {hours} hours, {minutes} minutes and {seconds} seconds')

            if self.cfg.TRAIN.lr_scheduler.use:
                self.scheduler.step()
                cur_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info('lr_scheduler: {lr:.6f}'.format(lr=cur_lr))
                print('lr_scheduler: {lr:.6f}'.format(lr=cur_lr))
            # ==================================
            # Save the trained model by sequence
            # ==================================
            if (epoch + 1) % self.cfg.TRAIN.save_freq == 0:
                checkpoint_filepath = os.path.join(checkpoint_dir_path, f'epoch_{epoch + 1}.pth')
                self.save_checkpoint(checkpoint_filepath)

        # ==================================
        # Save the final trained model
        # ==================================
        if on_timer:
            ender_train.record()
            torch.cuda.synchronize()  # Waits for everything to finish running
            total_seconds_train = starter_train.elapsed_time(ender_train) * 1e-3  # millisecond to second

            # Calculate hours, minutes, and seconds
            hours = total_seconds_train // 3600
            minutes = (total_seconds_train % 3600) // 60
            seconds = total_seconds_train % 60
            self.logger.info(f'Time for train:{hours} hours, {minutes} minutes, and {seconds} seconds')
            print(f'Time for train:{hours} hours, {minutes} minutes, and {seconds} seconds')

        checkpoint_filepath = os.path.join(checkpoint_dir_path, f'epoch_{end_epoch}.pth')
        self.save_checkpoint(checkpoint_filepath)

    def test(self, test_dataloader, video_labels, checkpoint_filepath, visualization_dir_path):
        # =========================
        # Load the pretrained model
        # =========================
        epoch = self.load_checkpoint(checkpoint_filepath)
        if epoch == -1:
            return
        self.model = self.model.to(self.device)
        # =========================
        # Activate evaluation mode
        # =========================
        self.model.eval()
