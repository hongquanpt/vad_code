import math

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
https://www.geeksforgeeks.org/initialize-weights-in-pytorch/

https://saturncloud.io/blog/how-to-initialize-weights-in-pytorch-a-guide-for-data-scientists/

https://stackoverflow.com/questions/62246656/how-to-initialize-weights-in-a-pytorch-model

https://www.askpython.com/python-modules/initialize-model-weights-pytorch

https://machinelearningmastery.com/initializing-weights-for-deep-learning-models/

https://pydev.vn/d/85-khoi-tao-weight-cho-deep-learning-neural-networks
'''
# https://github.com/vt-le/astnet
def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()