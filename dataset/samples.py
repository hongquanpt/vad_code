import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, StanfordCars, MNIST


# ============================================================
# CIFAR10
# ============================================================
def get_dataset_cifar10():
    cifar10_dataset = CIFAR10('/home/dataset/CIFAR10',
                              train=True,
                              download=True,
                              transform=transforms.ToTensor())
    cifar10_dataloader = torch.utils.data.DataLoader(cifar10_dataset, batch_size=1)
    return cifar10_dataset, cifar10_dataloader


# ============================================================
# CIFAR100
# ============================================================
def get_dataset_cifar100():
    cifar100_dataset = CIFAR100('/home/dataset/CIFAR100',
                                train=True,
                                download=True,
                                transform=transforms.ToTensor())
    cifar100_dataloader = torch.utils.data.DataLoader(cifar100_dataset, batch_size=1)
    return cifar100_dataset, cifar100_dataloader


# ============================================================
# StanfordCars
# ============================================================
def get_dataset_stanfordCars():
    stanfordCars_dataset = StanfordCars('/home/dataset/StanfordCars',
                                        download=True,
                                        transform=transforms.ToTensor())
    stanfordCars_dataloader = torch.utils.data.DataLoader(stanfordCars_dataset, batch_size=1)
    return stanfordCars_dataset, stanfordCars_dataloader


# ============================================================
# MNIST
# ============================================================
def get_dataset_mnist():
    mnist_dataset = MNIST('/home/dataset/MNIST',
                          train=True,
                          download=True,
                          transform=transforms.ToTensor())
    mnist_dataloader = torch.utils.data.DataLoader(mnist_dataset, batch_size=1)
    return mnist_dataset, mnist_dataloader
# ============================================================
