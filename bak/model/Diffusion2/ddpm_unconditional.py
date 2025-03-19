import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim

from .utils import *
from .modules import UNet

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        """
        1. Setting up the noising schedule
        """
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    
    def noise_images(self, x, t):
        """
        2. Function for noising images
        - x: the image
        - t: timestep t
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon
    
    def sample_timesteps(self, n):
        """
        Sampling some timesteps (in algorthm 1 for training)
        - n: the number of timesteps
        """
        # Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    
    def sample_images(self, model, n):
        """
        # 3. Sampling images (recoverd by model to check how model learned)
        - model: the model which we'll use for sampling 
        - n: the number images we want to sample.
        """
        print(f"Sampling {n} new images....")
        
        # Start Algorithm 2. Sampling from DDPM paper
        model.eval()
        with torch.no_grad():
            # create initial n images by sampling from Normal distribution
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            
            # a big loop going over all 1000 timesteps in a reversed order
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                # create a tensor of length n with the current timestep i
                t = (torch.ones(n) * i).long().to(self.device)
                
                # feed timestep i into the model together with the current images
                predicted_noise = model(x, t)
                
                # the last thing we'll need is noise which we add and scale with the variants later
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                
                # we only need noise for the timestep > 1
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                    
                # finally, we alter images and remove a little bit of noise 
                # (see line 4 in Algorithm 2. Sampling)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        
        # set the model back to training
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2 # convert images into (-1, 1)
        x = (x * 255).type(torch.uint8) # convert images into (0, 255)
        return x