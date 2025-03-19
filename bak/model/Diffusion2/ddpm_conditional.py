import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim

from .utils import *
from .modules import UNet_conditional, EMA

'''
1) Classifier Free Guidance (CFG): used to improve Generations if we train with classes
- it can help to avoid posterior collapse whic simply means that the
model ignores the conditional information and just generates any image.
- cifar10 (32x32) but I'm using an upscaled version (64x64).
- there are plenty ways to condition a model but one of the easiest to
implement and one that works reasonably well is just add condition 
information of the label to timesteps t.
- define UNet_conditional() class and add a new num_classes argument,
and create the label embedding for all classes. Note: it have the same 
number of Dimensions as the time embedding.
- in forward(): we'll add their embedding to the timestep embedding:
    t+= self.label_emb(y)
- How it works?:
+ During training for like 10 of the time we'll try unconditionally 
that way the model learns to do both conditional and unconditional sampling.
+ During sampling, we'll sample also both ways but linearly interpolate away
from the unconditional sample towards the conditional one, and we'll do that
in every iteration

2) Exponential Moving Average (EMA): a way of enforcing a smoother training 
- it literally smooths the trajectory of the model updates
- so if the training is really noisy and the direction of the optimization
changes a lot, EMA can smooth this trajectory and lead to a more robust outcome
since it's not so susceptible to outliers in terms of model updates as the main model.
- EMA works by making a copy of the initial model weights and then updating these
based on the moving average from the main model.
- formula for updating a single weight ...
- Code: I create a new EMA class takes beta as an argument
we'll let the EMA updates start after a certain number of iterations to give
the main model a quick warm-up. 
- During the warm-up we'll always just reset the EMA model parameters to the main one.
- After the warm-up we'll then always update the weights by iterating over all parameters
and apply the formula.

'''

class Diffusion_conditional:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample_images(self, model, n, labels, cfg_scale=3):
        print(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            # create initial n images by sampling from Normal distribution
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            
            # a big loop going over all 1000 timesteps in a reversed order
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                # create a tensor of length n with the current timestep i
                t = (torch.ones(n) * i).long().to(self.device)
                
                # feed timestep i into the model together with the current images
                predicted_noise = model(x, t, labels)
                
                # Classifier Free Guidance (CFG)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                    
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
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
