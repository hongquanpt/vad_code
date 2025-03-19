import torch.nn as nn
from functools import reduce
from operator import mul
import torch
# from Dong Gong's paper code: 
# 2019-Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection
class Flatten(nn.Module):
    def forward(self, input):
        return torch.flatten(input)

class Unflatten(nn.Module):
    def __init__(self, dim, unflattened_size):
        super(Unflatten, self).__init__()
        self.dim = dim
        self.unflattened_size = unflattened_size

    def forward(self, input):
        unflatten = nn.Unflatten(self.dim, self.unflattened_size)
        output = unflatten(input)
        return output

class Encoder_Recon3D(nn.Module):
    def __init__(self, chnum_in):
        super(Encoder_Recon3D, self).__init__()

        # Dong Gong's paper code
        self.chnum_in = chnum_in
        feature_num = 128
        feature_num_2 = 96
        feature_num_x2 = 256
        self.encoder = nn.Sequential(
            nn.Conv3d(self.chnum_in, feature_num_2, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num_2, feature_num, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num_x2, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2), #  [4, 256, 2, 16, 16]
            #Flatten(), # 256x2x16x16 = 131072
            nn.LeakyReLU(0.2, inplace=True)
            
        )
        self.latent_size = 16 # 32768 #65536
        # hidden => mu
        self.fc1 = nn.Linear(131072, self.latent_size)

        # hidden => logvar
        self.fc2 = nn.Linear(131072, self.latent_size)

    def forward(self, x):
        h = self.encoder(x) # output x = [4, 256, 2, 16, 16], h = [65536]
        #return h
        
        #mu, logvar = self.fc1(h), self.fc2(h)
        #return mu, logvar
        
class Decoder_Recon3D(nn.Module):
    def __init__(self, chnum_in):
        super(Decoder_Recon3D, self).__init__()

        # Dong Gong's paper code + Tanh
        self.chnum_in = chnum_in
        feature_num = 128
        feature_num_2 = 96
        feature_num_x2 = 256
        #self.latent_size = 32768
        
        self.decoder = nn.Sequential(
            #nn.Linear(self.latent_size, 131072),
            nn.LeakyReLU(0.2, inplace=True),
            #Unflatten(dim=0, unflattened_size=(256, 2, 16, 16)),
            nn.ConvTranspose3d(feature_num_x2, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num_x2, feature_num, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num, feature_num_2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                               output_padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num_2, self.chnum_in, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1),
                               output_padding=(0, 1, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.decoder(x) # output:[4, 1, 16, 256, 256]
        return x

class Conv3D_AE_Recon(torch.nn.Module):
    def __init__(self):  # for reconstruction
        super(Conv3D_AE_Recon, self).__init__()

        self.reconstruction = True

        # encoder-output_shape: # [4, 256, 2, 16, 16]
        self.encoder = Encoder_Recon3D(chnum_in=1)  # black and white
        
        # decoder-output_shape: [4, 1, 16, 256, 256]
        self.decoder = Decoder_Recon3D(chnum_in=1)  # black and white

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def forward(self, x):
        #mu, logvar = self.encoder(x)
        #z = self.reparameterize(mu, logvar)
        #return self.decoder(z), mu, logvar
        
        #return mu, logvar
        
        z = self.encoder(x)
        #out = self.decoder(z)
        return z