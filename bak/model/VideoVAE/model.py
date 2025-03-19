import torch
import torch.nn as nn

from .model_modules import Encoder, Decoder, DistributionNet, MLP 

class VideoVAE(nn.Module):
    def __init__(self, z_dim=512, h_dim=512,
                input_size=512*3, hidden_size=512,
                num_layers=1,
                bidirectional=False):
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        
        
        # Encoder
        self.enc = Encoder()
        
        # Posterior
        self.post_q = DistributionNet(in_dim=z_dim,
                                                    h_dim=128, 
                                                    out_dim=512)
        self.post_dy = DistributionNet(in_dim=(z_dim+z_dim+z_dim),
                                       h_dim=128,
                                       out_dim=512)
        
        self.mlp_lstm = MLP(in_dim=512, h_dim=128, out_dim=512)
        
        # Decoder
        self.dec = Decoder()
        
        # LSTM
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # inputs:  input, (h_0, c_0)
        # outputs: output, (h_n, c_n)
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            bidirectional=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, val=0)
          
    def approx_posterior(self, x_enc, phi_h):
        # For Posterior q
        mu_q, logvar_q = self.post_q.encode(x_enc)
        
        # For Posterior dy
        #   phi_q: [mu_q, logvar_q, ]
        #   merge: (phi_q, \phi(h_{t-1}))
        #print(f'mu_q.shape = {mu_q.shape}')
        #print(f'logvar_q.shape = {logvar_q.shape}')
        #print(f'phi_h.shape = {phi_h.shape}')
        
        phi_q_merged = torch.cat([mu_q, logvar_q, phi_h[0]], dim=1)
        z_dy, mu_dy, logvar_dy = self.post_dy(phi_q_merged)

        return mu_q, logvar_q, z_dy, mu_dy, logvar_dy
    
    def lstm_forward(self, mu_q, logvar_q, z_t, h_prev, c_prev):
        lstm_input = torch.cat([z_t, mu_q, logvar_q], dim=1)
        lstm_input = lstm_input.unsqueeze(dim=0)
        
        # Inputs: input, (h_0, c_0); Outputs: output, (h_n, c_n)
        lstm_output, (h_t, c_t) = self.lstm(lstm_input, (h_prev, c_prev))
        # assert (z_t - h_t).sum() == 0
        return lstm_output, h_t, c_t
    
    def forward(self, x_t, h_prev, c_prev):
        # torch.Size([8, 3, 64, 64])
        #print(f'VideoVAE.forward(): x_t.shape = {x_t.shape}') 
        batch_size = x_t.size(0)

        # NOTE: no_grad here
        with torch.no_grad():
            x_enc = self.enc(x_t)
                
        # transformed the h_prev
        phi_h = self.mlp_lstm(h_prev)
        
        # posterior
        # mu_q.shape = torch.Size([8, 512])
        mu_q, logvar_q, z_dy, mu_dy, logvar_dy = self.approx_posterior(x_enc, phi_h)
        
        # prior p
        mu_p = torch.zeros_like(mu_q)
        logvar_p = torch.ones_like(logvar_q)
        #print(f'mu_p.shape = {mu_p.shape}')
        #print(f'logvar_p.shape = {logvar_p.shape}')
        
        # Sampling: the sample z_t
        z_t = z_dy
        
        # LSTM forward (update hidden state of LSTM)
        # Result in Temporally Consistent Sequences
        lstm_output, h_t, c_t = self.lstm_forward(mu_q, logvar_q, z_t, h_prev, c_prev)

        # Reconstruction
        recon_x_t = self.dec(z_t)
        
        return recon_x_t, z_t, lstm_output, [h_t, c_t], [mu_p, logvar_p], [mu_dy, logvar_dy]
    
    def reset(self, batch_size=4, reset='zeros'):
        """ reset lstm state.

        Returns:
            h_0, c_0 for LSTM.
        """
        use_cuda = next(self.parameters()).is_cuda
    
        h_0 = torch.zeros(1, batch_size, self.z_dim)
        c_0 = torch.zeros(1, batch_size, self.z_dim)

        # should set to random if we are synthesizing using only prior distribution.
        if reset == 'random':
            h_0 = torch.randn_like(h_0)
            c_0 = torch.randn_like(c_0)
        
        if use_cuda:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        return h_0, c_0