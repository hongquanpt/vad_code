import torch
import torch.nn as nn
import torch.distributions as tdist

class VideoVAELoss(nn.Module):
    def __init__(self, recon='L2'):
        super().__init__()
        self.recon = recon
        if self.recon.lower() == 'l1':
            self.recon_loss = nn.L1Loss(reduction='sum')
        elif self.recon.lower() == 'l2':
            self.recon_loss = nn.MSELoss(reduction='sum')
        else:
            raise NotImplementedError
            

    def forward(self, recon_x_t, x_t, phi_p, phi_q):
        recon_loss = self.recon_loss(recon_x_t, x_t)
        
        mu_p, logvar_p = phi_p
        mu_q, logvar_q = phi_q
        std_p, std_q = torch.exp(0.5 * logvar_p), torch.exp(0.5 * logvar_q)

        p = tdist.Normal(mu_p, std_p)
        q = tdist.Normal(mu_q, std_q)

        KLD = torch.sum(tdist.kl.kl_divergence(q, p))

        return recon_loss + KLD, recon_loss, KLD