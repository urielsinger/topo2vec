import torch
import torch.nn as nn

from common.list_conversions_utils import str_to_int_list

__all__ = ['BasicAutoencoder']


class BasicAutoencoder(nn.Module):
    def __init__(self, hparams):
        super(BasicAutoencoder, self).__init__()
        self.radii = str_to_int_list(hparams.radii)
        self.is_outpaint = hparams.pytorch_module == 'Outpainting'
        in_dim = 1 if self.is_outpaint else len(self.radii)
        out_dim = 1 if self.is_outpaint else len(self.radii)
        self.index_in = hparams.index_in
        self.variational = False

        self.encoder = nn.Sequential(
            nn.Conv2d(in_dim, 4, 3, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(4, 8, 5, stride=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 5, stride=1),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 5, stride=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 5, stride=1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, stride=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 5, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 16, 5, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 4, 5, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(4, out_dim, 3, stride=1, padding=2),
            nn.Sigmoid()
        )
        self.fc_miu = nn.Linear(32, 32)
        self.fc_log_std = nn.Linear(32, 32)

    def forward(self, x, return_variational=False):
        if self.is_outpaint:
            x = x[:, [self.index_in]]

        latent = torch.flatten(self.encoder(x), 1)
        if self.variational:
            miu = self.fc_miu(latent)
            log_std = self.fc_log_std(latent)
            latent = torch.rand_like(log_std) * torch.exp(0.5*log_std) + miu

        decoded = self.decoder(latent.unsqueeze(2).unsqueeze(3))

        if self.variational and return_variational:
            return decoded, (latent, miu, log_std)
        else:
            return decoded, latent
