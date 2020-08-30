import torch
import torch.nn as nn

from common.list_conversions_utils import str_to_int_list

__all__ = ['BasicAutoencoder']


class BasicAutoencoder(nn.Module):
    def __init__(self, hparams):
        super(BasicAutoencoder, self).__init__()
        self.radii = str_to_int_list(hparams.radii)
        in_dim = len(self.radii)-1 if hparams.pytorch_module == 'Outpainting' else len(self.radii)
        out_dim = 1 if hparams.pytorch_module == 'Outpainting' else len(self.radii)

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
            nn.ReLU(True),
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

    def forward(self, x):
        encoded = self.encoder(x)
        latent = torch.flatten(encoded, 1)
        decoded = self.decoder(encoded)
        return decoded, latent
