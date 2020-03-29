import torch
import torch.nn as nn

from topo2vec.common.other_scripts import str_to_int_list

__all__ = ['BasicAutoencoder']


class BasicAutoencoder(nn.Module):
    def __init__(self, hparams):
        super(BasicAutoencoder, self).__init__()
        self.radii = str_to_int_list(hparams.radii)

        self.encoder = nn.Sequential(
            nn.Conv2d(len(self.radii), 4, 15, stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 3, 17, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        latent = torch.flatten(encoded, 1)
        decoded = self.decoder(encoded)
        return decoded, latent
