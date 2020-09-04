import torch
import torch.nn as nn

from common.list_conversions_utils import str_to_int_list

__all__ = ['BasicLinearAutoencoder']


class BasicLinearAutoencoder(nn.Module):
    '''
    Basic Linear autoencoder
    '''

    def __init__(self, hparams):
        super(BasicLinearAutoencoder, self).__init__()
        self.radii = str_to_int_list(hparams.radii)
        self.radius = min(self.radii)
        self.h_w = 2 * self.radius + 1

        self.im_size = self.h_w * self.h_w * len(self.radii)
        self.encoder = nn.Sequential(
            nn.Linear(self.im_size, hparams.latent_space_size),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hparams.latent_space_size, self.im_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        final_view = x.shape
        encoded = self.encoder(x.view(-1, self.im_size))
        latent = encoded
        decoded = self.decoder(encoded)
        decoded = decoded.view(final_view)
        return decoded, latent
