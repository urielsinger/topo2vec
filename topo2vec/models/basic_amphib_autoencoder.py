import torch
import torch.nn as nn

from topo2vec.common.other_scripts import str_to_int_list

__all__ = ['BasicAmphibAutoencoder']

class BasicAmphibAutoencoder(nn.Module):
    '''
    Amphib = linear+ convolutional autoencoder
    '''
    def __init__(self, hparams):
        super(BasicAmphibAutoencoder, self).__init__()
        self.radii = str_to_int_list(hparams.radii)

        self.radius = min(self.radii)
        self.h_w = 2 * self.radius + 1
        self.im_size = self.h_w * self.h_w * len(self.radii)

        self.encoder = nn.Sequential(
            nn.Conv2d(len(self.radii), 4, 3, stride=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
        )
        self.encoder_linear = nn.Sequential(
            nn.Linear(196, hparams.latent_space_size),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hparams.latent_space_size, self.im_size),
            nn.Sigmoid()
        )


    def forward(self, x):
        final_view = x.shape
        encoded = self.encoder(x)
        latent = torch.flatten(encoded, 1)
        latent = self.encoder_linear(latent)
        decoded = self.decoder(latent)
        decoded = decoded.view(final_view)
        return decoded, latent
