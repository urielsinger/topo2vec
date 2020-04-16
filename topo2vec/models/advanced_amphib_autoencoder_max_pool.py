import torch
import torch.nn as nn

from topo2vec.common.other_scripts import str_to_int_list

__all__ = ['AdvancedAmphibAutoencoderMaxPool']


class AdvancedAmphibAutoencoderMaxPool(nn.Module):
    '''
    Amphib = linear+ convolutional autoencoder
    '''

    def __init__(self, hparams):
        super(AdvancedAmphibAutoencoderMaxPool, self).__init__()
        self.radii = str_to_int_list(hparams.radii)

        self.radius = min(self.radii)
        self.h_w = 2 * self.radius + 1
        self.im_size = self.h_w * self.h_w * len(self.radii)

        self.encoder = nn.Sequential(
            nn.Conv2d(len(self.radii), 8, 3, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.ReLU(True),
            nn.MaxPool2d(2),

        )
        self.encoder_linear = nn.Sequential(
            nn.Linear(512, hparams.latent_space_size),
            nn.ReLU(True),
        )
        self.decoder_linear = nn.Sequential(
            nn.Linear(hparams.latent_space_size, 512),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, len(self.radii), 3, stride=1, padding=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        final_view = x.shape
        encoded = self.encoder(x)
        middle_view = encoded.shape
        latent = torch.flatten(encoded, 1)
        latent = self.encoder_linear(latent)
        decoded = self.decoder_linear(latent)
        decoded = decoded.view(middle_view)
        decoded = self.decoder(decoded)
        return decoded, latent
