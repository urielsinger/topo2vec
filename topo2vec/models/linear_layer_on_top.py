import torch
import torch.nn as nn

__all__ = ['LinearLayerOnTop']


class LinearLayerOnTop(nn.Module):
    def __init__(self, hparams):
        '''

        Args:
            hparams: should contain:
                autoencoder: The Autoencoder class to forward on
                retrain: True if want to reretrain the moder, and Flase if want to train only on top
        '''
        super(LinearLayerOnTop, self).__init__()
        self.autoencoder = hparams.autoencoder
        # I assume that the model of the autoencoder is implementing forward that out, latent
        self.num_classes = hparams.num_classes
        self.retrain = hparams.retrain
        self.linear_layer = nn.Sequential(
            nn.Linear(hparams.latent_space_size, self.num_classes),
        )

    def forward(self, x):
        if self.retrain:
            _, latent = self.autoencoder.forward(x)
        else:
            with torch.no_grad():
                _, latent = self.autoencoder.forward(x)

        x = self.linear_layer(latent)
        return x, latent
