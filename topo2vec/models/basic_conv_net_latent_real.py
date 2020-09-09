import torch
import torch.nn as nn

from common.list_conversions_utils import str_to_int_list

__all__ = ['BasicConvNetLatentReal']


class BasicConvNetLatentReal(nn.Module):
    def __init__(self, hparams):
        super(BasicConvNetLatentReal, self).__init__()
        self.radii = str_to_int_list(hparams.radii)
        self.radius = min(self.radii)
        self.w = 2 * self.radius + 1
        self.h = 2 * self.radius + 1
        self.patch_size = self.w * self.h
        self.num_classes = hparams.num_classes
        self.latent_space_size = hparams.latent_space_size

        self.features = nn.Sequential(
            nn.Conv2d(len(self.radii), 10, kernel_size=5),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(10, self.latent_space_size, kernel_size=5),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        size_after_cnn_1 = self.w - 5 + 1
        size_after_relu_1 = int((size_after_cnn_1 + - 1 * (2 - 1) - 1) / 2 + 1)
        size_after_cnn_2 = size_after_relu_1 - 5 + 1
        size_after_relu_2 = int((size_after_cnn_2 + - 1 * (2 - 1) - 1) / 2 + 1)

        self.middle_seq = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(self.latent_space_size * size_after_relu_2 ** 2, self.latent_space_size)
            # nn.Linear(20 * size_after_relu_2 ** 2, self.latent_space_size)
        )

        self.classifier = nn.Sequential(
            nn.SELU(inplace=True),
            nn.Linear(self.latent_space_size, self.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        latent = torch.flatten(x, 1)
        latent = self.middle_seq(latent)
        x = self.classifier(latent)
        return x, latent
