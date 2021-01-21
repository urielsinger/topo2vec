import torch
import torch.nn as nn

__all__ = ['BasicConvNetLatentDTM']


class BasicConvNetLatentDTM(nn.Module):
    def __init__(self, hparams):
        super(BasicConvNetLatentDTM, self).__init__()
        self.radii = [8]  # str_to_int_list(hparams.radii)
        self.radius = min(self.radii)
        self.w = 2 * self.radius + 1
        self.h = 2 * self.radius + 1
        self.patch_size = self.w * self.h
        # self.num_classes = hparams.num_classes
        self.latent_space_size = 20  # hparams.latent_space_size

        self.features = nn.Sequential(
            nn.Conv2d(len(self.radii), 10, kernel_size=5),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.SELU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        size_after_cnn_1 = self.w - 5 + 1
        size_after_relu_1 = int((size_after_cnn_1 + - 1 * (2 - 1) - 1) / 2 + 1)
        size_after_cnn_2 = size_after_relu_1 - 5 + 1
        size_after_relu_2 = int((size_after_cnn_2 + - 1 * (2 - 1) - 1) / 2 + 1)

        self.middle_seq = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(20 * size_after_relu_2 ** 2, self.latent_space_size)
        )

        # self.classifier = nn.Sequential(
        #     nn.SELU(inplace=True),
        #     nn.Linear(self.latent_space_size, self.num_classes),
        # )

    def forward(self, x):
        x_v1 = self.features(x)
        latent = torch.flatten(x_v1, 1)
        # breakpoint()
        latent = self.middle_seq(latent)
        # x = self.classifier(latent)
        # return x, latent
        return latent

from torchvision import models
import torch.nn.init as init
import torch.nn.functional as F

def fc_init_weights(m):
    # https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch/49433937#49433937
    if type(m) == nn.Linear:
        init.kaiming_normal_(m.weight.data)
class ResNet18(nn.Module):
    def __init__(self, dim = 128):
        super().__init__()

        resnet = models.resnet18(pretrained=False)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # converted for DTM

        self.encoder = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.FC = nn.Linear(512, dim)
        self.FC.apply(fc_init_weights)

    def forward(self, x):

        x = self.encoder(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)

        x = self.FC(x)

        e = F.normalize(x)

        return e