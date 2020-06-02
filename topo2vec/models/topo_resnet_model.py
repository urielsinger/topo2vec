import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock, model_urls

from common.list_conversions_utils import str_to_int_list

__all__ = ['TopoResNet']


class TopoResNet(ResNet):
    def __init__(self, hparams):
        super(TopoResNet, self).__init__(BasicBlock, [2, 2, 2, 2])
        state_dict = load_state_dict_from_url(model_urls['resnet18'], progress=True)
        self.load_state_dict(state_dict)
        self.num_classes = hparams.num_classes
        self.latent_space_size = 512
        if not hparams.train_all_resnet:
            for param in self.parameters():
                param.requires_grad = False
        self.fc = nn.Sequential(
            nn.SELU(inplace=True),
            nn.Linear(512, self.num_classes),
        )
        self.radii = str_to_int_list(hparams.radii)
        self.radius = min(self.radii)


    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        latent = torch.flatten(x, 1)
        x = self.fc(latent)
        return x, latent

    def forward(self, x):
        x, latent = self._forward_impl(x)
        return x, latent
