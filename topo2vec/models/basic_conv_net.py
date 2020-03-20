import torch
import torch.nn as nn

__all__ = ['SimpleConvNet', 'simpleconvnet']


class SimpleConvNet(nn.Module):
    def __init__(self, num_classes=1,  radii=[10]):
        super(SimpleConvNet, self).__init__()
        self.radii = radii
        self.radius = min(radii)
        self.w = 2 * self.radius + 1
        self.h = 2 * self.radius + 1
        self.patch_size = self.w * self.h

        self.features = nn.Sequential(
            nn.Conv2d(len(radii), 10, kernel_size=5),
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

        self.classifier = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(20 * size_after_relu_2**2, 50),
            nn.SELU(inplace=True),
            nn.Linear(50, num_classes),
        )

        self.activation_function = nn.Softmax()


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.activation_function(x)
        return x


def simpleconvnet(pretrained=False, progress=True, **kwargs):
    r"""Simple convolutional network

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = SimpleConvNet(**kwargs)
    if pretrained:
        # state_dict = load_state_dict_from_url(model_urls['alexnet'],
        #                                      progress=progress)
        state_dict = None
        model.load_state_dict(state_dict)
    return model
