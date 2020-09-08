""" Full assembly of the parts to form the complete network """
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.list_conversions_utils import str_to_int_list


class UNet(nn.Module):
    def __init__(self, hparams, in_channels=1, out_channels=1, bilinear=True):
        self.radii = str_to_int_list(hparams.radii)
        self.index_in = hparams.index_in
        self.index_out = hparams.index_out
        resize = int((int((2 * self.radii[hparams.index_out] + 1) / (2 * self.radii[hparams.index_in] + 1) * (
                2 * self.radii[0] + 1)) - 1) / 2)
        self.start_index = self.radii[0] - resize
        self.end_index = self.radii[0] + resize + 1

        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.upsample = hparams.upsample

        self.inc = DoubleConv(in_channels, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)
        self.down4 = Down(64, 128)
        self.up1 = Up(128, 64, bilinear)
        self.up2 = Up(64, 32, bilinear)
        self.up3 = Up(32, 16, bilinear)
        self.up4 = Up(16, 8, bilinear)
        if self.upsample:
            self.up5 = Up(8, 4, bilinear)
            self.up6 = Up(4, 2, bilinear)
            self.outc = OutConv(2, out_channels)
        else:
            self.outc = OutConv(8, out_channels)

    def forward(self, x):
        if self.training:
            x = x[:, [self.index_in], self.start_index:self.end_index, self.start_index:self.end_index]
        else:
            x = x[:, [self.index_out], self.start_index:self.end_index, self.start_index:self.end_index]

        h = self.inc(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.down4(h)
        latent = torch.flatten(h, 1)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.up4(h)
        if self.upsample:
            h = self.up5(h)
            h = self.up6(h)
        out = self.outc(h)

        return out, latent


""" Parts of the U-Net model """


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class CNR2d(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, norm='bnorm', relu=0.0, drop=[], bias=[]):
        super().__init__()

        if bias == []:
            if norm == 'bnorm':
                bias = False
            else:
                bias = True

        layers = []
        layers += [Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

        if norm != []:
            layers += [Norm2d(nch_out, norm)]

        if relu != []:
            layers += [ReLU(relu)]

        if drop != []:
            layers += [nn.Dropout2d(drop)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)


class Conv2d(nn.Module):
    def __init__(self, nch_in, nch_out, kernel_size=4, stride=1, padding=1, bias=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)


class Linear(nn.Module):
    def __init__(self, nch_in, nch_out):
        super(Linear, self).__init__()
        self.linear = nn.Linear(nch_in, nch_out)

    def forward(self, x):
        return self.linear(x)


class Norm2d(nn.Module):
    def __init__(self, nch, norm_mode):
        super(Norm2d, self).__init__()
        if norm_mode == 'bnorm':
            self.norm = nn.BatchNorm2d(nch)
        elif norm_mode == 'inorm':
            self.norm = nn.InstanceNorm2d(nch)

    def forward(self, x):
        return self.norm(x)


class ReLU(nn.Module):
    def __init__(self, relu):
        super(ReLU, self).__init__()
        if relu > 0:
            self.relu = nn.LeakyReLU(relu, True)
        elif relu == 0:
            self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(x)


class Discriminator(nn.Module):
    def __init__(self, hparams):
        super(Discriminator, self).__init__()

        self.nch_in = 2
        self.nch_out = 1
        self.nch_ker = 8
        self.norm = 'bnorm'

        self.dsc1 = CNR2d(1 * self.nch_in, 1 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm,
                          relu=0.2)
        self.dsc2 = CNR2d(1 * self.nch_ker, 2 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm,
                          relu=0.2)
        self.dsc3 = CNR2d(2 * self.nch_ker, 4 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm,
                          relu=0.2)
        self.dsc4 = CNR2d(4 * self.nch_ker, 8 * self.nch_ker, kernel_size=4, stride=2, padding=1, norm=self.norm,
                          relu=0.2)
        self.dsc5 = CNR2d(8 * self.nch_ker, 16 * self.nch_ker, kernel_size=4, stride=1, padding=0, norm=[], relu=[],
                          bias=False)

        self.fc1 = nn.Linear(16 * self.nch_ker, 8 * self.nch_ker)
        self.fc2 = nn.Linear(8 * self.nch_ker, 4 * self.nch_ker)
        self.fc3 = nn.Linear(4 * self.nch_ker, self.nch_out)
        self.sig = nn.Sigmoid()

    def forward(self, img):
        x = self.dsc1(img)
        x = self.dsc2(x)
        x = self.dsc3(x)
        x = self.dsc4(x)
        x = self.dsc5(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        validity = self.sig(x)

        return validity