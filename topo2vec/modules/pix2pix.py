import argparse
import logging
from argparse import Namespace
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

import topo2vec.models as models
from topo2vec.modules import Superresolution


class pix2pix(Superresolution):

    def __init__(self, hparams: Namespace):
        """
        a simple classifier to train on MultiRadiusDataset dataset
        using hparams, containing:
        arch - the architecture of the classifier
        and all other params defined in the "multi_class_experiment" script.
        """
        super(pix2pix, self).__init__(hparams)
        self.discriminator = models.__dict__[hparams.discriminator](hparams)

        self.generated_imgs = None

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch: Tensor, batch_idx: int, optimizer_idx) -> Dict:
        x, y = batch
        x_out = x[:, [self.hparams.index_out], 3:-2, 2:-3]
        x_in = x[:, [self.hparams.index_in], self.start_index:self.end_index, self.start_index:self.end_index]
        x_in = F.interpolate(x_in, size=x_out.shape[-1])
        # train generator
        if optimizer_idx == 0:
            # pix2pix images
            self.generated_imgs, _ = self(x)

            # log sampled images
            # sample_imgs = self.generated_imgs[:6]
            # grid = torchvision.utils.make_grid(sample_imgs)
            # self.logger.experiment.add_image('generated_images', grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(x.size(0), 1)
            if self.hparams.use_gpu:
                valid = valid.cuda(x.device.index)

            # adversarial loss is binary cross-entropy
            fake_pair = torch.cat((x_in, self.generated_imgs), 1)
            g_loss = self.adversarial_loss(self.discriminator(fake_pair), valid)
            g_loss += 100.0 * self.loss_fn(self.generated_imgs.float(), x_out.float())

            tqdm_dict = {'g_loss': g_loss}
            return {
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            }

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(x.size(0), 1)
            if self.hparams.use_gpu:
                valid = valid.cuda(x.device.index)

            real_pair = torch.cat((x_in, x_out), 1)
            real_loss = self.adversarial_loss(self.discriminator(real_pair), valid)

            # how well can it label as fake?
            fake = torch.zeros(x.size(0), 1)
            if self.hparams.use_gpu:
                fake = fake.cuda(x.device.index)

            fake_pair = torch.cat((x_in, self.generated_imgs.detach()), 1)
            fake_loss = self.adversarial_loss(self.discriminator(fake_pair), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2

            tqdm_dict = {'d_loss': d_loss}
            return {
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            }

    def _evaluation_step(self, batch: Tensor, name: str) -> Dict:
        '''
        the autoencoder does much less "evaluation_experiments step" operations because it is not about classifying.

        Args:
            batch:
            name: validation / test

        Returns:

        '''
        x, y = batch
        x_out = x[:, [self.hparams.index_out], 3:-2, 2:-3]
        x_in = x[:, [self.hparams.index_in], self.start_index:self.end_index, self.start_index:self.end_index]
        x_in = F.interpolate(x_in, size=x_out.shape[-1])

        x_t = x.cpu().numpy().copy()
        x_t[:, [self.hparams.index_in, self.hparams.index_out]] = x_t[:, [self.hparams.index_out, self.hparams.index_in]]
        x_t = torch.from_numpy(x_t)
        if self.hparams.use_gpu:
            x_t = x_t.cuda(x.device.index)
        ############## generator ##############

        # pix2pix images
        self.generated_imgs, _ = self(x_t)

        # log sampled images
        # sample_imgs = self.generated_imgs[:6]
        # grid = torchvision.utils.make_grid(sample_imgs)
        # self.logger.experiment.add_image('generated_images', grid, 0)

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(x.size(0), 1)
        if self.hparams.use_gpu:
            valid = valid.cuda(x.device.index)

        # adversarial loss is binary cross-entropy
        fake_pair = torch.cat((x_in, self.generated_imgs), 1)
        g_loss = self.adversarial_loss(self.discriminator(fake_pair), valid)
        g_loss += 100.0 * self.loss_fn(self.generated_imgs.float(), x_out.float())
        ############## discriminator ##############
        # Measure discriminator's ability to classify real from generated samples

        # how well can it label as real?
        valid = torch.ones(x.size(0), 1)
        if self.hparams.use_gpu:
            valid = valid.cuda(x.device.index)

        real_pair = torch.cat((x_in, x_out), 1)
        real_loss = self.adversarial_loss(self.discriminator(real_pair), valid)

        # how well can it label as fake?
        fake = torch.zeros(x.size(0), 1)
        if self.hparams.use_gpu:
            fake = fake.cuda(x.device.index)

        fake_pair = torch.cat((x_in, self.generated_imgs.detach()), 1)
        fake_loss = self.adversarial_loss(self.discriminator(fake_pair), fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2

        return {f'{name}_d_loss': d_loss, f'{name}_g_loss': g_loss}

    def _evaluation_epoch_end(self, outputs: list, name: str) -> Dict:
        '''
        the autoencoder does much less "evaluation_experiments epoch end" operations because it is not about classifying.
        Args:
            outputs:
            name: validation / test

        Returns:

        '''
        self.plot_before_after(name)
        self.plot_before_after('train')

        avg_g_loss = torch.stack([x[name + '_g_loss'] for x in outputs]).mean()
        avg_d_loss = torch.stack([x[name + '_d_loss'] for x in outputs]).mean()
        tensorboard_logs = {name + '_g_loss': avg_g_loss, name + '_d_loss': avg_d_loss}
        return {'avg_' + name + '_g_loss': avg_g_loss, 'avg_' + name + '_d_loss': avg_d_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        b1 = 0.5  # self.hparams.b1
        b2 = 0.999  # self.hparams.b2

        opt_g = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []
