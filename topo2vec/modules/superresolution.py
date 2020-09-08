from typing import Dict

import torch
import torchvision
from torch import Tensor

from topo2vec.modules import Autoencoder
from common.pytorch.pytorch_lightning_utilities import get_random_part_of_dataset
from common.pytorch.visualizations import convert_multi_radius_tensor_to_printable


class Superresolution(Autoencoder):
    def __init__(self, hparams):
        """
        Superresolution to train on MultiRadiusDataset dataset
        using hparams, containing:
        arch - the architecture of the super-resolution
        and all other params defined in the "autoencoder_experiment" script.
        """
        self.hparams = hparams
        super(Superresolution, self).__init__(hparams)

        if hparams.index_in > hparams.index_out:
            resize = int(
                (int((2 * self.radii[hparams.index_out] + 1) / (2 * self.radii[hparams.index_in] + 1) * (
                            2 * self.radii[0] + 1)) - 1) / 2)
            self.start_index = self.radii[0] - resize
            self.end_index = self.radii[0] + resize + 1
        else:
            raise Exception('doesnt support out-painting, just super-resolution')

    def training_step(self, batch: Tensor, batch_idx: int) -> Dict:
        x, y = batch
        decoded, latent = self.forward(x.float())

        # mask = torch.ones_like(decoded, dtype=torch.bool)
        # mask[:, :, self.start_index:self.end_index, self.start_index:self.end_index] = False
        # decoded = torch.masked_select(decoded, mask)
        # x_masked = torch.masked_select(x[:, [self.hparams.index_out]], mask)
        x_masked = x[:, [self.hparams.index_out], 3:-2, 2:-3]

        loss = self.loss_fn(decoded.float(), x_masked.float())
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def _evaluation_step(self, batch: Tensor, name: str) -> Dict:
        '''
        the autoencoder does much less "evaluation step" operations because it is not about classifying.

        Args:
            batch:
            name: validation / test

        Returns:

        '''
        x, y = batch
        decoded, latent = self.forward(x.float())

        # mask = torch.ones_like(decoded, dtype=torch.bool)
        # mask[:, :, self.start_index:self.end_index, self.start_index:self.end_index] = False
        # decoded = torch.masked_select(decoded, mask)
        # x_masked = torch.masked_select(x[:, [self.hparams.index_out]], mask)
        x_masked = x[:, [self.hparams.index_out], 3:-2, 2:-3]

        loss = self.loss_fn(decoded.float(), x_masked.float())
        return {name + '_loss': loss}

    def plot_before_after(self, dataset_name: str, number_to_plot: int = 5):
        '''
        adds to the ternsorboard the before and after autoencoder images.
        Args:
            dataset_name: the dataset to plot from
            number_to_plot: number of images we want to plot

        Returns:

        '''
        random_images_as_tensor, y = get_random_part_of_dataset(self.datasets[dataset_name], number_to_plot)
        random_images_as_tensor = random_images_as_tensor.float()
        if self.hparams.use_gpu:
            random_images_as_tensor = random_images_as_tensor.cuda()

        random_images_after_autoencoder, _ = self.forward(random_images_as_tensor)
        grid_gt = torchvision.utils.make_grid(
            convert_multi_radius_tensor_to_printable(random_images_as_tensor[:, [self.hparams.index_out], 3:-2, 2:-3]))
        grid_input = torchvision.utils.make_grid(
            convert_multi_radius_tensor_to_printable(
                random_images_as_tensor[:, [self.hparams.index_in], self.start_index:self.end_index,
                self.start_index:self.end_index]))
        grid_after = torchvision.utils.make_grid(
            convert_multi_radius_tensor_to_printable(random_images_after_autoencoder))
        self.logger.experiment.add_image(f'{dataset_name}_GT', grid_gt, 0)
        self.logger.experiment.add_image(f'{dataset_name}_input', grid_input, 0)
        self.logger.experiment.add_image(f'{dataset_name}_pred', grid_after, 0)
