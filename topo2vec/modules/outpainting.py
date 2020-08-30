from typing import Dict

import torchvision
from torch import Tensor

from topo2vec.modules import Autoencoder
from common.pytorch.pytorch_lightning_utilities import get_random_part_of_dataset
from common.pytorch.visualizations import convert_multi_radius_tensor_to_printable


class Outpainting(Autoencoder):
    def __init__(self, hparams):
        """
        Outpainting to train on MultiRadiusDataset dataset
        using hparams, containing:
        arch - the architecture of the autoencoder
        and all other params defined in the "autoencoder_experiment" script.
        """
        self.hparams = hparams
        super(Outpainting, self).__init__(hparams)

    def training_step(self, batch: Tensor, batch_idx: int) -> Dict:
        x, y = batch
        decoded, latent = self.forward(x[:, :len(self.radii) - 1].float())
        loss = self.loss_fn(decoded.float(), x[:, len(self.radii) - 1:].float())
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
        decoded, latent = self.forward(x[:, :len(self.radii) - 1].float())
        loss = self.loss_fn(decoded.float(), x[:, len(self.radii) - 1:].float())
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

        random_images_after_autoencoder, _ = self.model(random_images_as_tensor[:, :len(self.radii) - 1])
        grid_before = torchvision.utils.make_grid(
            convert_multi_radius_tensor_to_printable(random_images_as_tensor[:, len(self.radii) - 1:]))
        grid_after = torchvision.utils.make_grid(
            convert_multi_radius_tensor_to_printable(random_images_after_autoencoder))
        self.logger.experiment.add_image(f'{dataset_name}_original', grid_before, 0)
        self.logger.experiment.add_image(f'{dataset_name}_outpainting', grid_after, 0)
