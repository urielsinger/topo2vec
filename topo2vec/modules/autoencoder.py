from typing import Dict

import torchvision
from torch import Tensor

from topo2vec.background import TRAIN_HALF, VALIDATION_HALF, LOAD_CLASSES_LARGE, class_names
from topo2vec.common.visualizations import get_random_part_of_dataset, convert_multi_radius_tensor_to_printable
from topo2vec.constants import class_paths_test
from topo2vec.datasets.random_dataset import RandomDataset
from topo2vec.datasets.several_classes_datasets import SeveralClassesDataset
from topo2vec.modules.classifier import Classifier

import torch


class Autoencoder(Classifier):
    def __init__(self, hparams):
        """
        an autoencoder to train on MultiRadiusDataset dataset
        using hparams, containing:
        arch - the architecture of the autoencoder
        and all other params defined in the "autoencoder_experiment" script.
        """
        self.hparams = hparams
        super(Autoencoder, self).__init__(hparams)
        self.loss_fn = torch.nn.MSELoss()
        self.w_h = min(self.radii) * 2 + 1
        self.img_size = len(self.radii) * self.w_h ** 2
        self.train_portion = hparams.train_portion


    def prepare_data(self):
        '''

        prepare the datasets for the autoencoder performance

        '''
        size_train = int(self.train_portion * self.total_dataset_size)
        size_val = int((1-self.train_portion) * self.total_dataset_size)
        self.train_dataset = RandomDataset(size_train, self.radii, TRAIN_HALF)
        self.validation_dataset = RandomDataset(size_val, self.radii, VALIDATION_HALF)

        if LOAD_CLASSES_LARGE:
            size_test = 55
            self.test_dataset = SeveralClassesDataset(self.radii, VALIDATION_HALF, size_test, class_paths_test,
                                                      class_names, 'test')
        else:
            self.test_dataset = None

        self.datasets = {
            'validation': self.validation_dataset,
            'train': self.train_dataset,
            'test': self.test_dataset
        }

    def training_step(self, batch: Tensor, batch_idx: int) -> Dict:
        x, y = batch
        decoded, latent = self.forward(x.float())
        loss = self.loss_fn(decoded.float(), x.float())
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def evaluation_step(self, batch: Tensor, name: str) -> Dict:
        x, y = batch
        decoded, latent = self.forward(x.float())
        loss = self.loss_fn(decoded.float(), x.float())
        return {name + '_loss': loss}

    def plot_before_after(self, dataset_name: str, number_to_plot: int = 5):
        random_images_as_tensor, y = get_random_part_of_dataset(self.datasets[dataset_name], number_to_plot)
        random_images_after_autoencoder, _ = self.model(random_images_as_tensor.float())
        grid_before = torchvision.utils.make_grid(
            convert_multi_radius_tensor_to_printable(random_images_as_tensor))
        grid_after = torchvision.utils.make_grid(
            convert_multi_radius_tensor_to_printable(random_images_after_autoencoder))
        self.logger.experiment.add_image(f'{dataset_name}_before autoencoder', grid_before, 0)
        self.logger.experiment.add_image(f'{dataset_name}_after autoencoder', grid_after, 0)

    def evaluation_epoch_end(self, outputs: list, name: str) -> Dict:
        avg_loss = torch.stack([x[name + '_loss'] for x in outputs]).mean()
        tensorboard_logs = {name + '_loss': avg_loss}
        self.plot_before_after(name)
        self.plot_before_after('train')
        return {'avg_' + name + '_loss': avg_loss, 'log': tensorboard_logs}
