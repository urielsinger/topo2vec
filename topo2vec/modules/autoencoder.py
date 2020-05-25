from typing import Dict

import torchvision
from torch import Tensor

from topo2vec.background import TRAIN_HALF, VALIDATION_HALF, LOAD_CLASSES_LARGE, CLASS_PATHS_TEST, \
    CLASS_NAMES_TEST
from common.pytorch.pytorch_lightning_utilities import get_random_part_of_dataset
from common.pytorch.visualizations import convert_multi_radius_tensor_to_printable
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
        self.svm_validation_accuracy = 0
        self.svm_test_accuracy = 0



    def prepare_data(self):
        '''

        prepare the datasets for the autoencoder performance

        '''
        size_train = int(self.train_portion * self.total_dataset_size)
        size_val = int((1-self.train_portion) * self.total_dataset_size)
        self.train_dataset = RandomDataset(size_train, self.original_radiis, TRAIN_HALF, self.radii)
        self.validation_dataset = RandomDataset(size_val, self.original_radiis, VALIDATION_HALF, self.radii)

        if LOAD_CLASSES_LARGE:
            self.test_dataset = SeveralClassesDataset(self.original_radiis, VALIDATION_HALF, self.size_test, CLASS_PATHS_TEST,
                                                      CLASS_NAMES_TEST, 'num_classes_' + str(self.num_classes) + 'test', radii=self.radii)
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
        loss = self.loss_fn(decoded.float(), x.float())
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

        random_images_after_autoencoder, _ = self.model(random_images_as_tensor)
        grid_before = torchvision.utils.make_grid(
            convert_multi_radius_tensor_to_printable(random_images_as_tensor))
        grid_after = torchvision.utils.make_grid(
            convert_multi_radius_tensor_to_printable(random_images_after_autoencoder))
        self.logger.experiment.add_image(f'{dataset_name}_before autoencoder', grid_before, 0)
        self.logger.experiment.add_image(f'{dataset_name}_after autoencoder', grid_after, 0)

    def _evaluation_epoch_end(self, outputs: list, name: str) -> Dict:
        '''
        the autoencoder does much less "evaluation epoch end" operations because it is not about classifying.
        Args:
            outputs:
            name: validation / test

        Returns:

        '''
        avg_loss = torch.stack([x[name + '_loss'] for x in outputs]).mean()
        tensorboard_logs = {name + '_loss': avg_loss}
        self.plot_before_after(name)
        self.plot_before_after('train')
        return {'avg_' + name + '_loss': avg_loss, 'log': tensorboard_logs}

    def get_hyperparams_value_for_maximizing(self):
        '''

        Returns: the value we want to maximize when running an optuna hyper-params search for autoencoders

        '''
        return self.svm_validation_accuracy
