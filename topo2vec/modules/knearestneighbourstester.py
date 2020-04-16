from typing import List

import torch
import torchvision
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch.utils.data import ConcatDataset

from topo2vec.background import VALIDATION_HALF, POINT_TO_SEARCH_SIMILAR
from topo2vec.common.other_scripts import get_dataset_as_tensor
from topo2vec.common.visualizations import get_grid_sample_images_at_indexes, \
    convert_multi_radius_tensor_to_printable
from topo2vec.constants import GROUP_TO_SEARCH_SIMILAR_LONGS_LARGE
from topo2vec.datasets.class_dataset import ClassDataset
from topo2vec.datasets.random_dataset import RandomDataset
from topo2vec.modules import Classifier

import numpy as np


class KNearestNeighboursTester:
    def __init__(self, random_set_size: int, radii: List[int],
                 feature_extractor: Classifier, k: int, method: str,
                 json_file_of_group: str):
        self.random_set_size = random_set_size
        self.radii = radii

        self.feature_extractor = feature_extractor
        self.feature_extractor.freeze()
        self.k = k
        self.method = method
        self.json_file_of_group = json_file_of_group

    def prepare_data(self):
        '''

        prepare the self.random_dataset, and self.typical_images_dataset Datasets

        '''
        random_dataset = RandomDataset(self.random_set_size,
                                       self.radii, VALIDATION_HALF)
        size_typical_images = 5
        if self.method == 'regular':
            typical_images_dataset = ClassDataset(POINT_TO_SEARCH_SIMILAR, 1, self.radii,
                                                  size_typical_images, VALIDATION_HALF,  'test_knn')
        elif self.method == 'group_from_file':
            typical_images_dataset = ClassDataset(GROUP_TO_SEARCH_SIMILAR_LONGS_LARGE, 1, self.radii,
                                                  size_typical_images, VALIDATION_HALF,  'test_knn_group')
        else:
            raise Exception(f'The knn method provided, {self.method} is not acceptable')

        self.random_dataset = random_dataset
        self.typical_images_dataset = typical_images_dataset

    def test(self):
        '''

        log, through the self.feature_extractor logger,
        the test of the k nearest neighbours of the self.typical_images_dataset dataset

        '''
        random_dataset = ConcatDataset([self.random_dataset, self.typical_images_dataset])

        random_images_as_tensor = get_dataset_as_tensor(random_dataset)[0]
        random_images_as_np = random_images_as_tensor.data.numpy()

        typical_images_as_tensor = get_dataset_as_tensor(self.typical_images_dataset)[0]
        typical_images_as_np = typical_images_as_tensor.data.numpy()

        # calc closest images in images space
        print('started nn init')
        nn_classifier_images = NearestNeighbors(n_neighbors=self.k, metric='euclidean')
        nn_classifier_images.fit(random_images_as_np.reshape(
            random_images_as_np.shape[0], -1))
        distances_images, indices_images = nn_classifier_images.kneighbors(
            typical_images_as_np.reshape(
                typical_images_as_np.shape[0], -1))

        # images in latent space
        random_images_latent_as_np = self._get_dataset_latent_space_as_np(random_images_as_tensor)
        typical_images_latent_as_np = self._get_dataset_latent_space_as_np(typical_images_as_tensor)

        typical_images_set_to_show = convert_multi_radius_tensor_to_printable(typical_images_as_tensor)

        for i in range(len(typical_images_latent_as_np)):
            grid = get_grid_sample_images_at_indexes(random_images_as_tensor,
                                                     torch.tensor(indices_images[i]),
                                                     number_to_log=self.k)
            self.feature_extractor.logger.experiment.add_image(f'closest_samples_images_{i}', grid, 0)

        # calc closest images in latent space
        if self.method == 'regular':
            nn_classifier_latent = NearestNeighbors(n_neighbors=self.k, metric='euclidean')
            nn_classifier_latent.fit(random_images_latent_as_np)
            distances_latent, indices_latent = nn_classifier_latent.kneighbors(typical_images_latent_as_np)

            # log the actual images to the correct place
            for i in range(len(typical_images_latent_as_np)):
                grid = get_grid_sample_images_at_indexes(random_images_as_tensor,
                                                         torch.tensor(indices_latent[i]),
                                                         number_to_log=self.k)
                self.feature_extractor.logger.experiment.add_image(f'closest_samples_latent_{i}', grid, 0)

                grid = torchvision.utils.make_grid(typical_images_set_to_show[i])
                self.feature_extractor.logger.experiment.add_image(f'knn_example images_{i}', grid, 0)

        if self.method == 'group_from_file':
            nn_classifier_latent = NearestNeighbors(n_neighbors=self.k, metric='euclidean')
            nn_classifier_latent.fit(random_images_latent_as_np)
            typical_images_mean_as_np = np.mean(typical_images_latent_as_np, axis=0, keepdims=True)

            distances_latent, indices_latent = nn_classifier_latent.kneighbors(typical_images_mean_as_np)

            # log the actual images to the correct place
            grid = get_grid_sample_images_at_indexes(random_images_as_tensor,
                                                     torch.tensor(indices_latent[0]),
                                                     number_to_log=self.k)

            self.feature_extractor.logger.experiment.add_image(f'closest_samples_latent_group_mean', grid, 0)

            grid = torchvision.utils.make_grid(typical_images_set_to_show)
            self.feature_extractor.logger.experiment.add_image(f'group_of_example_images', grid, 0)

    def _get_dataset_latent_space_as_np(self, images_as_tensor: Tensor) -> Tensor:
        '''

        Args:
            images_as_tensor: the images to put in the feature_extractor

        Returns:

        '''
        images_as_tensor = images_as_tensor.float()
        if self.feature_extractor.hparams.use_gpu:
            images_as_tensor = images_as_tensor.cuda()
        _, images_latent_as_tensor = self.feature_extractor.forward(images_as_tensor)
        images_latent_as_np = images_latent_as_tensor.data
        if self.feature_extractor.hparams.use_gpu:
            images_latent_as_np = images_latent_as_np.cpu()
        images_latent_as_np = images_latent_as_np.numpy()
        return images_latent_as_np
