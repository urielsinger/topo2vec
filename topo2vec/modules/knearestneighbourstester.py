from typing import List

import torch
import torchvision
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch.utils.data import ConcatDataset

from topo2vec.background import VALIDATION_HALF, POINT_TO_SEARCH_SIMILAR
from topo2vec.common.visualizations import get_grid_sample_images_at_indexes, get_dataset_as_tensor, \
    convert_multi_radius_tensor_to_printable
from topo2vec.datasets.class_dataset import ClassDataset
from topo2vec.datasets.random_dataset import RandomDataset
from topo2vec.modules import Classifier


class KNearestNeighboursTester:
    def __init__(self, random_set_size: int, radii: List[int],
                 feature_extractor: Classifier, k: int):
        self.random_set_size = random_set_size
        self.radii = radii

        self.feature_extractor = feature_extractor
        self.feature_extractor.freeze()
        self.k = k

    def prepare_data(self):
        '''

        prepare the self.random_dataset, and self.typical_images_dataset Datasets

        '''
        random_dataset = RandomDataset(self.random_set_size,
                                       self.radii, VALIDATION_HALF)
        size_typical_images = 5
        typical_images_dataset = ClassDataset(POINT_TO_SEARCH_SIMILAR, 1, self.radii,
                                              size_typical_images, VALIDATION_HALF)
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
        nn_classifier_images = NearestNeighbors(n_neighbors=self.k, metric='euclidean')
        nn_classifier_images.fit(random_images_as_np.reshape(
            random_images_as_np.shape[0], -1))
        distances_images, indices_images = nn_classifier_images.kneighbors(
            typical_images_as_np.reshape(
                typical_images_as_np.shape[0], -1))

        # images in latent space
        random_images_latent_as_np = self._get_dataset_latent_space_as_np(random_images_as_tensor)
        typical_images_latent_as_np = self._get_dataset_latent_space_as_np(typical_images_as_tensor)

        # calc closest images in latent space
        nn_classifier_latent = NearestNeighbors(n_neighbors=self.k, metric='euclidean')
        nn_classifier_latent.fit(random_images_latent_as_np)
        distances_latent, indices_latent = nn_classifier_latent.kneighbors(typical_images_latent_as_np)

        typical_images_set_to_show = convert_multi_radius_tensor_to_printable(typical_images_as_tensor)

        # log the actual images to the correct place
        for i in range(len(typical_images_latent_as_np)):
            grid = get_grid_sample_images_at_indexes(random_images_as_tensor,
                                                     torch.tensor(indices_latent[i]),
                                                     number_to_log=self.k)
            self.feature_extractor.logger.experiment.add_image(f'closest_samples_latent_{i}', grid, 0)

            grid = get_grid_sample_images_at_indexes(typical_images_as_tensor,
                                                     torch.tensor(indices_images[i]),
                                                     number_to_log=self.k)
            self.feature_extractor.logger.experiment.add_image(f'closest_samples_images_{i}', grid, 0)

            grid = torchvision.utils.make_grid(typical_images_set_to_show[i])
            self.feature_extractor.logger.experiment.add_image(f'example images_{i}', grid, 0)

    def _get_dataset_latent_space_as_np(self, images_as_tensor: Tensor) -> Tensor:
        '''

        Args:
            images_as_tensor: the images to put in the feature_extractor

        Returns:

        '''
        _, images_latent_as_tensor = self.feature_extractor.forward(images_as_tensor.float())
        images_latent_as_np = images_latent_as_tensor.data.numpy()
        return images_latent_as_np
