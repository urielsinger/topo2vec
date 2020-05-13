from typing import List

import torch
import torchvision
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch.utils.data import ConcatDataset

from topo2vec.background import VALIDATION_HALF, POINT_TO_SEARCH_SIMILAR
from common.pytorch.pytorch_lightning_utilities import get_dataset_as_tensor, get_dataset_latent_space_as_np
from common.pytorch.visualizations import convert_multi_radius_tensor_to_printable, get_grid_sample_images, \
    plot_list_to_tensorboard
from topo2vec.constants import GROUP_TO_SEARCH_SIMILAR_LONGS_LARGE
from topo2vec.datasets.class_dataset import ClassDataset
from topo2vec.datasets.random_dataset import RandomDataset
from topo2vec.modules import Classifier

import numpy as np


class KNearestNeighboursTester:
    def __init__(self, radii: List[int], feature_extractor: Classifier, k: int, method: str,
                 random_set_size: int, json_file_of_group: str = GROUP_TO_SEARCH_SIMILAR_LONGS_LARGE):

        self.random_set_size = random_set_size
        self.radii = radii

        self.feature_extractor = feature_extractor
        self.feature_extractor.freeze()
        self.k = k
        self.method = method
        self.json_file_of_group = json_file_of_group

    def prepare_data(self):
        '''

        init the:
        self.random_dataset - dataset to search in
        self.typical_images_dataset - Dataset
        self.random_images_as_tensor - the dataset as tensor

        according to the self.method method.

        '''
        random_dataset = RandomDataset(self.random_set_size,
                                       self.radii, VALIDATION_HALF)
        size_typical_images = 5
        if self.method == 'regular':
            typical_images_dataset = ClassDataset(POINT_TO_SEARCH_SIMILAR, 1, self.radii,
                                                  size_typical_images, VALIDATION_HALF, 'test_knn')
        elif self.method == 'group_from_file':
            typical_images_dataset = ClassDataset(self.json_file_of_group, 1, self.radii,
                                                  size_typical_images, VALIDATION_HALF, 'test_knn_group')
        else:
            raise Exception(f'The knn method provided, {self.method} is not acceptable')

        self.random_dataset = random_dataset
        self.typical_images_dataset = typical_images_dataset
        self.random_dataset = ConcatDataset([self.random_dataset, self.typical_images_dataset])
        self.random_images_as_tensor = get_dataset_as_tensor(self.random_dataset)[0]
        self.random_points_list = None

    def set_pre_defined_datasets(self, random_dataset, typical_images_dataset, random_points_list=None):
        '''

        prepare the self.random_dataset, and self.typical_images_dataset Datasets

        '''
        self.random_dataset = random_dataset
        self.typical_images_dataset = typical_images_dataset
        self.random_images_as_tensor = get_dataset_as_tensor(self.random_dataset)[0]
        self.random_points_list = random_points_list
        assert len(self.random_points_list) == len(self.random_images_as_tensor)

    def test(self, number_to_log=5):
        '''

        log, through the self.feature_extractor logger,
        the test of the k nearest neighbours of the self.typical_images_dataset dataset

        '''
        random_images_as_np = self.random_images_as_tensor.data.numpy()

        typical_images_as_tensor = get_dataset_as_tensor(self.typical_images_dataset)[0]
        typical_images_as_np = typical_images_as_tensor.data.numpy()

        # images in latent space
        random_images_latent_as_np = get_dataset_latent_space_as_np(self.feature_extractor, self.random_images_as_tensor)
        typical_images_latent_as_np = get_dataset_latent_space_as_np(self.feature_extractor, typical_images_as_tensor)

        typical_images_set_to_show = convert_multi_radius_tensor_to_printable(typical_images_as_tensor)

        closest_images_list_image_space, _ = self.knn_algo_image_space(random_images_as_np.reshape(
            random_images_as_np.shape[0], -1),
            typical_images_as_np.reshape(
                typical_images_as_np.shape[0], -1))

        closest_images_list_latent_space, closest_points_list_latent_space = self.knn_algo_image_space(
            random_images_latent_as_np,
            typical_images_latent_as_np,
            take_mean=(self.method == 'group_from_file'))

        # the return is:
        # closest images (in images space) list
        # closest images (in latent space) list
        # the latent of images that where the input,
        # the "visual" way of the images that where the input
        # number of images that were asked to be logged
        # locations of the closest images (in latent space) list
        # locations of the closest images (in latent space) list
        return closest_images_list_image_space, closest_images_list_latent_space, typical_images_latent_as_np, \
               typical_images_set_to_show, number_to_log, closest_points_list_latent_space

    def knn_algo_image_space(self,
                             random_images_as_np, typical_images_as_np,
                             take_mean=False) -> List[torch.tensor]:
        '''
        Args:
            random_images_as_np: the images to search in
            typical_images_as_np: the images to  search for typical - one by one

        Returns: a list of the images that are closest to each point,
                and a list of the points locations that are closest to each point if provided in the constructor
                                                                                  (else - an empty list)
        '''
        # calc closest images in images space
        nn_classifier_images = NearestNeighbors(n_neighbors=self.k, metric='euclidean')
        nn_classifier_images.fit(random_images_as_np)

        if take_mean:
            typical_images_mean_as_np = np.mean(typical_images_as_np, axis=0, keepdims=True)
            typical_images_as_np = typical_images_mean_as_np

        distances_images, indices_images = nn_classifier_images.kneighbors(
            typical_images_as_np)

        images_list = []
        points_list = []
        for i in range(len(typical_images_as_np)):
            torch.tensor(indices_images[i])

            images = self.random_images_as_tensor[indices_images[i]]
            images_list.append(images)
            if self.random_points_list is not None:
                points_chosen_locations = [point for index, point in enumerate(self.random_points_list)
                                           if index in indices_images[i]]
                points_list.append(points_chosen_locations)

        return images_list, points_list

    def test_and_plot_via_feature_extractor_tensorboard(self):
        '''

        plot the knn examples patches to the tensorboard:
            'closest_samples_images' - the samples that are closest in th L2 norm on the original images
            in the regular case:
            'closest_samples_latent' - the samples that are closest in the L2 norm in the latent space
            'knn_example images_{i}' - the original images

            in the group case:
            'group_of_example_images' - the images that are in the group.
            'closest_samples_latent_group_mean' -  the samples that are closest in the L2 norm in the
                                                    latent space to the mean of the example images

        '''
        closest_images_list_image_space, closest_images_list_latent_space, typical_images_latent_as_np, \
        typical_images_set_to_show, number_to_log, _ = self.test()

        plot_list_to_tensorboard(self.feature_extractor, closest_images_list_image_space, number_to_log, 'closest_samples_images')
        if self.method == 'regular':
            plot_list_to_tensorboard(self.feature_extractor, closest_images_list_latent_space, number_to_log, 'closest_samples_latent')

            for i in range(len(typical_images_latent_as_np)):
                grid = torchvision.utils.make_grid(typical_images_set_to_show[i])
                self.feature_extractor.logger.experiment.add_image(f'knn_example_images_{i}', grid, 0)

        if self.method == 'group_from_file':
            plot_list_to_tensorboard(self.feature_extractor, closest_images_list_latent_space, number_to_log,
                                          'closest_samples_latent_group_mean')

            grid = torchvision.utils.make_grid(typical_images_set_to_show)
            self.feature_extractor.logger.experiment.add_image(f'group_of_example_images', grid, 0)

