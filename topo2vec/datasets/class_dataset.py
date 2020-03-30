import os
from pathlib import Path
from typing import List, Tuple

from torch import tensor, Tensor

from topo2vec.background import classes_data_handlers
from topo2vec.common.other_scripts import floats_list_to_points_list, cache_path_name_to_full_path, load_list_from_file, \
    save_list_to_file, points_list_to_floats_list, get_dataset_dir_base_path, full_path_name_to_full_path
from topo2vec.constants import CACHE_BASE_DIR, CLASSES_CACHE_SUB_DIR
from topo2vec.data_handlers.classes_data_file_handler import ClassesDataFileHadler
from topo2vec.datasets.multi_radius_dataset import MultiRadiusDataset

import numpy as np


class ClassDataset(MultiRadiusDataset):
    '''
    A dataset contains only one class
    '''

    def __init__(self, first_class_path: str, first_class_label: float,
                 radii: List[int], wanted_size: int, outer_polygon=None,
                 dataset_type_name:str=None):
        '''

        Args:
            first_class_path: The path to the data of the first class wanted in the dataset
            first_class_label: The label of the first class wanted in the dataset
            radii:
            outer_polygon:
        '''
        super().__init__(radii, outer_polygon)
        self.points_locations = []
        self.labels = []
        self.wanted_size = wanted_size
        self.dataset_type_name = dataset_type_name
        self.add_class_from_file(first_class_path, float(first_class_label))
        self.full_base_dir = None

    def __getitem__(self, index) -> Tuple[np.ndarray, tensor]:
        '''
        184 River points

        Args:
            index:

        Returns: a tuple of the data and the label

        '''
        return self.actual_patches[index], tensor([self.labels[index]])

    def add_class_from_file(self, file_path: str, label: float):
        '''
        updates self.actual_patches and self.labels
        Args:
            file_path: The path to the data of the class wanted to be added to the dataset
            label: The label of the class wanted in the dataset

        Returns: nothing

        '''
        self.full_base_dir = get_dataset_dir_base_path(CACHE_BASE_DIR, file_path, self.dataset_type_name)
        Path(self.full_base_dir).mkdir(parents=True, exist_ok=True)

        points_list = None
        if self.full_base_dir is not None:
            full_path_points_list = full_path_name_to_full_path(self.full_base_dir, 'points')
            points_list = load_list_from_file(full_path_points_list)

        if points_list is None:
            if file_path in classes_data_handlers.keys():
                class_data_handler = classes_data_handlers[file_path]
            else:
                class_data_handler = ClassesDataFileHadler(file_path)
            points_list = class_data_handler.\
                get_random_subset_in_polygon(self.wanted_size, self.outer_polygon)
            if self.full_base_dir is not None:
                save_list_to_file(full_path_points_list, points_list_to_floats_list(points_list))
        else:
            points_list = floats_list_to_points_list(points_list)

        self.add_points_as_patches_to_actual_patches(points_list, file_path)
        full_path_labels = full_path_name_to_full_path(self.full_base_dir, 'labels')
        labels = load_list_from_file(full_path_labels)
        if labels is None:
            self.labels += [label] * len(self.actual_patches)
            save_list_to_file(full_path_labels, self.labels)
        else:
            self.labels = labels




