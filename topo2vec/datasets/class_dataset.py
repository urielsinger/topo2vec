import os
from pathlib import Path
from typing import List, Tuple

from torch import tensor

from topo2vec.background import classes_data_handlers
from common.dataset_utils import full_path_name_of_dataset_data_to_full_path
from common.list_conversions_utils import points_list_to_floats_list, floats_list_to_points_list, load_list_from_file, \
    save_list_to_file
from topo2vec.constants import CACHE_BASE_DIR
from topo2vec.data_handlers.classes_data_file_handler import ClassesDataFileHadler
from topo2vec.datasets.multi_radius_dataset import MultiRadiiDataset

import numpy as np


class ClassDataset(MultiRadiiDataset):
    '''
    A dataset contains only one class
    '''

    def __init__(self, class_path: str, class_label: float,
                 original_radiis: List[int],
                 wanted_size: int, radii: List[int] = None, outer_polygon=None,
                 dataset_type_name: str = None, load_save=True, return_point=False, seed=None, only_higher_than=None):
        '''

        Args:
            class_path: The path to the data of the first class wanted in the dataset
            class_label: The label of the first class wanted in the dataset
            original_radiis:
            outer_polygon:
        '''
        super().__init__(original_radiis, radii, outer_polygon)
        self.points_locations = []
        self.labels = []
        self.wanted_size = wanted_size
        self.load_save = load_save
        self.seed = seed
        file_name, _ = os.path.splitext(class_path)
        type_area_name = file_name.split('/')[-1]
        self.full_base_dir = os.path.join(CACHE_BASE_DIR, 'datasets', type_area_name,
                                          f'{dataset_type_name}_{wanted_size}_label_{class_label}_{original_radiis}_{radii}_{seed}')
        if self.load_save:
            Path(self.full_base_dir).mkdir(parents=True, exist_ok=True)
        # self.full_base_dir = None  ## remove if wants cache
        self.return_point = return_point
        self.only_higher_than = only_higher_than
        self.add_class_from_file(class_path, float(class_label))


    def __getitem__(self, index) -> Tuple[np.ndarray, tensor]:
        '''
        184 River points

        Args:
            index:

        Returns: a tuple of the data and the label

        '''
        if self.return_point:
            return self.actual_patches[index], tensor([self.labels[index]]), \
                   np.array([self.points_locations[index].x, self.points_locations[index].y])
        else:
            return self.actual_patches[index], tensor([self.labels[index]])

    def add_class_from_file(self, file_path: str, label: float):
        '''
        updates self.actual_patches and self.labels
        Args:
            file_path: The path to the data of the class wanted to be added to the dataset
            label: The label of the class wanted in the dataset

        Returns: nothing

        '''
        points_list = None
        if self.full_base_dir is not None:
            full_path_points_list = full_path_name_of_dataset_data_to_full_path(self.full_base_dir, 'points')
            points_list = load_list_from_file(full_path_points_list)

        if points_list is None:
            if file_path not in classes_data_handlers.keys():
                class_data_handler = ClassesDataFileHadler(file_path)
                classes_data_handlers[file_path] = class_data_handler

            points_list = classes_data_handlers[file_path]. \
                get_random_subset_in_polygon(self.wanted_size, self.outer_polygon, seed=self.seed,
                                             only_higher_than=self.only_higher_than)
            if self.full_base_dir is not None:
                save_list_to_file(full_path_points_list, points_list_to_floats_list(points_list))
        else:
            points_list = floats_list_to_points_list(points_list)

        self.add_points_as_patches_to_actual_patches(points_list, file_path)

        labels = None
        if self.full_base_dir is not None:
            full_path_labels = full_path_name_of_dataset_data_to_full_path(self.full_base_dir, 'labels')
            labels = load_list_from_file(full_path_labels)
        if labels is None:
            self.labels += [label] * len(self.actual_patches)
            if self.full_base_dir is not None:
                save_list_to_file(full_path_labels, self.labels)
        else:
            self.labels = labels
