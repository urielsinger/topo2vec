
from typing import List, Tuple

from torch import tensor

from topo2vec.background import classes_data_handlers
from topo2vec.data_handlers.classes_data_file_handler import ClassesDataFileHadler
from topo2vec.datasets.multi_radius_dataset import MultiRadiusDataset

import numpy as np


class ClassDataset(MultiRadiusDataset):
    '''
    A dataset contains only one class
    '''

    def __init__(self, first_class_path: str, first_class_label: float,
                 radii: List[int], wanted_size:int, outer_polygon=None):
        '''

        Args:
            first_class_path: The path to the data of the first class wanted in the dataset
            first_class_label: The label of the first class wanted in the dataset
            radii:
            outer_polygon:
        '''
        super().__init__(radii, outer_polygon)
        self.features = []
        self.points_locations = []
        self.labels = []
        self.wanted_size = wanted_size
        self.add_class_from_file(first_class_path, float(first_class_label))

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
        if file_path in classes_data_handlers.keys():
            class_data_handler = classes_data_handlers[file_path]
        else:
            class_data_handler = ClassesDataFileHadler(file_path)
        points_list = class_data_handler.\
            get_random_subset_in_polygon(self.wanted_size, self.outer_polygon)

        self.add_points_as_patches_to_actual_patches(points_list)
        self.labels += [label] * len(self.actual_patches)
