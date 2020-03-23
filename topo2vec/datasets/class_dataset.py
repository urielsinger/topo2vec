import json
import os
import random
from typing import List, Tuple

import fiona
from shapely.geometry import Point
from torch import tensor
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
        points_list = self.load_points_list_from_file(file_path)
        self.add_points_as_patches_to_actual_patches(points_list)
        self.labels += [label] * len(self.actual_patches)

    def load_points_list_from_file(self, file_path: str) -> List[Point]:
        '''
        load all the points that are inside a points list
        Args:
            file_path: The .shp or. geojson file of the class's data

        Returns: a Points list of all the points in the file
        (if the file contains lines - all the points in the line)

        '''
        filename, file_extension = os.path.splitext(file_path)
        print(file_extension)
        new_features = None
        if file_extension == '.shp':
            collection = fiona.open(file_path, encoding='ISO8859-1')
            new_features = list(collection)

        elif file_extension == '.geojson':
            with open(file_path, encoding='utf-8') as bottom_peaks_file:
                data = json.load(bottom_peaks_file)
            new_features = data['features']

        elif file_extension == '.json':
            with open(file_path, encoding='utf-8') as bottom_peaks_file:
                data = json.load(bottom_peaks_file)
            new_features = data['elements']

        else:
            raise Exception('No points in file!')

        points_list = []
        for index in range(len(new_features)):
            coord_as_points_list = self._get_coord_as_points_list(index, new_features, file_extension)
            points_list += coord_as_points_list
            if len(points_list) >= self.wanted_size:
                break


        if len(points_list) >= self.wanted_size:
            return points_list[:self.wanted_size]
        else:
            raise Exception('Not enough points in class in this region as needed!')

    def _get_coord_as_points_list(self, index: int, new_features: np.ndarray, file_extension:str) -> List[Point]:
        '''

        Args:
            index:
            new_features: The features ndarray of the coordinate, got from the image

        Returns: a list of all the points inside a row.

        '''
        if file_extension == '.shp' or file_extension == '.geojson':
            curr_idx_coords = new_features[index]['geometry']['coordinates']
            if len(curr_idx_coords) != 0:
                if type(curr_idx_coords[0]) == float:
                    return [Point(curr_idx_coords[0], curr_idx_coords[1])]
                elif type(curr_idx_coords[0]) == list:
                    curr_idx_coord = random.choice(curr_idx_coords)
                    # return [Point(curr_idx_coord[0], curr_idx_coord[1]) for curr_idx_coord in curr_idx_coords]
                    return [Point(curr_idx_coord[0], curr_idx_coord[1])]
        elif file_extension == '.json':
            return [Point(new_features[index]['lon'], new_features[index]['lat'])]
        return []
