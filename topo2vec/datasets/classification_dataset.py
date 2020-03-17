import json
import os
from typing import List

import fiona
from shapely.geometry import Point
from torch import tensor
from torch.utils.data import Dataset

from topo2vec.datasets.multi_radius_dataset import MultiRadiusDataset

BOTTOM_PEAKS = None

from topo2vec import  visualizer

import numpy as np

class ClassificationDataset(MultiRadiusDataset):
    def __init__(self, radii: List[int] =[10], first_class_path: str = BOTTOM_PEAKS, first_class_label: str ='',
                 outer_polygon = None):
        super().__init__(radii, outer_polygon)
        self.features = []
        self.points_locations = []
        self.labels = []
        self.add_class_from_file(first_class_path, first_class_label)


    def __getitem__(self, index):
        #return (self.actual_patches[index], f'{self.labels[index]}'
        #                                    f'{str(self.points_used[index].x)}, {str(self.points_used[index].y)}')
        return (self.actual_patches[index], tensor([1.]))
    #TODO: change so the get item will make all calculations inside, altohugh I worked a lot for it.

    def __len__(self):
        return len(self.actual_patches)


    def add_class_from_file(self, file_path:str, label:str):
        points_list = self.load_points_list_from_file(file_path)
        self.add_points_as_patches_to_actual_patches(points_list)
        self.labels += [label]*len(self.actual_patches)

    def load_points_list_from_file(self, file_path:str) -> List[Point]:
        filename, file_extension = os.path.splitext(file_path)
        print(file_extension)
        if file_extension == '.shp':
            collection = fiona.open(file_path, encoding='ISO8859-1')
            new_features = list(collection)

        elif file_extension == '.geojson':
            with open(file_path, encoding='utf-8') as bottom_peaks_file:
                data = json.load(bottom_peaks_file)
            new_features = data['features']

        points_list = []
        for index in range(len(new_features)):
            coord_as_points_list = self._get_coord_as_points_list(index, new_features)
            points_list += coord_as_points_list

        return points_list

    def _get_coord_as_points_list(self, index, new_features):
        curr_idx_coords = new_features[index]['geometry']['coordinates']
        if len(curr_idx_coords)!=0:
            if type(curr_idx_coords[0]) == float:
                return [Point(curr_idx_coords[0], curr_idx_coords[1])]
            elif type(curr_idx_coords[0]) == list:
                return [Point(curr_idx_coord[0], curr_idx_coord[1]) for curr_idx_coord in curr_idx_coords]
        return []





    def __str__(self):
        if self.features is not None:
            print('the features that were loaded:')
            for feature in self.features:
                print(feature['geometry']['type'])
                print(feature['geometry']['coordinates'])


