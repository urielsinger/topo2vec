import os
from typing import List

from shapely.geometry import Point, Polygon
from torch import Tensor
from torch.utils.data import Dataset

from topo2vec import visualizer
from topo2vec import mask_visualizer

import numpy as np

from topo2vec.common.geographic.geo_utils import check_if_point_in_polygon
from topo2vec.common.other_scripts import cache_path_name_to_full_path, load_list_from_file, save_list_to_file, \
    points_list_to_floats_list, floats_list_to_points_list, full_path_name_to_full_path
from topo2vec.constants import CACHE_BASE_DIR, CLASSES_CACHE_SUB_DIR




class MultiRadiusDataset(Dataset):
    '''
    A dataset that supports the making of a point to an 3-dim ndarray
    of the neighbourhood of the point in different radii.
    '''

    def __init__(self, radii: List[int], outer_polygon: Polygon = None):
        '''

        Args:
            radii: the radii of the neighbourhoods.
            outer_polygon: if None - ignore, otherwise - take only
            points that are inside it.
        '''
        self.radii = radii
        self.actual_patches = None  # the actual data of the dataset.
        self.mask_patches = None
        self.use_masks = False
        self.outer_polygon = outer_polygon
        self.points_locations = None

    def add_points_as_patches_to_actual_patches(self, points: List[Point], file_path: str = None):
        '''
        add the ndarrays that represent the points to the self.actual_points list
        that is the actual data of the dataset
        Args:
            points: points list
        '''
        actual_patches = None
        points_locations = None
        if self.full_base_dir is not None:
            full_path_actual_patches = full_path_name_to_full_path(self.full_base_dir, 'actual_patches')
            actual_patches = load_list_from_file(full_path_actual_patches)

            full_path_points_locations = full_path_name_to_full_path(self.full_base_dir, 'points_locations')
            points_locations = load_list_from_file(full_path_points_locations)

        if actual_patches is None or points_locations is None:
            if self.outer_polygon is not None:
                points = [point for point in points if
                          check_if_point_in_polygon(point, self.outer_polygon)]

            new_patches, points_locations_list = visualizer.get_points_as_np_array(points, self.radii)
            self.points_locations = points_locations_list
            if self.actual_patches is not None:
                all_patches = [self.actual_patches, new_patches]
                self.actual_patches = np.concatenate(all_patches)
            else:
                self.actual_patches = new_patches

            if self.full_base_dir is not None:
                save_list_to_file(full_path_actual_patches, self.actual_patches)
                save_list_to_file(full_path_points_locations, points_list_to_floats_list(self.points_locations))

        else:
            self.actual_patches = Tensor(actual_patches)
            self.points_locations = floats_list_to_points_list(points_locations)

    def add_points_as_patches_to_mask_patches(self, points: List[Point]):
        '''
        The same as add_points_as_patches_to_actual_patches,
        but adds the masks of the points to the self.mask_patches object,
        instead of to the self.actual_patches
        Args:
            points:

        Returns:

        '''
        if self.outer_polygon is not None:
            points = [point for point in points
                      if check_if_point_in_polygon(point, self.outer_polygon)]

        new_patches, _ = mask_visualizer.get_points_as_np_array(points, self.radii)

        if self.mask_patches is not None:
            all_mask_patches = [self.mask_patches, new_patches]
            self.mask_patches = np.concatenate(all_mask_patches)
        else:
            self.mask_patches = new_patches

    def __len__(self):
        return len(self.actual_patches)
