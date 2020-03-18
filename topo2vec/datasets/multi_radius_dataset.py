from typing import List

from shapely.geometry import Point, Polygon
from torch.utils.data import Dataset

from topo2vec import visualizer
from topo2vec import mask_visualizer

import numpy as np

from topo2vec.common.geographic.geo_utils import check_if_point_in_range


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

    def add_points_as_patches_to_actual_patches(self, points: List[Point]):
        '''
        add the ndarrays that represent the points to the self.actual_points list
        that is the actual data of the dataset
        Args:
            points: points list
        '''
        if self.outer_polygon is not None:
            points = [point for point in points if
                      check_if_point_in_range(point, self.outer_polygon)]

        new_patches, _ = visualizer.get_points_as_np_array(points, self.radii)

        if self.actual_patches is not None:
            all_patches = [self.actual_patches, new_patches]
            self.actual_patches = np.concatenate(all_patches)
        else:
            self.actual_patches = new_patches

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
            points = [point for point in points if check_if_point_in_range(point)]

        new_patches, _ = mask_visualizer.get_points_as_np_array(points, self.radii)

        if self.mask_patches is not None:
            all_mask_patches = [self.mask_patches, new_patches]
            self.mask_patches = np.concatenate(all_mask_patches)
        else:
            self.mask_patches = new_patches

    def __len__(self):
        return len(self.actual_patches)
