from typing import List

from shapely.geometry import Point
from torch.utils.data import Dataset

from topo2vec import visualizer
from topo2vec import mask_visualizer

import numpy as np

from topo2vec.common.geographic.geo_utils import check_if_point_in_range


class MultiRadiusDataset(Dataset):

    def __init__(self, radii:List[int] = [10], outer_polygon = None):
        self.radii = radii
        self.actual_patches = None
        self.mask_patches = None
        self.use_masks = False
        self.outer_polygon = outer_polygon

    def add_points_as_patches_to_actual_patches(self, points: List[Point]):
        if self.outer_polygon is not None:
            points = [point for point in points if check_if_point_in_range(point)]

        new_patches, _ = visualizer.get_points_as_np_array(points, self.radii)

        if self.actual_patches is not None:
            all_patches = [self.actual_patches, new_patches]
            self.actual_patches = np.concatenate(all_patches)
        else:
            self.actual_patches = new_patches


    def add_points_as_patches_to_mask_patches(self, points: List[Point]):
        if self.outer_polygon is not None:
            points = [point for point in points if check_if_point_in_range(point)]

        new_patches, _ = mask_visualizer.get_points_as_np_array(points, self.radii)

        if self.mask_patches is not None:
            all_mask_patches = [self.mask_patches, new_patches]
            self.mask_patches = np.concatenate(all_mask_patches)
        else:
            self.mask_patches = new_patches

