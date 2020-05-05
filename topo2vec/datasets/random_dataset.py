import os
from pathlib import Path
from typing import List

from shapely.geometry import Polygon
from torch import tensor

from common.geographic import geo_utils
from topo2vec.constants import CACHE_BASE_DIR
from topo2vec.datasets.multi_radius_dataset import MultiRadiusDataset


class RandomDataset(MultiRadiusDataset):
    def __init__(self, num_points: int, radii: List[int], outer_polygon: Polygon, label: int = 0):
        '''

        Args:
            num_points: mi,ber of random points to generate
            radii: radii needed around each point.
        '''
        super().__init__(radii)
        random_points = geo_utils.sample_points_in_polygon(outer_polygon, num_points)
        self.full_base_dir = os.path.join(CACHE_BASE_DIR, 'datasets', 'random', f'size_{num_points}')

        Path(self.full_base_dir).mkdir(parents=True, exist_ok=True)

        self.add_points_as_patches_to_actual_patches(random_points)
        if self.use_masks:
            self.add_points_as_patches_to_mask_patches(random_points)
        self.label = label

    def __getitem__(self, index):
        '''
        a simple getitem, but of course if the mask is used,
        returns it too.
        Args:
            index:

        Returns:

        '''
        # if self.mask_patches is not None and self.use_masks:
        #    return (self.actual_patches[index], self.mask_patches[index], tensor([0.]))
        return (self.actual_patches[index], tensor([float(self.label)]))
