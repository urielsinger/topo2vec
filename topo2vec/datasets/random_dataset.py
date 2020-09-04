import os
import time
from pathlib import Path
from typing import List

from shapely.geometry import Polygon
from torch import tensor

from common.geographic import geo_utils
from topo2vec.constants import CACHE_BASE_DIR
from topo2vec.datasets.multi_radius_dataset import MultiRadiiDataset


class RandomDataset(MultiRadiiDataset):
    def __init__(self, num_points: int, original_radiis: List[List[int]],
                 outer_polygon: Polygon, radii: List[int] = None,
                 label: int = 0, seed=None):
        '''

        Args:
            num_points: mi,ber of random points to generate
            original_radiis: original_radiis needed around each point. list of lists
            outer_polygon: the polygon to pick randomliu from
            radii: a list that the first number ios the size we want the arrays to be resized to
            seed: random seed for building the random stuff
        '''
        super().__init__(original_radiis, radii)
        random_points = geo_utils.sample_points_in_poly(outer_polygon, num_points, seed=seed)
        self.full_base_dir = os.path.join(CACHE_BASE_DIR, 'datasets', 'random', f'size_{num_points}_radii{radii}_orig_radii_{original_radiis}_{time.time()}')
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
        return self.actual_patches[index], tensor([float(self.label)])
