from typing import List

from shapely.geometry import Point, Polygon
from torch import Tensor
from torch.utils.data import Dataset

from visualization_server import visualizer
import numpy as np

from common.geographic.geo_utils import check_if_point_in_polygon
from topo2vec.helper_functions import full_path_name_of_dataset_data_to_full_path
from common.list_conversions_utils import points_list_to_floats_list, floats_list_to_points_list, load_list_from_file, \
    save_list_to_file


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
            full_path_actual_patches = full_path_name_of_dataset_data_to_full_path(self.full_base_dir, 'actual_patches')
            actual_patches = load_list_from_file(full_path_actual_patches)

            full_path_points_locations = full_path_name_of_dataset_data_to_full_path(self.full_base_dir, 'points_locations')
            points_locations = load_list_from_file(full_path_points_locations)

        if actual_patches is None or points_locations is None:
            if self.outer_polygon is not None:
                points = [point for point in points if
                          check_if_point_in_polygon(point, self.outer_polygon)]

            new_patches, points_locations_list = visualizer.get_points_as_np_array(points, self.radii)
            self.points_locations = points_locations_list
            if self.actual_patches is not None:
                all_patches = [self.actual_patches, new_patches]
                self.actual_patches = Tensor(np.concatenate(all_patches))
            else:
                self.actual_patches = Tensor(new_patches)

            if self.full_base_dir is not None:
                save_list_to_file(full_path_actual_patches, self.actual_patches)
                save_list_to_file(full_path_points_locations, points_list_to_floats_list(self.points_locations))

        else:
            self.actual_patches = Tensor(actual_patches)
            self.points_locations = floats_list_to_points_list(points_locations)

    def __len__(self):
        return len(self.actual_patches)
