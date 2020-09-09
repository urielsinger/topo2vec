import pickle
from pathlib import Path
from typing import List

from shapely.geometry import Point, Polygon
from torch import Tensor
from torch.utils.data import Dataset

from visualization_server import visualizer
import numpy as np

from common.geographic.geo_utils import check_if_point_in_polygon
from common.dataset_utils import full_path_name_of_dataset_data_to_full_path
from common.list_conversions_utils import points_list_to_floats_list, floats_list_to_points_list, load_list_from_file, \
    save_list_to_file


class MultiRadiiDataset(Dataset):
    '''
    A dataset that supports the making of a point to an k-dim ndarray
    of the neighbourhood of the point in different original_radiis.
    '''

    def __init__(self, original_radiis: List[List[int]], radii: List[int],
                 outer_polygon: Polygon = None):
        '''

        Args:
            original_radiis: the original_radiis of the neighbourhoods.
            radii: if None - the size will be of the first of original radiis
            outer_polygon: if None - ignore, otherwise - take only
            points that are inside it.
        '''
        self.original_radiis = original_radiis
        if radii is not None:
            self.radii = radii[0]
        else:
            self.radii = original_radiis[0][0]
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
            full_path_points_locations = full_path_name_of_dataset_data_to_full_path(self.full_base_dir,
                                                                                     'points_locations')
            points_locations = load_list_from_file(full_path_points_locations)

        if actual_patches is None or points_locations is None:
            if self.outer_polygon is not None:
                points = [point for point in points if
                          check_if_point_in_polygon(point, self.outer_polygon)]

            new_patches_list = []
            all_points_locations_list = []
            for original_radii in self.original_radiis:
                new_patches, points_locations_list = visualizer.get_points_as_np_array(points, original_radii,
                                                                                       self.radii)
                new_patches_list.append(new_patches)
                all_points_locations_list += points_locations_list
            points_locations_list = all_points_locations_list
            new_patches = np.concatenate(new_patches_list)
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

    def save_to_pickle(self, location='data/datasets_for_simclr', name=''):
        Path(location).mkdir(parents=True, exist_ok=True)
        try:
            filename = f'{location}/{name}_dataset_{self.original_radiis}_{self.radii}_{self.labels[0]}.pickle'
        except:
            filename = f'{location}/{name}_dataset_{self.original_radiis}_size_{len(self.actual_patches)}_{self.radii}_{self.label}.pickle'

        pickle_dict = {}
        pickle_dict['data'] = self.actual_patches
        try:
            pickle_dict['labels'] = self.labels
        except:
            pass
        with open(filename, 'wb') as handle:
            pickle.dump(pickle_dict, handle)
