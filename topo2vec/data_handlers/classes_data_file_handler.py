import os
import fiona
import json
import random
import numpy as np

from shapely.geometry import Point, Polygon
from typing import List

from common.dataset_utils import cache_path_name_to_full_path
from common.list_conversions_utils import floats_list_to_points_list_till_size, load_list_from_file, save_list_to_file
from topo2vec.constants import CACHE_BASE_DIR
from topo2vec.data_handlers.data_handler import DataHandler
from common.geographic.geo_utils import check_if_point_in_polygon


class ClassesDataFileHadler(DataHandler):
    '''
    A class that helps loading classes data
    '''
    def __init__(self, file_path, cache_dir=CACHE_BASE_DIR):
        '''
        load all the points that are inside a points list
        Args:
            file_path: The .shp or. geojson file of the class's data

        Returns: a Points list of all the points in the file
        (if the file contains lines - all the points in the line)

        '''
        file_name, file_extension = os.path.splitext(file_path)
        self.file_name = file_name.split('\\')[-1]
        full_path = cache_path_name_to_full_path(cache_dir, file_path, 'points')

        points_list = load_list_from_file(full_path)
        if points_list is not None:
            self.points_list = points_list
        else:
            if file_extension == '.shp':
                collection = fiona.open(file_path, encoding='ISO8859-1')
                new_features = list(collection)

            elif file_extension == '.geojson':
                with open(file_path, encoding='utf-8') as bottom_peaks_file:
                    data = json.load(bottom_peaks_file)
                if 'features' in data:
                    new_features = data['features']
                else:
                    new_features = data['elements']

            elif file_extension == '.json':
                with open(file_path, encoding='utf-8') as bottom_peaks_file:
                    data = json.load(bottom_peaks_file)
                new_features = data['elements']

            else:
                raise Exception('No points in file!')

            self.points_list = []
            for index in range(len(new_features)):
                coord_as_points_list = self._get_coord_as_floats_list(index, new_features, file_extension)
                self.points_list += coord_as_points_list

            save_list_to_file(full_path, self.points_list)

    def get_random_subset_in_polygon(self, wanted_size: int, outer_polygon: Polygon = None, seed=None):
        if seed is not None:
            random.seed(seed)
        if outer_polygon is not None:
            points_inside_polygon = []
            for i in range(0, len(self.points_list), 2):
                point = Point(self.points_list[i], self.points_list[i+1])
                if check_if_point_in_polygon(point, outer_polygon):
                    points_inside_polygon.append(point)
                    if len(points_inside_polygon) >= 5 * wanted_size:
                        break
        else:
            points_inside_polygon = floats_list_to_points_list_till_size(self.points_list, 5 * wanted_size)

        if len(points_inside_polygon) >= wanted_size:
            return random.sample(points_inside_polygon, wanted_size)
        else:
            raise Exception(f'Not enough points in class in this region as needed!\n'
                            f'class: {self.file_name}, wanted size: {wanted_size}, '
                            f'available: {len(points_inside_polygon)}')

    def _get_coord_as_floats_list(self, index: int, new_features: np.ndarray, file_extension:str) -> List[Point]:
        '''
        Args:
            index:
            new_features: The features ndarray of the coordinate, got from the image

        Returns: a list of all the points inside a row.

        '''
        if file_extension == '.shp': # or file_extension == '.geojson':
            curr_idx_coords = new_features[index]['geometry']['coordinates']
            if len(curr_idx_coords) != 0:
                if type(curr_idx_coords[0]) == float:
                    chosen_point = [curr_idx_coords[0], curr_idx_coords[1]]
                    return chosen_point

                elif type(curr_idx_coords[0]) == list:
                    curr_idx_coord = random.choice(curr_idx_coords)
                    # return [Point(curr_idx_coord[0], curr_idx_coord[1]) for curr_idx_coord in curr_idx_coords]
                    chosen_point = [curr_idx_coord[0], curr_idx_coord[1]]
                    return chosen_point
        elif file_extension == '.json' or  file_extension == '.geojson':
            try:
                chosen_point = [new_features[index]['lon'], new_features[index]['lat']]
                return chosen_point
            except:
                pass
        return []