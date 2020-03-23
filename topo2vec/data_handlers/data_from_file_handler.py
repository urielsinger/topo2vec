import math
import os
from typing import List, Tuple

import numpy as np
from shapely.geometry import Point
from skimage import io
from sklearn.externals._pilutil import imresize

from topo2vec.data_handlers.data_handler import DataHandler


class DataFromFileHandler(DataHandler):
    def __init__(self, elevation_base_dir, bounding_box=(5, 49, 6, 50)):
        '''

        Args:
            elevation_base_dir: The directory in which all the images are stored
            bounding_box: the bounding box of the data
        '''
        super().__init__()
        self.min_lon, self.min_lat, self.max_lon, self.max_lat = bounding_box
        self.images = {}
        for min_lin in range(self.min_lon, self.max_lon, 1):
            for min_lat in range(self.min_lat, self.max_lat, 1):
                im_name = self.lon_lat_to_string(min_lin, min_lat) + '_AVE_DSM.tif'
                im = self.load_image(os.path.join(elevation_base_dir, im_name))
                self.images[self.lon_lat_to_string(min_lin, min_lat)] = im

        self.H, self.W = self.images[self.lon_lat_to_string(
            self.min_lon, self.min_lat)].shape

        self.step_lon = 1 / self.W
        self.step_lat = 1 / self.H

    def point_to_string(self, point: Point) -> str:
        return self.lon_lat_to_string(point.x, point.y)

    def lon_lat_to_string(self, lon: float, lat: float) -> str:
        lon_floor, lat_floor = self.floor_lon_lat(lon, lat)
        zeros_for_lon = '0' * (3 - len(str(lon_floor)))
        zeros_for_lat = '0' * (3 - len(str(lat_floor)))

        return f'N{zeros_for_lat}{lat_floor}E{zeros_for_lon}{lon_floor}'

    def floor_lon_lat(self, lon: float, lat: float) -> Tuple[int, int]:
        lon_floor = int(math.floor(lon))
        lat_floor = int(math.floor(lat))
        return lon_floor, lat_floor

    def load_image(self, image_path):
        '''

        Args:
            image_path: the path to the image: .tif or .hgt files acceptable

        Returns: an ndarray of the data image

        '''
        file_name, file_extension = os.path.splitext(image_path)
        if file_extension == '.tif':
            print('.tif extension')
            return io.imread(image_path)

        elif file_extension == '.hgt':
            SAMPLES = 3601
            with open(image_path, 'rb') as hgt_data:
                # Each data is 16bit signed integer(i2) - big endian(>)
                elevations = np.fromfile(hgt_data, np.dtype('>i2'), SAMPLES * SAMPLES) \
                    .reshape((SAMPLES, SAMPLES))

                return elevations

    def get_data_as_np_array(self, center_point: Point, radius: int, dtype='tiff') -> np.ndarray:
        '''

        Args:
            center_point: the point around which to extract the data
            radius: the radius to 2 sides the patch should go through
            dtype: 'tiff' or 'png'

        Returns:

        '''
        normalize = False if dtype == "tiff" else True
        return self.get_elevation_map(center_point.x, center_point.y, radius, normalize)

    def get_points_as_np_array(self, center_points: List[Point], radii: List[int]) -> np.ndarray:
        '''

        Args:
            center_points: the points to get as np arrays, if possible
            radii: the radii (L1 norm) to look in for the points

        Returns: an np array of shape:
         (num of possible points, len(radii) , 2*min(radii) + 1, 2*min(radii) + 1)
        that is the actual elevation map in the neighbourhood of each point.
        '''
        min_radius = min(radii)
        point_multi_patches = []
        standard_size = (2 * min_radius + 1, 2 * min_radius + 1)
        points_used = []
        for point in center_points:
            point_patches = []
            for radius in radii:
                patch = self.get_point_as_np_array(point, radius)
                if radius != min_radius and patch.size != 0:
                    patch = imresize(patch, size=standard_size)
                if patch.size != 0 and patch.shape == standard_size and np.min(patch) > -3000:
                    point_patches.append(patch)
            if len(point_patches) == len(radii):
                point_patches_ndarray = np.stack(point_patches)
                point_multi_patches.append(point_patches_ndarray)
                points_used.append(point)
        point_multipatches_ndarray = np.stack(point_multi_patches)
        return point_multipatches_ndarray, points_used

    def get_point_as_np_array(self, center_point: Point, radius: int) -> np.ndarray:
        '''

        Args:
            center_point: the point to do it around
            radius: the radius to 2 sides the patch should go through

        Returns: the certain patch

        '''
        #TODO: WHAT IF FROM TWO FILES?
        normalize = True
        lon, lat = center_point.x, center_point.y

        im_min_lon, im_min_lat = self.floor_lon_lat(lon, lat)
        lon_index = int((lon - im_min_lon) / self.step_lon)
        lat_index = int((lat - im_min_lat) / self.step_lat)
        relevant_image = self.images[self.lon_lat_to_string(im_min_lon, im_min_lat)]
        res_map = relevant_image[-lat_index - radius:-lat_index + radius + 1,
                  lon_index - radius:lon_index + radius + 1]
        if normalize and res_map.size != 0:
            res_map = (res_map - np.min(res_map)) / (np.max(res_map) - np.min(res_map))
        return res_map
