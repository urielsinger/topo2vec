import math
import os
from typing import List, Tuple

import cv2
import numpy as np
from shapely.geometry import Point
from skimage import io
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
        images = []
        for min_lon in range(self.min_lon, self.max_lon, 1):
            this_lon_images = []
            for min_lat in range(self.min_lat, self.max_lat, 1):
                im_name = self.lon_lat_to_string(min_lon, min_lat) + '_AVE_DSM.tif'
                im = self.load_image(os.path.join(elevation_base_dir, im_name))
                this_lon_images.append(im)
            this_lon_images = np.concatenate(list(reversed(this_lon_images)), axis=0)
            images.append(this_lon_images)
        self.im = np.concatenate(images, axis=1)

        self.H, self.W = self.im.shape
        self.step_lon = (self.max_lon-self.min_lon) / self.W
        self.step_lat = (self.max_lat-self.min_lat) / self.H

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
            image_path:

        Returns: an ndarray of the image of type

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
            lon:
            lat:
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
        point_multi_patches, points_used = \
            self.get_points_as_list_of_np_arrays(center_points, radii)
        point_multipatches_ndarray = np.stack(point_multi_patches)
        return point_multipatches_ndarray, points_used

    def get_points_as_list_of_np_arrays(self, center_points:
    List[Point], radii: List[int]) -> List[Point]:
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
                    patch = cv2.resize(patch, dsize=standard_size)
                if patch.size != 0 and patch.shape == standard_size and np.min(patch) > -3000:
                    point_patches.append(patch)
            if len(point_patches) == len(radii):
                point_patches_ndarray = np.stack(point_patches)
                point_multi_patches.append(point_patches_ndarray)
                points_used.append(point)
        return point_multi_patches, points_used

    def get_point_as_np_array(self, center_point: Point, radius: int) -> np.ndarray:
        '''

        Args:
            center_point: the point to do it arrounf
            radius: the radius to 2 sides the patch should go through

        Returns:

        '''
        normalize = True
        return self.get_elevation_map(center_point.x, center_point.y, radius, normalize)

    def crop_image(self, lon, lat, radius):
        '''

        Args:
            lon: patches center lon
            lat: patches center lat
            radius: the distance from lon,lat to each side

        Returns: The certain patch

        '''
        lon_index = int((lon - self.min_lon) / self.step_lon)
        lat_index = int((lat - self.min_lat) / self.step_lat)
        res_map = self.im[-lat_index - radius:-lat_index + radius + 1,
                  lon_index - radius:lon_index + radius + 1]
        return res_map

    def get_elevation_map(self, lon, lat, radius, normalize=False):
        '''
        normalize the crop_image's result
        Args:
            lon:
            lat:
            radius:
            normalize: normalize or not the elevation map

        Returns: the certain patch normalized if needed

        '''
        res_map = self.crop_image(lon, lat, radius)
        if normalize and res_map.size > 1:
            res_map = (res_map - np.min(res_map)) / (np.max(res_map) - np.min(res_map))
        return res_map
