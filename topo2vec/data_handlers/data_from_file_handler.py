import os
from typing import List

import numpy as np
from shapely.geometry import Point
from skimage import io
from sklearn.externals._pilutil import imresize

from topo2vec.data_handlers.data_handler import DataHandler


class DataFromFileHandler(DataHandler):
    def __init__(self, image_path):
        '''
        Args:
            image_path: the path where the basic image is saved
        '''
        super().__init__()
        self.min_lon, self.min_lat, self.max_lon, self.max_lat = 5, 49, 6, 50
        self.im = self.load_image(image_path)
        self.H, self.W = self.im.shape

    def load_image(self, image_path):
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



    def get_data_as_np_array(self, center_point: Point, radius: int, dtype='tiff')-> np.ndarray:
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

    def get_points_as_np_array(self, center_points:List[Point], radii: List[int]) -> np.ndarray:
        min_radius = min(radii)
        point_multi_patches = []
        standard_size =(2*min_radius+1, 2*min_radius+1)
        points_used = []
        for point in center_points:
            point_patches = []
            for radius in radii:
                patch = self.get_point_as_np_array(point, radius)
                if radius != min_radius and patch.size!=0:
                    patch = imresize(patch, size=standard_size)
                if patch.size!=0 and patch.shape == standard_size and np.min(patch) > -3000:
                    point_patches.append(patch)
            if len(point_patches)!=0:
                point_patches_ndarray = np.stack(point_patches)
                point_multi_patches.append(point_patches_ndarray)
                points_used.append(point)
        point_multipatches_ndarray = np.stack(point_multi_patches)
        return point_multipatches_ndarray, points_used


    def get_point_as_np_array(self, center_point:Point, radius: int)-> np.ndarray:
        '''

        Args:
            center_point: the point to do it arrounf
            radius: the radius to 2 sides the patch should go through

        Returns:

        '''
        normalize = False
        return self.get_elevation_map(center_point.x, center_point.y, radius, normalize)

    def crop_image(self, lon, lat, radius):
        '''

        Args:
            lon: patches center lon
            lat: patches center lat
            radius: the distance from lon,lat to each side

        Returns: The certain patch

        '''
        step_lon = (self.max_lon - self.min_lon) / self.W
        step_lat = (self.max_lat - self.min_lat) / self.H
        lon_index = int((lon - self.min_lon) / step_lon)
        lat_index = int((lat - self.min_lat) / step_lat)
        res_map = self.im[-lat_index - radius:-lat_index + radius + 1, lon_index - radius:lon_index + radius + 1]
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
        if normalize:
            res_map = (res_map - np.min(res_map)) / (np.max(res_map) - np.min(res_map))
        return res_map
