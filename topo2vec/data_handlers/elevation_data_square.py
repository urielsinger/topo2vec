import logging

import tqdm
from skimage import io
import math
import os
from typing import Tuple

import numpy as np
from shapely.geometry import Point

from common.geographic.geo_utils import lon_lat_to_string


class ElevationDataSquare:
    def __init__(self, elevation_base_dir, bounding_box):
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
                im_name = 'ALPSMLC30_' + lon_lat_to_string(min_lon, min_lat) + '_DSM.tif'
                logging.info(im_name)
                tqdm.tqdm([0], desc=im_name)
                im = self.load_image(os.path.join(elevation_base_dir, im_name))
                this_lon_images.append(im)
            this_lon_images = np.concatenate(list(reversed(this_lon_images)), axis=0)
            images.append(this_lon_images)
        self.im = np.concatenate(images, axis=1)

        self.H, self.W = self.im.shape
        self.step_lon = (self.max_lon - self.min_lon) / self.W
        self.step_lat = (self.max_lat - self.min_lat) / self.H

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

    def point_inside(self, lon, lat):
        return lon > self.min_lon and lon < self.max_lon and \
               lat < self.max_lat and lat > self.min_lat

    def point_to_string(self, point: Point) -> str:
        return lon_lat_to_string(point.x, point.y)

    @staticmethod
    def load_image(image_path):
        '''

        Args:
            image_path:

        Returns: an ndarray of the image of type

        '''
        file_name, file_extension = os.path.splitext(image_path)
        if file_extension == '.tif':
            logging.info('loading a .tif extension')
            return io.imread(image_path)

        elif file_extension == '.hgt':
            SAMPLES = 3601
            with open(image_path, 'rb') as hgt_data:
                # Each data is 16bit signed integer(i2) - big endian(>)
                elevations = np.fromfile(hgt_data, np.dtype('>i2'), SAMPLES * SAMPLES) \
                    .reshape((SAMPLES, SAMPLES))

                return elevations