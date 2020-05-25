from typing import List

import cv2
import numpy as np
from shapely.geometry import Point
from topo2vec.data_handlers.data_handler import DataHandler
from topo2vec.data_handlers.elevation_data_square import ElevationDataSquare


class DataFromFileHandler(DataHandler):
    def __init__(self, elevation_base_dirs, bounding_boxs=[(5, 49, 6, 50)]):
        '''

        Args:
            elevation_base_dir: The directory in which all the images are stored
            bounding_box: the bounding box of the data
        '''
        super().__init__()
        self.elevation_data_squares = {}
        for base_dir, bounding_box in zip(elevation_base_dirs, bounding_boxs):
            self.elevation_data_squares[str(bounding_box)] = ElevationDataSquare(base_dir, bounding_box)


    def get_points_as_np_array(self, center_points: List[Point], original_radii: List[int]
                               , radii: int = None) -> np.ndarray:
        '''
        Args:
            center_points: the points to get as np arrays, if possible
            original_radii: the original_radiis (L1 norm) to look in for the points

        Returns: an np array of shape:
         (num of possible points, len(original_radiis) , 2*min(original_radiis) + 1, 2*min(original_radiis) + 1)
        that is the actual elevation map in the neighbourhood of each point.
        '''
        if radii is None:
            radii = original_radii[0]
        point_multi_patches, points_used = \
            self.get_points_as_list_of_np_arrays(center_points, original_radii, radii)
        point_multipatches_ndarray = np.stack(point_multi_patches)
        return point_multipatches_ndarray, points_used

    def get_points_as_list_of_np_arrays(self, center_points:
    List[Point], original_radii: List[int], radii: int) -> List[Point]:
        '''
        Args:
            center_points: the points to get as np arrays, if possible
            original_radii: the original_radiis (L1 norm) to look in for the points

        Returns: an np array of shape:
         (num of possible points, len(original_radiis) , 2*min(original_radiis) + 1, 2*min(original_radiis) + 1)
        that is the actual elevation map in the neighbourhood of each point.
        '''
        point_multi_patches = []
        points_used = []
        resized_shape = (2 * radii + 1, 2 * radii + 1)
        for point in center_points:
            point_patches = []
            for original_radius in original_radii:
                patch = self.get_point_as_np_array(point, original_radius)
                if original_radius != radii and patch.size != 0:
                    patch = cv2.resize(patch, dsize=resized_shape)
                if patch.size != 0 and patch.shape == resized_shape and np.min(patch) > -3000:
                    point_patches.append(patch)
            if len(point_patches) == len(original_radii):
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
        res_map = self.crop_image(center_point.x, center_point.y, radius)
        if normalize and res_map.size > 1:
            res_map = (res_map - np.min(res_map)) / (np.max(res_map) - np.min(res_map))
        return res_map


    def crop_image(self, lon, lat, radius):
        for image in self.elevation_data_squares.values():
            if image.point_inside(lon, lat):
                return image.crop_image(lon, lat, radius)
                break

