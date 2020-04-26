import Flask
import math
from shapely.geometry import Point

from topo2vec.data_handlers.data_from_file_handler import DataFromFileHandler

BASE_LOCATION = '/home/topo2vec_kavitzky/topo2vec/'

# run in background - the service for getting visualizations of lon, lats
ELEVATION_BASE_DIR1 = BASE_LOCATION + 'data/elevation/45,5,50,15'

MASK_BASE_DIR = BASE_LOCATION + 'data/elevation/45,5,50,15'
equatorial_circumference_of_earth = 40075016.686 #m
pixel_to_radius = 10 #m

class TileServer():
    def __init__(self, data_dir, available_polygon):
        self.available_polygon = available_polygon
        self.data_visualizer = DataFromFileHandler([ELEVATION_BASE_DIR1], [(5, 45, 15, 50)])

    def get_point(self, lon:float, lat: float, r:int):

    def get_patch(self, longitude, latitude, zoomlevel):
        horizontal_distance_tile = equatorial_circumference_of_earth * math.cos(latitude) / 2**(zoomlevel)
        radius_needed = horizontal_distance_tile/pixel_to_radius
        self.data_visualizer.get_point_as_np_array(Point(longitude, latitude), )
        return self.get_point(longitude)