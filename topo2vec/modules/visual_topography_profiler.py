from typing import List

from shapely.geometry import Point, Polygon

import topo2vec.modules.topography_profiler as tp
import folium
import base64
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from folium import IFrame

from topo2vec.common.visualizations import convert_multi_radius_ndarray_to_printable

FOLIUM_COLORS = [
    'red',
    'blue',
    'gray',
    'darkred',
    'lightred',
    'orange',
    'beige',
    'green',
    'darkgreen',
    'lightgreen',
    'darkblue',
    'lightblue',
    'purple',
    'darkpurple',
    'pink',
    'cadetblue',
    'lightgray',
    'black'
]


def get_working_polygon():
    return tp.WORKING_POLYGON


def point_to_location(point: Point):
    return [point.x, point.y]


def get_working_polygon_center():
    center_point = get_working_polygon().centroid
    return point_to_location(center_point)[::-1]


class TopoMap:
    def __init__(self, polygon_of_interest: Polygon = None):
        self.polygon_of_interest = polygon_of_interest
        if self.polygon_of_interest is not None:
            self.center = point_to_location(self.polygon_of_interest.centroid)[::-1]
        else:
            self.center = get_working_polygon_center()

        self.init_basic_map()

    def get_folium_map(self):
        return self.map

    def init_basic_map(self, zoom: int = 11):
        geo_map = folium.Map(
            location=self.center,
            zoom_start=zoom,
            tiles='Stamen Terrain'
        )
        self.map = geo_map

    def add_points_with_text(self, points: List[Point], color: str = 'red'):
        tooltip = 'Click me!'
        for point in points:
            folium.Marker(point[::-1], popup='<i>peaks</i>', tooltip=tooltip,
                          icon=folium.Icon(color=color)).add_to(self.map)

    def add_points_with_images(self, points: List[Point], images: np.ndarray, color: str = 'red',
                               resolution=100):
        tooltip = 'Click me!'
        print(images.shape)
        _, height, width = images.shape
        scale = 10
        for point, image in zip(points, images):
            html = '<img src="data:image/png;base64,{}">'.format
            encoded = self.get_encoded_image(image, resolution=resolution, scale=scale)
            iframe = IFrame(html(encoded), width=(width * scale) + 20, height=(height * scale) + 20)
            popup = folium.Popup(iframe, max_width=530, max_height=300)
            icon = folium.Icon(color=color, icon="ok")
            folium.Marker(point[::-1], popup=popup, tooltip=tooltip,
                          icon=icon).add_to(self.map)

    def get_encoded_image(self, image: np.ndarray, resolution: int = 75, station='42', scale=10) -> List:
        png = 'mpld3_{}.png'.format(station)
        larger_image = np.kron(image, np.ones((scale, scale)))
        plt.imsave(png, larger_image, dpi=resolution)
        encoded = base64.b64encode(open(png, 'rb').read()).decode()
        return encoded

    def add_all_class_points(self, polygon: Polygon, meters_step: float, class_name: str, color='red'):
        points_used, np_points_patches_used = tp.get_all_class_points_in_polygon(polygon, meters_step, class_name)
        np_points_patches_used = convert_multi_radius_ndarray_to_printable(np_points_patches_used, dir=False)
        self.add_points_with_images(points_used, np_points_patches_used, color)

    def add_similar_points(self, points: List[Point], polygon: Polygon, meters_step: float, n:int, color='red'):
        closest_images, closest_points = tp.get_top_n_similar_points_in_polygon(points, n, polygon, meters_step)
        closest_images = closest_images.numpy()
        closest_points = [point_to_location(point) for point in closest_points]
        closest_images = convert_multi_radius_ndarray_to_printable(closest_images, dir=False)
        self.add_points_with_images(closest_points, closest_images, color)

    def add_random_class_points(self, polygon: Polygon, meters_step: float, class_name: str, color='red', max_num=25):
        np_points_used, np_points_patches_used = tp.get_all_class_points_in_polygon(polygon, meters_step, class_name)
        assert len(np_points_used) == len(np_points_patches_used)
        picked_indices = np.random.choice(list(range(len(np_points_used))), max_num)
        points_used_picked = np_points_used[picked_indices]
        points_patches_used_picked = np_points_patches_used[picked_indices]

        points_patches_used_picked = convert_multi_radius_ndarray_to_printable(points_patches_used_picked, dir=False)
        self.add_points_with_images(points_used_picked, points_patches_used_picked, color)
