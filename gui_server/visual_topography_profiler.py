from typing import List

from shapely.geometry import Point, Polygon

import folium
import base64
import matplotlib

from api_client import client_lib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from folium import IFrame

from common.pytorch.visualizations import convert_multi_radius_ndarray_to_printable

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


def _get_working_polygon():
    '''
    Returns:

    '''
    return client_lib.get_working_polygon()


def point_to_location(point: Point) -> List:
    '''
    convert a point to the the list type that folium uses
    Args:
        point: the point

    Returns: a lo

    '''
    return [point.x, point.y]


def get_working_polygon_center():
    center_point = _get_working_polygon().centroid
    return point_to_location(center_point)[::-1]


class TopoMap:
    '''
    A class for building a folium map containing th topography profiler data.
    '''

    def __init__(self, polygon_of_interest: Polygon = None, draw_polygon=True):
        self.version = 0
        self.polygon_of_interest = polygon_of_interest
        if self.polygon_of_interest is not None:
            self.center = point_to_location(self.polygon_of_interest.centroid)[::-1]
        else:
            self.center = get_working_polygon_center()
        self.init_basic_map()

        if self.polygon_of_interest is not None and draw_polygon:
            # draw points on the polygon
            exterior = self.polygon_of_interest.exterior.coords.xy
            lons = exterior[1]
            lats = exterior[0]
            for lon, lat in zip(lons, lats):
                folium.Marker([lon, lat], icon=folium.Icon(color='orange', prefix='fa', icon='flag')).add_to(self.map)
            # draw an actual polygon
            # folium.GeoJson(self.polygon_of_interest).add_to(self.map)

            # folium.vector_layers.Polygon([[point.y, point.x] for point in self.polygon_of_interest])

    def get_all_available_classes(self) -> List[str]:
        '''

        Returns: all available classes for the classifier

        '''
        return client_lib.get_available_class_names()

    def get_folium_map(self):
        '''

        Returns: The folium map that was built.

        '''
        return self.map

    def init_basic_map(self, zoom: int = 11):
        geo_map = folium.Map(
            location=self.center,
            zoom_start=zoom,
            tiles='Stamen Terrain'
        )
        self.map = geo_map

    def add_points_with_text(self, points: List[Point], color: str = 'red',
                             text: str = 'None', tooltip='Click me!'):
        '''

        Args:
            points: the points to show on the map
            color: the color the points should have
            test: the tesxt that all the points should contain
            tooltip: the text pops with hover

        Returns:

        '''
        for point in points:
            point = [point.x, point.y]
            folium.Marker(point[::-1], popup=f'<i>{text}</i>', tooltip=tooltip,
                          icon=folium.Icon(color=color)).add_to(self.map)

    def add_points_as_lists_with_text(self, points: List[List[int]], color: str = 'red',
                                      text: str = 'None'):
        '''

        Args:
            points: the points to show on the map
            color: the color the points should have
            test: the tesxt that all the points should contain
            tooltip: the text pops with hover

        Returns:

        '''
        for point in points:
            folium.Marker(point[::-1], popup=f'<i>{text}</i>',
                          icon=folium.Icon(color=color)).add_to(self.map)

    def add_points_with_images(self, points: List[Point], images: np.ndarray, color: str = 'red',
                               resolution=100, tooltip='Click me!'):
        '''
        add points with images.
        Args:
            points: the points to show on the map
            images: the images that pop when clicking. len(images) should be len(points)
            color: the color of the points
            resolution:
            tooltip: text that comes on hover

        Returns:

        '''
        _, height, width = images.shape
        scale = 10
        for point, image in zip(points, images):
            if type(point) is Point:
                point = [point.x, point.y]
            html = '<img src="data:image/png;base64,{}">'.format
            encoded = self.get_encoded_image(image, resolution=resolution, scale=scale)
            iframe = IFrame(html(encoded), width=(width * scale) + 20, height=(height * scale) + 20)
            popup = folium.Popup(iframe, max_width=530, max_height=300)
            icon = folium.Icon(color=color, icon="ok")
            folium.Marker(point[::-1], popup=popup, tooltip=tooltip,
                          icon=icon).add_to(self.map)

    def get_encoded_image(self, image: np.ndarray, resolution: int = 75, station='42', scale=10) -> List:
        '''
        wncode an image
        Args:
            image: the image to encode
            resolution:
            station:
            scale: scale up the image (n,n) -> (scale*n, scale*n)

        Returns: a decoded image.

        '''
        png = 'mpld3_{}.png'.format(station)
        larger_image = np.kron(image, np.ones((scale, scale)))
        plt.imsave(png, larger_image, dpi=resolution)
        encoded_decoded = base64.b64encode(open(png, 'rb').read()).decode()
        return encoded_decoded

    def add_all_class_points(self, polygon: Polygon, meters_step: float, class_name: str, color='red'):
        '''
        plot all points of class to show
        Args:
            polygon: the polygon to search in
            meters_step: the resolution of the search in the polygon
            class_name: the class to search for (should be one of "class_names" defined in constants file.
            color: the color of the points

        Returns:

        '''
        points_used, np_points_patches_used = client_lib.get_all_class_points_in_polygon(polygon, meters_step,
                                                                                         class_name)
        np_points_patches_used = convert_multi_radius_ndarray_to_printable(np_points_patches_used, dir=False)
        self.add_points_with_images(points_used, np_points_patches_used, color)

    def add_similar_points(self, points: List[Point], polygon: Polygon, meters_step: float, n: int, color='red'):
        '''
        aded similar points to the map
        Args:
            points: the points the new point should be similar to
            polygon: the polygon to search in
            meters_step: the resolution of the search in the polygon
            n: the number of points to plot that are similar
            color: the color of points that are similar. from FOLIUM_COLORS

        Returns:

        '''
        closest_points, closest_images = client_lib.get_top_n_similar_points_in_polygon(points, n, polygon, meters_step)
        closest_images = convert_multi_radius_ndarray_to_printable(closest_images, dir=False)
        self.add_points_with_images(closest_points, closest_images, color)

    def add_random_class_points(self, polygon: Polygon, meters_step: float, class_name: str,
                                color='red', max_num: int = 25, threshold: float = 0):
        '''
        add only a random portion of the class_points that were retrived.
        Args:
            polygon: the polygon to search in
            meters_step: the resolution of the search in the polygon
            class_name: the class to add
            color: the color of the points wanted. from FOLIUM_COLORS
            max_num: the maximum number of points wanted (if there are less - it won't work...)

        Returns:

        '''
        np_points_used, np_points_patches_used = client_lib.get_all_class_points_in_polygon(polygon, meters_step,
                                                                                            class_name, threshold)
        assert len(np_points_used) == len(np_points_patches_used)
        if max_num != 100:
            picked_indices = np.random.choice(list(range(len(np_points_used))), max_num)
            points_used_picked = np_points_used[picked_indices]
            points_patches_used_picked = np_points_patches_used[picked_indices]
        else:
            points_used_picked = np_points_used
            points_patches_used_picked = np_points_patches_used

        points_patches_used_picked = convert_multi_radius_ndarray_to_printable(points_patches_used_picked, dir=False)
        self.add_points_with_images(points_used_picked, points_patches_used_picked, color)
