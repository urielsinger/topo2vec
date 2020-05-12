from typing import List, Tuple

import requests
import json
import numpy as np
from shapely import wkt
from shapely.geometry import Polygon, Point

from common.list_conversions_utils import points_list_to_floats_list

PROTOCOL = 'http'
# inside docker:
# IP = 'topo2vec_web_api4_1'
# outside docker:
IP = '159.122.160.134'
PORT = '9876'

ADDRESS = f'{PROTOCOL}://{IP}:{PORT}'


def build_polygon(low_lon, low_lat, high_lon, high_lat):
    '''

    Args:
        low_lon:
        low_lat:
        high_lon:
        high_lat:

    Returns: a rectangular polygon with the corners according to the input

    '''
    poly = Polygon([Point(low_lon, low_lat), Point(low_lon, high_lat), Point(high_lon, high_lat),
                    Point(high_lon, low_lat), Point(low_lon, low_lat)])
    return poly


def get_all_classifications_in_polygon(polygon: Polygon, meters_step: int, class_names: List[str],
                                       thresholds: List[float]) -> Tuple[List[Point], List[int]]:
    '''
    get the list of all points of the certain class_name.
    (according to topo2vec.topography_profiler.get_all_points_and_classes())

    Uses the server_api_instance.get_all_classifications() function

    Args:
        polygon: the polygon to earch in. should be inside the get_working_polygon()
        meters_step: the resolution to build the grid we search on
        class_names: the classes i want to get the data about.
        thresholds: list of thresholds for the probability that the points are of the corresponding class,
        according to the server's classifier

    Returns: get the points, and the indices of each of them

    '''
    request_dict = {'polygon': polygon.wkt,
                    'meters_step': meters_step,
                    'class_names': json.dumps(class_names),
                    'thresholds': json.dumps(thresholds)}
    url = f'{ADDRESS}/get_all_classifications'
    print(f'waiting for {url}')
    response = requests.post(url=url, json=request_dict)
    json_dictionary_in = response.json()
    locations_jsoned = json_dictionary_in['locations']
    class_indices_jsoned = json_dictionary_in['class_indices']
    locations = json.loads(locations_jsoned)
    class_indices = json.loads(class_indices_jsoned)
    print(f'retrieved{len(locations)} points')
    return locations, class_indices


def get_all_class_points_in_polygon(polygon: Polygon, meters_step: int, class_name: str,
                                    threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    '''
    get the list of all points of the certain class_name.
    (according to topo2vec.topography_profiler.get_all_class_points_in_polygon())

    Uses the server_api_instance.get_class_points() function
    Args:
        points: the list of Points
        polygon: the polygon to earch in. should be inside the get_working_polygon()
        meters_step: the resolution to build the grid we search on
        thresholds: a threshold for the probability that the points are of the corresponding class,
        according to the server's classifier


    Returns: a tuple: the points, the 3-layers patches (in dims according to the radii)

    '''

    request_dict = {'polygon': polygon.wkt,
                    'meters_step': meters_step,
                    'class_name': class_name,
                    'threshold': threshold}
    url = f'{ADDRESS}/get_class'
    print(f'waiting for {url}')
    response = requests.post(url=url, json=request_dict)
    json_dictionary_in = response.json()
    class_points_jsoned = json_dictionary_in['class_points']
    class_patches_jsoned = json_dictionary_in['class_patches']
    class_points = json.loads(class_points_jsoned)
    class_patches = json.loads(class_patches_jsoned)
    print(f'retrieved{len(class_points)} points')
    return np.array(class_points), np.array(class_patches)


def get_top_n_similar_points_in_polygon(points: List[Point], n: int,
                                        polygon: Polygon, meters_step: int) -> List[Point]:
    '''
    get the list of most similar points to the average of the points in the latent space
    (according to topo2vec.topography_profiler. get_top_n_similar_points_in_polygon)

    Uses the server_api_instance.get_top_n_similar_points_in_polygon() function
    Args:
        points: the list of Points
        n: number of points to get
        polygon: the polygon to earch in. should be inside the get_working_polygon()
        meters_step: the resolution to build the grid we search on

    Returns: a tuple: the points, the 3-layers patches (in dims according to the radii)

    '''
    points_to_send = json.dumps(points_list_to_floats_list(points))
    request_dict = {'points': points_to_send,
                    'polygon': polygon.wkt,
                    'meters_step': meters_step,
                    'n': n}
    url = f'{ADDRESS}/get_similar'
    print(f'waiting for {url}')
    response = requests.post(url=url, json=request_dict)
    json_dictionary_in = response.json()
    class_points_jsoned = json_dictionary_in['class_points']
    class_patches_jsoned = json_dictionary_in['class_patches']
    class_points = json.loads(class_points_jsoned)
    class_patches = json.loads(class_patches_jsoned)
    return np.array(class_points), np.array(class_patches)


def get_latent_for_points(points: List[Point]) -> Tuple[np.ndarray, np.ndarray]:
    '''
    according to topo2vec.topography_profiler.get_features()

    Uses the server_api_instance.get_features() function
    Args:
        points: a list of the points to get the latent

    Returns: the latent for each point, row per point

    '''
    points_to_send = json.dumps(points_list_to_floats_list(points))
    request_dict = {'points': points_to_send}
    url = f'{ADDRESS}/get_features'
    print(f'waiting for {url}')
    response = requests.post(url=url, json=request_dict)
    json_dictionary_in = response.json()
    features_jsoned = json_dictionary_in['features']
    points_jsoned = json_dictionary_in['points']
    features = json.loads(features_jsoned)
    points = json.loads(points_jsoned)
    return np.array(points), np.array(features)


def get_working_polygon() -> Polygon:
    '''
    according to topo2vec.topography_profiler.get_working_polygon()
            Uses the server_api_instance.get_working_polygon() function

    Returns: the polygon in which the server is able to work now
    '''
    url = f'{ADDRESS}/get_working_polygon'
    print(f'waiting for {url}')
    response = requests.post(url=url)
    json_dictionary_in = response.json()
    polygon_wkt = json_dictionary_in['polygon']
    Polygon = wkt.loads(polygon_wkt)
    return Polygon


def set_working_polygon(polygon: Polygon) -> str:
    '''
    according to topo2vec.topography_profiler.set_working_polygon()
        Uses the server_api_instance.set_working_polygon() function

    Returns: The response status code

    '''
    polygon_wkt = polygon.wkt
    request_dict = {'polygon': polygon_wkt}
    url = f'{ADDRESS}/set_working_polygon'
    print(f'waiting for {url}')
    response = requests.post(url=url, json=request_dict)
    print(response.content)
    return response.status_code


def get_available_class_names() -> List[str]:
    '''
    according to topo2vec.topography_profiler.get_available_class_names()
        Uses the server_api_instance.get_available_class_names() function
    Returns:  a list of string names of the classes the classifier is classifying to

    '''
    url = f'{ADDRESS}/get_available_class_names'
    print(f'waiting for {url}')
    response = requests.post(url=url)
    json_dictionary_in = response.json()
    class_names_jsoned = json_dictionary_in['class_names']
    class_names = json.loads(class_names_jsoned)
    return class_names
