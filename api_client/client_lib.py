import requests
import json
import numpy as np
from shapely import wkt
from shapely.geometry import Polygon, Point

from common.list_conversions_utils import points_list_to_floats_list

PROTOCOL = 'http'
IP = '159.122.160.134'
PORT = '9876'

ADDRESS = f'{PROTOCOL}://{IP}:{PORT}'

def build_polygon(low_lon, low_lat, high_lon, high_lat):
    poly = Polygon([Point(low_lon, low_lat), Point(low_lon, high_lat), Point(high_lon, high_lat),
                    Point(high_lon, low_lat), Point(low_lon, low_lat)])
    return poly


def get_all_class_points_in_polygon(polygon, meters_step, class_name):
    request_dict = {'polygon': polygon.wkt,
                    'meters_step': meters_step,
                    'class_name': class_name}
    url = f'{ADDRESS}/get_class'
    print(f'waiting for {url}')
    response = requests.post(url=url, json=request_dict)
    json_dictionary_in = response.json()
    class_points_jsoned = json_dictionary_in['class_points']
    class_patches_jsoned = json_dictionary_in['class_patches']
    class_points = json.loads(class_points_jsoned)
    class_patches = json.loads(class_patches_jsoned)
    return np.array(class_points), np.array(class_patches)

def get_top_n_similar_points_in_polygon(points, n, polygon, meters_step):
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

def get_latent_for_points(points):
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

def get_working_polygon():
    '''
    Returns:

    '''
    url = f'{ADDRESS}/get_working_polygon'
    print(f'waiting for {url}')
    response = requests.post(url=url)
    json_dictionary_in = response.json()
    polygon_wkt = json_dictionary_in['polygon']
    Polygon = wkt.loads(polygon_wkt)
    return Polygon

def get_available_class_names():
    '''
    Returns:

    '''
    url = f'{ADDRESS}/get_available_class_names'
    print(f'waiting for {url}')
    response = requests.post(url=url)
    json_dictionary_in = response.json()
    class_names_jsoned = json_dictionary_in['class_names']
    class_names = json.loads(class_names_jsoned)
    return class_names


