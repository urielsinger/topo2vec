import logging
from typing import List

import requests
import json
import numpy as np
from shapely import wkt
from shapely.geometry import Polygon, Point

from common.list_conversions_utils import points_list_to_floats_list

PROTOCOL = 'http'
# inside docker:
IP = 'topo2vec_kavitzky_web_api_1'
# outside docker:
# IP = '159.122.160.134'
PORT = '7653'

ADDRESS = f'{PROTOCOL}://{IP}:{PORT}'


def build_polygon(low_lon, low_lat, high_lon, high_lat):
    poly = Polygon([Point(low_lon, low_lat), Point(low_lon, high_lat), Point(high_lon, high_lat),
                    Point(high_lon, low_lat), Point(low_lon, low_lat)])
    return poly


def get_all_classifications_in_polygon(polygon: Polygon, meters_step: int, class_names: List[str],
                                       thresholds: List[float], test_radius:int):
    request_dict = {'polygon': polygon.wkt,
                    'meters_step': meters_step,
                    'class_names': json.dumps(class_names),
                    'thresholds': json.dumps(thresholds),
                    'test_radius': test_radius}
    url = f'{ADDRESS}/get_all_classifications'
    logging.info(f'waiting for {url}')
    response = requests.post(url=url, json=request_dict)
    json_dictionary_in = response.json()
    locations_jsoned = json_dictionary_in['locations']
    class_indices_jsoned = json_dictionary_in['class_indices']
    locations = json.loads(locations_jsoned)
    class_indices = json.loads(class_indices_jsoned)
    logging.info(f'retrieved{len(locations)} points')
    return locations, class_indices


def get_all_class_points_in_polygon(polygon, meters_step, class_name, threshold, test_radius):
    request_dict = {'polygon': polygon.wkt,
                    'meters_step': meters_step,
                    'class_name': class_name,
                    'threshold': threshold,
                    'test_radius': test_radius}
    url = f'{ADDRESS}/get_class'
    logging.info(f'waiting for {url}')
    response = requests.post(url=url, json=request_dict)
    json_dictionary_in = response.json()
    class_points_jsoned = json_dictionary_in['class_points']
    class_patches_jsoned = json_dictionary_in['class_patches']
    class_points = json.loads(class_points_jsoned)
    class_patches = json.loads(class_patches_jsoned)
    logging.info(f'retrieved{len(class_points)} points')
    return np.array(class_points), np.array(class_patches)


def get_top_n_similar_points_in_polygon(points, n, polygon, meters_step, test_radius):
    points_to_send = json.dumps(points_list_to_floats_list(points))
    request_dict = {'points': points_to_send,
                    'polygon': polygon.wkt,
                    'meters_step': meters_step,
                    'n': n,
                    'test_radius': test_radius}
    url = f'{ADDRESS}/get_similar'
    logging.info(f'waiting for {url}')
    response = requests.post(url=url, json=request_dict)
    json_dictionary_in = response.json()
    class_points_jsoned = json_dictionary_in['class_points']
    class_patches_jsoned = json_dictionary_in['class_patches']
    class_points = json.loads(class_points_jsoned)
    class_patches = json.loads(class_patches_jsoned)
    return np.array(class_points), np.array(class_patches)


def get_latent_for_points(points, test_radius):
    points_to_send = json.dumps(points_list_to_floats_list(points))
    request_dict = {'points': points_to_send,
                    'test_radius': test_radius}
    url = f'{ADDRESS}/get_features'
    logging.info(f'waiting for {url}')
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
    logging.info(f'waiting for {url}')
    response = requests.post(url=url)
    json_dictionary_in = response.json()
    polygon_wkt = json_dictionary_in['polygon']
    Polygon = wkt.loads(polygon_wkt)
    return Polygon


def set_working_polygon(polygon: Polygon):
    '''
    Returns:

    '''
    polygon_wkt = polygon.wkt
    request_dict = {'polygon': polygon_wkt}
    url = f'{ADDRESS}/set_working_polygon'
    logging.info(f'waiting for {url}')
    response = requests.post(url=url, json=request_dict)
    logging.info(response.content)
    return response.status_code


def get_available_class_names():
    '''
    Returns:

    '''
    url = f'{ADDRESS}/get_available_class_names'
    logging.info(f'waiting for {url}')
    response = requests.post(url=url)
    json_dictionary_in = response.json()
    class_names_jsoned = json_dictionary_in['class_names']
    class_names = json.loads(class_names_jsoned)
    return class_names

def get_available_final_model_file_names():
    '''
    Returns:

    '''
    url = f'{ADDRESS}/get_final_model_file_names'
    logging.info(f'waiting for {url}')
    response = requests.post(url=url)
    json_dictionary_in = response.json()
    class_names_jsoned = json_dictionary_in['file_names']
    class_names = json.loads(class_names_jsoned)
    return class_names

def set_final_model(file_name: str):
    '''
    Returns:

    '''
    request_dict = {'final_model_name': file_name}
    url = f'{ADDRESS}/load_final_model'
    logging.info(f'waiting for {url}')
    response = requests.post(url=url, json=request_dict)
    logging.info(response.content)
    return response.status_code

