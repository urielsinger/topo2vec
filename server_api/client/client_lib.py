import requests
import json
import numpy as np
from shapely.geometry import Polygon, Point

from topo2vec.common.other_scripts import points_list_to_floats_list

PROTOCOL = 'http'
IP = '159.122.160.134'
PORT = '6543'

ADDRESS = f'{PROTOCOL}://{IP}:{PORT}'

def build_polygon(low_lon, low_lat, high_lon, high_lat):
    poly = Polygon([Point(low_lon, low_lat), Point(low_lon, high_lat), Point(high_lon, high_lat),
                    Point(high_lon, low_lat), Point(low_lon, low_lat)])
    return poly


def get_class_points(polygon, meters_step, class_name):
    request_dict = {'polygon': polygon.wkt,
                    'meters_step': meters_step,
                    'class_name': class_name}
    url = f'{ADDRESS}/get_class'
    response = requests.post(url=url, json=request_dict)
    print(response)
    json_dictionary_in = response.json()
    class_points_jsoned = json_dictionary_in['class_points']
    class_patches_jsoned = json_dictionary_in['class_patches']
    class_points = json.loads(class_points_jsoned)
    class_patches = json.loads(class_patches_jsoned)
    return np.array(class_points), np.array(class_patches)

def get_top_n_similar_points_in_polygon(points, n, polygon, meters_step):
    request_dict = {'points': json.dumps(points_list_to_floats_list(points)),
                    'polygon': polygon.wkt,
                    'meters_step': meters_step,
                    'n': n}
    url = f'{ADDRESS}/get_similar'
    response = requests.post(url=url, json=request_dict)
    print(response)
    json_dictionary_in = response.json()
    class_points_jsoned = json_dictionary_in['class_points']
    class_patches_jsoned = json_dictionary_in['class_patches']
    class_points = json.loads(class_points_jsoned)
    class_patches = json.loads(class_patches_jsoned)
    return np.array(class_points), np.array(class_patches)

