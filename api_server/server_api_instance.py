import json
import os
from typing import Dict

import shapely
import sys

from flask import Flask, request
from pathlib import Path

from shapely import wkt
from tqdm import tqdm

from common.list_conversions_utils import points_list_to_lists_list, floats_list_to_points_list

my_path = os.path.abspath(__file__)
parent_path = Path(my_path).parent.parent
sys.path.append(str(parent_path))

from topo2vec.modules import topography_profiler as tp

app = Flask(__name__)


@app.route('/get_all_classifications', methods=['POST'])
def get_all_classifications() -> Dict:
    '''
    the request should contain a json dictionary contains the following:
    1. a WKT representing a polygon in the 'polygon' field
    2. an int representing the wanted longtitugonal an lattitugonal resolution in the 'meters_step' field
    3. a json representing a list of strings - the wanted names - in the 'class_names' field
    4. a json representing a list of floats in [0,1] - the wanted classes' thresholds for probability to accept as
    the class' instance - in the 'thresholds' field

    an example of how to create an appropriate request is in the -
    client_lib.get_all_classifications_in_polygon() function
    Returns: a json dictionary of the data the user asked for, including:
        a. the locations of all the points
        b. the index of each point's class in the get_available_class_names() returned array.

    '''
    if request.is_json:
        json_dictionary_in = request.get_json()
        polygon = shapely.wkt.loads(json_dictionary_in['polygon'])
        meters_step = int(json_dictionary_in['meters_step'])
        class_names = json.loads(json_dictionary_in['class_names'])
        thresholds = json.loads(json_dictionary_in['thresholds'])
        _, locations, class_indices = tp.get_all_points_and_classes(polygon, meters_step, class_names, thresholds)
        locations = locations.tolist()
        locations_jsoned = json.dumps(locations)
        class_indices_jsoned = json.dumps(class_indices)
        json_dictionary_out = {
            'locations': locations_jsoned,
            'class_indices': class_indices_jsoned
        }
        data = json.dumps(json_dictionary_out)
        return data
    return None

@app.route('/get_class', methods=['POST'])
def get_class_points():
    '''
        the request should contain a json dictionary contains the following:
    1. a WKT representing a polygon in the 'polygon' field
    2. an int representing the wanted longtitugonal an lattitugonal resolution in the 'meters_step' field
    3. a string representing the wanted class name - in the 'class_name' field
    4. a float in [0,1] - the wanted class' threshold - in the 'thresholds' field

    an example of how to create an appropriate request is in the -
    client_lib.get_all_class_points_in_polygon() function
    Returns: a json dictionary of the data the user asked for, including:
        a. the locations of all the points
        b.the tiles arround the point in the multiple Radii's format.

    '''
    if request.is_json:
        json_dictionary_in = request.get_json()
        polygon = shapely.wkt.loads(json_dictionary_in['polygon'])
        meters_step = int(json_dictionary_in['meters_step'])
        class_name = json_dictionary_in['class_name']
        threshold = float(json_dictionary_in['threshold'])
        class_points, class_patches = tp.get_all_class_points_in_polygon(polygon, meters_step, class_name, threshold)
        class_points_jsoned = json.dumps(class_points.tolist())
        class_patches_jsoned = json.dumps(class_patches.tolist())
        json_dictionary_out = {
            'class_points': class_points_jsoned,
            'class_patches': class_patches_jsoned
        }
        data = json.dumps(json_dictionary_out)
        return data
    return None


@app.route('/get_similar', methods=['POST'])
def get_top_n_similar_points_in_polygon():
    '''
        the request should contain a json dictionary contains the following:
    1. a WKT representing a polygon in the 'polygon' field
    2. an int representing the wanted longtitugonal an lattitugonal resolution in the 'meters_step' field
    3. an int representing the number of wanted similar points to send back - in the 'n' field
    4. json representing a list of points - the wanted points to search for anothers similar to it

    an example of how to create an appropriate request is in the -
    client_lib.get_top_n_similar_points_in_polygon() function
    Returns: a json dictionary of the data the user asked for, including:
        a. the locations of all the points
        b.the tiles arround the point in the multiple Radii's format.

    '''
    if request.is_json:
        json_dictionary_in = request.get_json()
        polygon = shapely.wkt.loads(json_dictionary_in['polygon'])
        meters_step = int(json_dictionary_in['meters_step'])
        n = int(json_dictionary_in['n'])
        points_got = json.loads(json_dictionary_in['points'])
        points = floats_list_to_points_list(points_got)
        class_patches, class_points = tp.get_top_n_similar_points_in_polygon(points, n, polygon, meters_step)
        class_points_jsoned = json.dumps(points_list_to_lists_list(class_points))
        class_patches_jsoned = json.dumps(class_patches.numpy().tolist())
        json_dictionary_out = {
            'class_points': class_points_jsoned,
            'class_patches': class_patches_jsoned
        }
        data = json.dumps(json_dictionary_out)
        return data
    return None


@app.route('/get_features', methods=['POST'])
def get_features():
    '''

        the request should contain a json dictionary contains the following:
    1. json representing a list of points - the wanted points to latent space / features.

    an example of how to create an appropriate request is in the -
    client_lib.get_latent_for_points() function
    Returns: a json dictionary of the data the user asked for, including:
        a. the features/ latent space of every point
        b. the points that the user asked for.
    '''
    if request.is_json:
        json_dictionary_in = request.get_json()
        points_got = json.loads(json_dictionary_in['points'])
        points = floats_list_to_points_list(points_got)
        features_np = tp.get_features(points)
        features_jsoned = json.dumps(features_np.tolist())
        points_as_list_lists_jsoned = json.dumps(points_list_to_lists_list(points))

        json_dictionary_out = {
            'features': features_jsoned,
            'points': points_as_list_lists_jsoned
        }
        data = json.dumps(json_dictionary_out)
        return data
    return None


@app.route('/get_working_polygon', methods=['POST', 'GET'])
def get_working_polygon():
    '''
    an example of how to create an appropriate request is in the -
    client_lib.get_working_polygon() function
    Returns: a json dictionary of the data the user asked for, including:
        a. the polygon available currently to ask for.

    '''

    polygon = tp.get_working_polygon()
    polygon_wkt = polygon.wkt
    json_dictionary_out = {
        'polygon': polygon_wkt
    }
    data = json.dumps(json_dictionary_out)
    return data


@app.route('/set_working_polygon', methods=['POST', 'GET'])
def set_working_polygon():
    '''
    set the working polygon in the topogrpahy_profiler in the good way.
                the request should contain a json dictionary contains the following:
    1. a WKT representing a polygon in the 'polygon' field
        an example of how to create an appropriate request is in the -
    client_lib.set_working_polygon() function
    Returns:
        an informative string
    '''
    if request.is_json:
        json_dictionary_in = request.get_json()
        polygon_wkt = json_dictionary_in['polygon']
        Polygon = wkt.loads(polygon_wkt)
        tp.set_working_polygon(Polygon)
        return "the polygon is set to the new value"
    return "didnt work"


@app.route('/get_available_class_names', methods=['POST', 'GET'])
def get_available_class_names():
    '''
    an example of how to create an appropriate request is in the -
    client_lib.get_available_class_names() function
    Returns: a json dictionary of the data the user asked for, including:
        a. a json object of the list of class names, according to the order in which the server's
        classifier is dealing with them.

    '''
    class_names = tp.get_available_class_names()
    class_names_jsoned = json.dumps(class_names)
    json_dictionary_out = {
        'class_names': class_names_jsoned
    }
    data = json.dumps(json_dictionary_out)
    return data


if __name__ == '__main__':
    app.run()
