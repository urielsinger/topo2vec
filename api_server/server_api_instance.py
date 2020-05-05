import json
import os

import shapely
import sys

from flask import Flask, request
from pathlib import Path

from common.list_conversions_utils import points_list_to_lists_list, floats_list_to_points_list

my_path = os.path.abspath(__file__)
parent_path = Path(my_path).parent.parent
sys.path.append(str(parent_path))


from topo2vec.modules import topography_profiler as tp

app = Flask(__name__)

@app.route('/get_class', methods=['POST'])
def get_class_points():
     if request.is_json:
        json_dictionary_in = request.get_json()
        polygon = shapely.wkt.loads(json_dictionary_in['polygon'])
        meters_step = int(json_dictionary_in['meters_step'])
        class_name = json_dictionary_in['class_name']
        class_points, class_patches = tp.get_all_class_points_in_polygon(polygon, meters_step, class_name)
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
    polygon = tp.get_working_polygon()
    polygon_wkt = polygon.wkt
    json_dictionary_out = {
        'polygon': polygon_wkt
    }
    data = json.dumps(json_dictionary_out)
    return data


@app.route('/get_available_class_names', methods=['POST', 'GET'])
def get_available_class_names():
    class_names = tp.get_available_class_names()
    class_names_jsoned = json.dumps(class_names)
    json_dictionary_out = {
        'class_names': class_names_jsoned
    }
    data = json.dumps(json_dictionary_out)
    return data


if __name__ == '__main__':
    app.run()
