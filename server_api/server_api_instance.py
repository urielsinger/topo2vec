import json
import os
import pickle

import shapely
import sys

from flask import Flask, request
from pathlib import Path

from topo2vec.common.other_scripts import floats_list_to_points_list
from topo2vec.constants import GET_SIMILAR_POINTS_ROUTE

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
        points_list = floats_list_to_points_list(json_dictionary_in['points'])
        points = json.loads(points_list)
        class_points, class_patches = tp.get_top_n_similar_points_in_polygon(points, n, polygon, meters_step)
        class_points_jsoned = json.dumps(class_points.tolist())
        class_patches_jsoned = json.dumps(class_patches.tolist())
        json_dictionary_out = {
            'class_points': class_points_jsoned,
            'class_patches': class_patches_jsoned
        }
        data = json.dumps(json_dictionary_out)
    return data


@app.route('/get_latent', methods=['POST'])
def hello_world():
    return 'Hello World!'


if __name__ == '__main__':
    app.run()
