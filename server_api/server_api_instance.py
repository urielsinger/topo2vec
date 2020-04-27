import os
import pickle
import sys

from flask import Flask, request
from pathlib import Path

my_path = os.path.abspath(__file__)
parent_path = Path(my_path).parent.parent
sys.path.append(str(parent_path))

from topo2vec.constants import GET_CLASS_POINTS_ROUTE, GET_SIMILAR_POINTS_ROUTE

from topo2vec.modules import topography_profiler as tp

app = Flask(__name__)


# TODO: change to json
@app.route(GET_CLASS_POINTS_ROUTE, methods=['POST'])
def get_class_points():
    polygon, meters_step, class_name = pickle.loads(request.get_data())
    class_points, class_patches = tp.get_all_class_points_in_polygon(polygon, meters_step, class_name)
    data = pickle.dumps((class_points, class_patches), protocol=2)
    return data


@app.route(GET_SIMILAR_POINTS_ROUTE, methods=['POST'])
def get_top_n_similar_points_in_polygon():
    points, n, polygon, meters_step = pickle.loads(request.get_data())
    class_points, class_patches = tp.get_top_n_similar_points_in_polygon(points, n, polygon, meters_step)
    data = pickle.dumps((class_points, class_patches), protocol=2)
    return data


@app.route(GET_CLASS_POINTS_ROUTE)
def hello_world():
    return 'Hello World!'


if __name__ == '__main__':
    app.run()
