import os
from typing import List

from shapely.geometry import Point
import json
from datetime import datetime

def save_points_to_json_file(points: List[Point], class_name: str, file_dir: str):
    data={}
    data['elements'] = []
    for point in points:
        point_dict = {
            'type': 'node',
            'lon': point.x,
            'lat': point.y,
            'tags': {
                "user_defined": class_name
            }
        }
        data['elements'].append(point_dict)

    file_path = os.path.join(file_dir, class_name + str(datetime.now())+'.json')
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile)

    return file_path
