import os
from typing import List

from shapely.geometry import Point
import json
from datetime import datetime

def points_to_json_dict(points: List[Point], class_name: str) -> dict:
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
    return data
