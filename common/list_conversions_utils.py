import ast
import os
import time
from typing import List

import numpy as np
from shapely.geometry import Point


def str_to_int_list(string_list: str):
    list_list = ast.literal_eval(string_list)
    int_list = [int(x) for x in list_list]
    return int_list


def str_to_str_list(string_list: str):
    return ast.literal_eval(string_list)


def list_to_str(string_list: str):
    list_list = ast.literal_eval(string_list)
    int_list = [int(x) for x in list_list]
    return int_list


def points_list_to_floats_list(points_list):
    floats_list = []
    for point in points_list:
        floats_list.append(point.x)
        floats_list.append(point.y)
    return floats_list


def points_list_to_lists_list(points_list):
    lists_list = []
    for point in points_list:
        lists_list.append([point.x, point.y])
    return lists_list


def floats_list_to_points_list_till_size(floats_list: object, size: int) -> object:
    assert len(floats_list) % 2 == 0
    points_list = []
    for i in range(0, len(floats_list), 2):
        if i >= 2 * size:
            pass
        points_list.append(Point(floats_list[i], floats_list[i + 1]))
    return points_list


def floats_list_to_points_list(floats_list: object) -> object:
    assert len(floats_list) % 2 == 0
    points_list = []
    for i in range(0, len(floats_list), 2):
        points_list.append(Point(floats_list[i], floats_list[i + 1]))
    return points_list


def load_list_from_file(full_path: str) -> List:
    if os.path.exists(full_path):
        # print(f'loading from: {full_path}')
        t0 = time.time()
        the_list = np.load(full_path).tolist()
        t1 = time.time()
        # print(f'loaded. time it took:{t1 - t0} sec')
        return the_list
    return None


def save_list_to_file(full_path: str, the_list: List):
    # print(f'saving to: {full_path}')
    t0 = time.time()
    np.save(full_path, the_list)
    t1 = time.time()
    # print(f'saved. time it took:{t1 - t0} sec')