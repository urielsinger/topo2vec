import ast
import os
import time
from pathlib import Path
from typing import List

import numpy as np

from shapely.geometry import Point


def str_to_int_list(string_list: str):
    list_list = ast.literal_eval(string_list)
    int_list = [int(x) for x in list_list]
    return int_list

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


def floats_list_to_points_list(floats_list: object) -> object:
    assert len(floats_list) % 2 == 0
    points_list = []
    for i in range(0, len(floats_list), 2):
        points_list.append(Point(floats_list[i], floats_list[i + 1]))
    return points_list


def load_list_from_file(full_path: str) -> List:
    if os.path.exists(full_path):
        print(f'loading from: {full_path}')
        t0 = time.time()
        the_list = np.load(full_path).tolist()
        t1 = time.time()
        print(f'loaded. time it took:{t1 - t0} sec')
        return the_list
    return None


def get_dataset_dir_base_path(cache_dir: str, file_path: str,
                              dataset_type_name: str):
    file_name, _ = os.path.splitext(file_path)
    type_area_name = file_name.split('/')[-1]
    full_base_dir = os.path.join(cache_dir, 'datasets', type_area_name, dataset_type_name)
    return full_base_dir


def save_list_to_file(full_path: str, the_list: List):
    print(f'saving to: {full_path}')
    t0 = time.time()
    np.save(full_path, the_list)
    t1 = time.time()
    print(f'saved. time it took:{t1 - t0} sec')

def full_path_name_to_full_path(full_path:str, name:str):
    full_path = os.path.join(full_path, name + '.npy')
    return full_path

def cache_path_name_to_full_path(cache_dir: str, file_path: str, name: str):
    file_name, _ = os.path.splitext(file_path)
    file_name_end = name + '_' + file_name.split('/')[-1]
    full_path = os.path.join(cache_dir, file_name_end + '.npy')
    return full_path
