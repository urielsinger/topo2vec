import ast
import os
import time
from typing import List, Tuple

import numpy as np

from shapely.geometry import Point
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import json


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


def full_path_name_to_full_path(full_path: str, name: str):
    full_path = os.path.join(full_path, name + '.npy')
    return full_path


def cache_path_name_to_full_path(cache_dir: str, file_path: str, name: str):
    file_name, _ = os.path.splitext(file_path)
    file_name_end = name + '_' + file_name.split('/')[-1]
    full_path = os.path.join(cache_dir, file_name_end + '.npy')
    return full_path


def get_dataset_as_tensor(dataset: Dataset) -> Tuple[Tensor, Tensor]:
    '''

    Args:
        dataset:

    Returns: dataset as a tensor

    '''
    dataset_length = len(dataset)
    return get_random_part_of_dataset(dataset, dataset_length, shuffle=False)


def get_random_part_of_dataset(dataset: Dataset, size_wanted: int, shuffle=True) -> Tuple[Tensor, Tensor]:
    '''
    Args:
        dataset:
        size_wanted: number of rows wanted in the generated Tensor
        shuffle: get shuffled from the dataset(randomized) or not.
    Returns: a tuple: (the images tensor, the labels tensor)

    '''

    data_loader = DataLoader(dataset, shuffle=shuffle, num_workers=0, batch_size=size_wanted)
    images_as_tensor, y = next(iter(data_loader))
    return images_as_tensor, y


def get_paths_and_names_wanted(list_wanted, class_paths_list, class_names_list):
    index_names = list(enumerate(class_names_list))
    class_index_names_special = [index_name for index_name in
                                 index_names if index_name[1]
                                 in list_wanted]

    class_names_special = [index_name[1] for index_name in
                           class_index_names_special]
    class_paths_special = [class_paths_list[index_name[0]] for
                           index_name in class_index_names_special]

    return class_paths_special, class_names_special


def svm_accuracy_on_dataset_in_latent_space(SVMClassifier, dataset, model):
    X, y = get_dataset_as_tensor(dataset)
    if model.hparams.use_gpu:
        X = X.cuda()
    _, latent = model.forward(X)
    if model.hparams.use_gpu:
        latent = latent.cpu()
    predicted = SVMClassifier.predict(latent.numpy())
    accuracy = accuracy_score(y.numpy(), predicted)
    return accuracy


def save_points_to_json_file(points: List[Point], class_name: str, file_dir: str):
    data = {}
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

    file_path = os.path.join(file_dir, class_name + '.json')
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile)

    return file_path
