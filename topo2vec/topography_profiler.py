# the topography profiler module
import json
import logging
import os
import random
from datetime import datetime

from typing import List, Tuple

import torch
from shapely.geometry import Point, Polygon

import numpy as np
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.geographic.geo_utils import sample_grid_in_poly, build_polygon
from common.other_scripts import points_to_json_dict
from common.list_conversions_utils import str_to_int_list
from topo2vec.constants import BASE_LOCATION
from topo2vec.datasets.class_dataset import ClassDataset
from topo2vec.modules import Classifier
from pathlib import Path
from topo2vec.modules.knearestneighbourstester import KNearestNeighboursTester

################################################################################
# init profiling environment
################################################################################
USER_DEFINED = 'user_defined'
default_working_polygon = Polygon([Point(5, 45), Point(5, 50), Point(10, 50),
                                   Point(10, 45), Point(5, 45)])

goral_hights = build_polygon(34.7, 31.3, 34.9, 31.43)


def set_working_polygon(polygon: Polygon):
    '''
    the working polygon is the polygon which is assumed to contain all the data the user wants
    Args:
        polygon:

    Returns:

    '''
    global WORKING_POLYGON
    WORKING_POLYGON = polygon


def get_working_polygon():
    return WORKING_POLYGON


set_working_polygon(goral_hights)

FINAL_SEED = 665

FORWARD_BATCH_SIZE = 1024
random.seed(FINAL_SEED)
torch.manual_seed(FINAL_SEED)
cudnn.deterministic = True
np.random.seed(FINAL_SEED)

################################################################################
# Load the final model
################################################################################
FINAL_MODEL_DIR = BASE_LOCATION + 'data/final_model'
FINAL_HPARAMS = Classifier.get_args_parser().parse_args(
    ['--total_dataset_size', '2500',
     '--arch', 'BasicConvNetLatent',
     '--name', 'final_model',
     '--pytorch_module', 'Classifier',
     '--latent_space_size', '35',
     '--num_classes', '4',
     ]
)

SUPERRESOLUTION_HPARAMS = Classifier.get_args_parser().parse_args(
    [
        '--save_model',
        '--index_in', '1',
        '--index_out', '0',
        '--learning_rate', '0.0015',
        '--max_epochs', '300',
        '--total_dataset_size', '100000',
        '--arch', 'UNet',
        '--svm_classify_latent_space',
        '--knn_method_for_typical_choosing', 'regular',
        '--name', 'Superresolution',
        '--pytorch_module', 'Superresolution',
        '--random_set_size_for_svm', '2000',
        '--latent_space_size', '600',
        '--svm_classify_latent_space',
        '--test_knn',
        '--original_radiis', '[[34, 136], [68, 272], [102, 408]]',
        '--radii', '[34, 136]',
        '--upsample'
    ]
)
final_model_classifier = None


def load_final_model(final_model_name: str):
    load_path = os.path.join(FINAL_MODEL_DIR, final_model_name)
    global final_model_classifier
    final_model_classifier = Classifier(SUPERRESOLUTION_HPARAMS)
    final_model_classifier.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
    final_model_classifier.eval()


# load_final_model('final_model81624.pt')
load_final_model('Superresolution_UNet_[34, 136]_lr_0.0015_size_100000_num_classes_4_latent_size_600_train_all_resnet_False.pt')

FINAL_RADII = str_to_int_list(SUPERRESOLUTION_HPARAMS.radii)
FINAL_ORIGINAL_RADIIS = str_to_int_list(SUPERRESOLUTION_HPARAMS.original_radiis)


################################################################################
# module functions                                                             #
################################################################################


def _build_new_dataset_for_query(points: List[Point], original_radiis: List[int], test_radius: int,
                                 class_name: str = 'no_name') -> Tuple[ClassDataset, np.ndarray]:
    queried_classes_path = os.path.join(FINAL_MODEL_DIR, 'queried_classes')
    Path(queried_classes_path).mkdir(parents=True, exist_ok=True)
    data = points_to_json_dict(points, class_name)
    class_file_path = os.path.join(queried_classes_path, class_name + str(datetime.now())+'.json')
    with open(class_file_path, 'w') as outfile:
        json.dump(data, outfile)

    # take only the points from the query which are in the WORKING_POLYGON
    points_dataset = ClassDataset(class_path=class_file_path, class_label=0,
                                  # original_radiis=FINAL_ORIGINAL_RADIIS, radii=FINAL_RADII,
                                  original_radiis=[[int(test_radius), int(test_radius) * 2, int(test_radius) * 3]], radii=FINAL_RADII,
                                  wanted_size=len(points), outer_polygon=WORKING_POLYGON,
                                  dataset_type_name=USER_DEFINED, return_point=True)
    points_used = points_dataset.points_locations

    if len(points_dataset) != len(points):
        logging.info('some of the points are not in the right area and thus ignored')

    if len(points_dataset) == 0:
        raise Exception('there is no single point in the data that is acceptable')

    return points_dataset, points_used


def get_features(points: List[Point], test_radius, class_name: str = 'no_name') -> np.ndarray:
    '''
    get features the model extracted for the points

    Args:
        points: the points (x:lon, y:lat) for which to extract the features

    Returns: the points' features, as (len(points), latent_size) np.ndarray.

    '''
    points_dataset, _ = _build_new_dataset_for_query(points, FINAL_ORIGINAL_RADIIS, test_radius, class_name)
    class_dataloader = DataLoader(points_dataset, FORWARD_BATCH_SIZE)
    latent_features_list = []
    for batch in class_dataloader:
        X, _, _ = batch
        _, latent_features_batch = final_model_classifier(X.detach().float())
        latent_features_list.append(latent_features_batch.detach())
    np_latent_features = np.concatenate(latent_features_list, axis=0)

    return np_latent_features


def get_available_class_names() -> str:
    '''

    Returns: a string list of the available class names in the final model classifier

    '''
    return final_model_classifier.class_names


def get_available_final_model_file_names() -> str:
    '''

    Returns: a string list of the available final model files

    '''
    file_names = [f for f in os.listdir(FINAL_MODEL_DIR) if f[-3:] == '.pt']
    return file_names


def get_all_points_and_classes(polygon: Polygon, meters_step: int,
                               class_names: List[str], thresholds: List[int], test_radius: int):
    points_list = []
    patches_list = []
    indices_list = []
    class_numbers = [final_model_classifier.class_names.index(class_name) for class_name in class_names]
    points_in_polygon = sample_grid_in_poly(polygon, meters_step)

    for class_name, class_number, prob_threshold in zip(class_names, class_numbers, thresholds):
        points_in_polygon_dataset, points_used = _build_new_dataset_for_query(points_in_polygon, FINAL_ORIGINAL_RADIIS,
                                                                              test_radius, class_name)
        assert len(points_in_polygon_dataset) == len(points_used)

        class_dataloader = DataLoader(points_in_polygon_dataset, FORWARD_BATCH_SIZE)
        for batch in class_dataloader:
            X, _, points_used = batch
            predicted, probability = final_model_classifier.get_classification_and_probability(X.detach().float())
            class_indexes = predicted.detach() == class_number
            if prob_threshold is not None and prob_threshold > 0:
                class_indexes = class_indexes * (probability > prob_threshold)
            patches_list.append(X[class_indexes, :])
            new_points = points_used[class_indexes, :]
            points_list.append(new_points)
            indices_list += ([class_number] * new_points.shape[0])
    return np.concatenate(patches_list, axis=0), np.concatenate(points_list, axis=0), indices_list


def get_all_class_points_in_polygon(polygon: Polygon, meters_step: float, class_name: str,
                                    test_radius: int, prob_threshold: float = None) -> \
        Tuple[np.ndarray, np.ndarray]:
    '''
    get all the class points in a certain polygon.

    Args:
        polygon: The polygon to search in
        meters_step: what is the x-y resolution to search inside the polygon (in degrees)
        class_name: the class to search for
        prob_threshold: the threshold from which the points are accepted as part of the class.

    Returns: all the points in this area which are in the polygon and classified as
    the class_name using the classifier

    '''
    patches_np, points_np, _ = get_all_points_and_classes(polygon=polygon,
                                                          meters_step=meters_step,
                                                          class_names=[class_name],
                                                          test_radius=test_radius,
                                                          thresholds=[prob_threshold])
    return points_np, patches_np


def get_top_n_similar_points_in_polygon(points: List[Point], n: int, polygon: Polygon,
                                        meters_step: float, test_radius: int, class_name: str = 'no_name') -> List[
    Point]:
    '''
    Args:
        points: the points to search points similar to
        n: number of points to search for
        polygon: The polygon to search in
        meters_step: what is the x-y resolution to search inside the polygon (in meters)

    Returns: list of n points that are similar to the points in here.
    '''
    points_in_polygon = sample_grid_in_poly(polygon, meters_step)
    points_in_polygon_dataset, points_typical_used = _build_new_dataset_for_query(points_in_polygon,
                                                                                  FINAL_ORIGINAL_RADIIS, test_radius,
                                                                                  'dataset')

    typical_points_dataset, _ = _build_new_dataset_for_query(points, FINAL_ORIGINAL_RADIIS, test_radius, class_name)

    knn_tester = KNearestNeighboursTester(FINAL_RADII, FINAL_ORIGINAL_RADIIS, final_model_classifier, n,
                                          method='group_from_file', random_set_size=0,
                                          json_file_of_group='irelevant')

    knn_tester.set_pre_defined_datasets(points_in_polygon_dataset, typical_points_dataset,
                                        points_typical_used)

    closest_images_list_image_space, closest_images_list_latent_space, typical_images_latent_as_np, \
    typical_images_set_to_show, number_to_log, closest_points_list_latent_space = knn_tester.test()

    return closest_images_list_latent_space[0], closest_points_list_latent_space[0]

# polygon_to_search_in = Polygon([Point(5, 45), Point(5, 45.1), Point(5.1, 45.1),
#                                 Point(5.1, 45.1), Point(5, 45)])
#
# points, pathches_lis = get_all_class_points_in_polygon(polygon_to_search_in, 100, 'cliffs')
#
# patches_printable = convert_multi_radius_ndarray_to_printable(pathches_lis)
# patches_printable = patches_printable[12:24]
# plot_n_np_arrays(patches_printable, lines_number=3)
