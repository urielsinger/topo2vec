# the topography profiler module


import os
import random
from typing import List

import torch
from shapely.geometry import Point, Polygon

import numpy as np
from torch.backends import cudnn
from torch.utils.data import DataLoader

from topo2vec.common.geographic.geo_utils import sample_grid_in_poly
from topo2vec.common.other_scripts import save_points_to_json_file, str_to_int_list
from topo2vec.common.visualizations import convert_multi_radius_ndarray_to_printable, plot_n_np_arrays
from topo2vec.datasets.class_dataset import ClassDataset
from topo2vec.modules import Classifier
from pathlib import Path

################################################################################
# Load the extension module
################################################################################
from topo2vec.modules.knearestneighbourstester import KNearestNeighboursTester

FINAL_MODEL_DIR = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/data/final_model'
FINAL_SEED = 665
FINAL_HPARAMS = Classifier.get_args_parser().parse_args(
    ['--total_dataset_size', '1000',
     '--arch', 'AdvancedConvNetLatent',
     '--name', 'final_model',
     '--pytorch_module', 'Classifier',
     '--latent_space_size', '35',
     ]
)
USER_DEFINED = 'user_defined'
WORKING_POLYGON = Polygon([Point(5, 45), Point(5, 50), Point(10, 50),
                           Point(10, 45), Point(5, 45)])

FORWARD_BATCH_SIZE = 1024

random.seed(FINAL_SEED)
torch.manual_seed(FINAL_SEED)
cudnn.deterministic = True
np.random.seed(FINAL_SEED)

load_path = os.path.join(FINAL_MODEL_DIR, 'final_model.pt')

final_model_classifier = Classifier(FINAL_HPARAMS)
final_model_classifier.load_state_dict(torch.load(load_path))
final_model_classifier.eval()

FINAL_RADII = str_to_int_list(FINAL_HPARAMS.radii)

def build_new_dataset_for_query(points: List[Point], class_name: str = 'no_name') \
        -> ClassDataset:
    queried_classes_path = os.path.join(FINAL_MODEL_DIR, 'queried_classes')
    Path(queried_classes_path).mkdir(parents=True, exist_ok=True)
    class_file_path = save_points_to_json_file(points, class_name, queried_classes_path)
    points_dataset = ClassDataset(class_file_path, 0, FINAL_RADII,
                                  len(points), outer_polygon=WORKING_POLYGON,
                                  dataset_type_name=USER_DEFINED, return_point=True)
    points_used = points_dataset.points_locations

    if len(points_dataset) != len(points):
        print('some of the points are not in the right area and thus ignored')

    if len(points_dataset) == 0:
        raise Exception('there is no single point in the data that is acceptable')

    return points_dataset, points_used


def get_features(points: List[Point], class_name: str = 'no_name') -> np.ndarray:
    '''
    get features the model extracted for the points

    Args:
        points: the points (x:lon, y:lat) for which to extract the features

    Returns: the points' features, as (len(points), latent_size) np.ndarray.

    '''
    points_dataset, _ = build_new_dataset_for_query(points, class_name)
    class_dataloader = DataLoader(points_dataset, FORWARD_BATCH_SIZE)
    latent_features_list = []
    for batch in class_dataloader:
        X, _, _ = batch
        _, latent_features_batch = final_model_classifier(X.detach().float())
        latent_features_list.append(latent_features_batch.detach())
    np_latent_features = np.concatenate(latent_features_list, axis=0)

    return np_latent_features


def get_all_class_points_in_polygon(polygon: Polygon, deg_step: float, class_name: str) -> List[Point]:
    '''
        TODO: default of deg_eps is quarter the radius.

    get all the class points in a certain polygon.

    Args:
        polygon: The polygon to search in
        deg_step: what is the x-y resolution to search inside the polygon (in degrees)
        class_name: the class to search for

    Returns: all the points in this area which are in the polygon and classified as
    the class_name using the classifier

    '''
    points_in_polygon = sample_grid_in_poly(polygon, deg_step)

    points_in_polygon_dataset, points_used = build_new_dataset_for_query(points_in_polygon, class_name)
    assert len(points_in_polygon_dataset) == len(points_used)

    class_dataloader = DataLoader(points_in_polygon_dataset, FORWARD_BATCH_SIZE)

    points_list = []
    patches_list = []
    class_number = final_model_classifier.class_names.index(class_name)
    for batch in class_dataloader:
        X, _, points_used = batch
        predicted = final_model_classifier.get_classifications(X.detach().float())
        class_indexes = predicted.detach() == class_number
        patches_list.append(X[class_indexes, :])
        points_list.append(points_used[class_indexes, :])

    np_points_patches = np.concatenate(patches_list, axis=0)
    points_used = np.concatenate(points_list, axis=0)
    return points_used, np_points_patches


def get_top_n_similar_points_in_polygon(points: List[Point], n: int, polygon: Polygon,
                                        meters_step: float, class_name: str = 'no_name') -> List[Point]:
    '''
    Args:
        points: the points to search points similar to
        n: number of points to search for
        polygon: The polygon to search in
        meters_step: what is the x-y resolution to search inside the polygon (in meters)

    Returns: list of n points that are similar to the points in here.
    '''
    points_in_polygon = sample_grid_in_poly(polygon, meters_step)
    points_in_polygon_dataset, points_typical_used = build_new_dataset_for_query(points_in_polygon, 'dataset')

    typical_points_dataset, _ = build_new_dataset_for_query(points, class_name)

    knn_tester = KNearestNeighboursTester(FINAL_RADII, final_model_classifier, n,
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