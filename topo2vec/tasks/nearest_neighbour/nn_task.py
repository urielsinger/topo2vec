from typing import List

from shapely.geometry import Point
from sklearn.neighbors import NearestNeighbors

from topo2vec import visualizer
from topo2vec.common.visualizations import plot_n_np_arrays_one_row
from topo2vec.datasets.random_dataset_builder import DatasetBuilder

from topo2vec.feature_aggregators.naive_features_aggregator import NaiveFeaturesAggregator
from topo2vec.tasks.task import Task


DEF_POINTS_FOR_DEF_TASK = [Point(34.70495951,30.89286054),  Point(34.55495951,30.56986054),
                           Point(34.55495951,30.56586054), Point(34.75495951,30.76586054), ]


class NNTask(Task):
    def __init__(self, radius: int=10, random_validation_set_size: int = 1000, k: int = 6, feature_aggregator = None):
        '''
        Args:
            dataset_builder:  to build the ds
            feature_aggregator: The topo2vec feat_extractor to evaluate
            center_points: the points to calc the distances to
            radius: the radius of the patches I'm interested in for the evaluation.
            random_validation_set_size: how big should the random set be.
            k: the k-nearest_neighbour
        '''
        model = NearestNeighbors(n_neighbors=k, metric='euclidean')
        dataset_builder = DatasetBuilder(34.00495952, 30.00286054, 34.90495952, 30.89286054)
        if feature_aggregator is None:
            feature_aggregator = NaiveFeaturesAggregator(radius)

        self.random_validation_set_size = random_validation_set_size
        self.radius = radius
        super().__init__(dataset_builder, feature_aggregator, model)



    def fit_model(self):
        '''

        fit the NN model over the points array

        '''
        self.points = self.dataset_builder.get_random_dataset(self.random_validation_set_size)
        features_table = self.feature_aggregator.transform(self.points)
        self.model.fit(features_table)


    def run(self, evaluation_points: List[Point] = DEF_POINTS_FOR_DEF_TASK):
        '''

        Args:
            evaluation_points: the points to evaluate the model on

        Returns:

        '''
        features_table_center = self.feature_aggregator.transform(evaluation_points)
        for i, center_point in enumerate(evaluation_points):
            distances, indices = self.model.kneighbors(features_table_center[i, :].reshape(1, -1))  # look at it
            patches_lis = []
            closest_points =  [self.points[i] for i in indices[0]]
            closest_points = [center_point] + closest_points
            for pt in closest_points:
                patches_lis.append(visualizer.get_data_as_np_array(pt, self.radius))

            print('The center point[first]: and all the closest points to it[others]:')
            plot_n_np_arrays_one_row(patches_lis)



