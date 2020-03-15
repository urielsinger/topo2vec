from abc import ABC, abstractmethod

from topo2vec import visualizer

from typing import List

from shapely.geometry import Point

class Task(ABC):
    '''
    A class representing a task to handle
    '''
    def __init__(self, dataset_builder, feature_aggregator, model):
        self.dataset_builder = dataset_builder
        self.feature_aggregator = feature_aggregator
        self.model = model
        self.data_handler = visualizer

        self.train_feature_aggregator()
        self.fit_model()

    def train_feature_aggregator(self):
        '''
        fit the feature aggregator
        runs after initialization ends, before the run of the NN model itself
        '''
        pass

    def fit_model(self):
        '''
        fit the model, if needed
        called in the initialization, after training the feature extractor
        '''
        pass

    @abstractmethod
    def run(self, evaluation_points: List[Point]):
        '''
        run the evaluation phase itself.
        Args:
            evaluation_points:

        Returns: Nothing

        '''

        pass




