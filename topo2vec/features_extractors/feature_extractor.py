from abc import ABC
from typing import List

from shapely.geometry import Point
import numpy as np

class FeatureExtractor(ABC):
    def __init__(self):
        pass

    def fit(self):
        '''

        fit the FeatureExtractor object, only if needed.

        '''
        pass

    def transform(self, points: List[Point]) -> np.ndarray:
        '''

        to be implemented - 4extract all the relevant features_extractors from the code.

        '''
        pass

    def _transform(self, point: Point) -> np.ndarray:
        '''

        to be implemented - extract the feature for 1 point as an np.ndarray

        '''
        pass

