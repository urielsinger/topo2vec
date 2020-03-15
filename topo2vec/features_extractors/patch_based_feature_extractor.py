from shapely.geometry import Point

from topo2vec.features_extractors.feature_extractor import FeatureExtractor

from typing import List
import numpy as np

class PatchBasedFeatureExtractor(FeatureExtractor):
    def __init__(self, radius):
        self.radius = radius

    def transform(self, points: List[Point]) -> np.ndarray:
        '''

        Args:
            points: list of points to calculate the feature around


        Returns:  a (num of points, patch_size(=WxH)) numpy array consisting of the features_extractors per each point

        '''
        patch_list = []
        for point in points:
            feat = self._transform(point) # returns a 1 x feat_length vector
            patch_list.append(feat)

        return np.stack(patch_list)