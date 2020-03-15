from typing import List

from shapely.geometry import Point

from topo2vec import visualizer
import numpy as np

from topo2vec.features_extractors.patch_based_feature_extractor import PatchBasedFeatureExtractor


class PurePatchExtractor(PatchBasedFeatureExtractor):
    def __init__(self, radius):
        super().__init__(radius)

    def _transform(self, point:Point) -> np.ndarray:
        '''

        Args:
            points: list of points to calculate the feature around


        Returns:  a (num of points, patch_size(=WxH)) numpy array consisting of the features_extractors per each point

        '''
        patch = visualizer.get_data_as_np_array(point, self.radius)
        patch = patch - np.mean(patch)
        return patch

