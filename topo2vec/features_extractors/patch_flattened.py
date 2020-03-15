from shapely.geometry import Point

from topo2vec import visualizer
import numpy as np

from topo2vec.features_extractors.patch_based_feature_extractor import PatchBasedFeatureExtractor


class PatchFlattened(PatchBasedFeatureExtractor):
    def __init__(self, radius: int):
        super().__init__(radius)

    def _transform(self, point:Point):
        '''

        Args:
            points: list of points to calculate the feature aroiund


        Returns:  a (1, wxh) ndarray for the current point

        '''
        patch = visualizer.get_data_as_np_array(point, self.radius)
        w, h = patch.shape
        feat = patch.flatten() - np.mean(patch)

        return feat