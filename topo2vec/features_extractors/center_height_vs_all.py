from topo2vec.features_extractors.feature_extractor import FeatureExtractor
from topo2vec import visualizer
import numpy as np

class CenterHeightVsAll(FeatureExtractor):
    def __init__(self, r):
        '''

        Args:
            r: the radius of the patch to calculate the feature around
        '''
        self.r = r

    def transform(self, points:list):
        '''
        get center point's height vs all
        Args:
            points: list of points to calculate the feature aroiund

        Returns: a (num of points, 1) numpy array consisting of the data

        '''
        lis = []
        for point in points:
            patch = visualizer.get_data_as_np_array(point, self.r)
            n, m = patch.shape
            center_x = int(n / 2)
            center_y = int(m / 2)
            lis.append(np.array([[patch[center_x, center_y]]]) / np.mean(patch))

        return np.concatenate(lis, axis = 0)
