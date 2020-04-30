
from unittest import TestCase

from shapely.geometry import Point, Polygon

import topo2vec.modules.topography_profiler as tp

class Test(TestCase):
    @classmethod
    def setUpClass(cls):
        # actually checks almost everything...
        cls.points = [Point(9.5658811, 47.0851164),
                      Point(9.658811, 47.01164)]
        cls.points_inside = [Point(5.0658811, 45.0851164),
                      Point(5.058811, 45.01164)]
        cls.polygon_to_search_in = Polygon([Point(5, 45), Point(5, 45.1), Point(5.1, 45.1),
                                        Point(5.1, 45.1), Point(5, 45)])

    def test_get_features(self):
        lis = tp.get_features(self.points)
        self.assertTupleEqual(lis.shape, (2, tp.FINAL_HPARAMS.latent_space_size))

    def test_get_all_class_points_in_polygon(self):
        points, pathches_lis = tp.get_all_class_points_in_polygon(self.polygon_to_search_in, 100, 'peaks')
        self.assertTupleEqual(pathches_lis.shape, (len(points), 3, 17, 17))

    def test_get_top_n_similar_points_in_polygon(self):
        list_of_patches, list_of_points = tp.get_top_n_similar_points_in_polygon(self.points_inside,
                                                                                 10, self.polygon_to_search_in, 100)
        self.assertEqual(len(list_of_points), 10)
        self.assertTupleEqual(list_of_patches.shape, (10, 3, 17, 17))

