from unittest import TestCase

from shapely.geometry import Point, Polygon

import topo2vec.modules.topography_profiler as tp


class TestGet_features(TestCase):
    @classmethod
    def setUpClass(cls):
        # actually checks almost everything...
        cls.points = [Point(9.5658811, 47.0851164),
                      Point(9.658811, 47.01164)]

    def test_get_features(self):
        lis = tp.get_features(self.points)
        self.assertTupleEqual(lis.shape, (2, tp.FINAL_HPARAMS.latent_space_size))

    def test_get_all_class_points_in_polygon(self):
        polygon_to_search_in = Polygon([Point(5, 45), Point(5, 45.1), Point(5.1, 45.1),
                                        Point(5.1, 45.1), Point(5, 45)])

        points, lis = tp.get_all_class_points_in_polygon(polygon_to_search_in, 100, 'peaks')
        self.assertTupleEqual(lis.shape, (len(points), 3, 17, 17))
