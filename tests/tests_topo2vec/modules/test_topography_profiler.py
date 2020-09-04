import logging
from unittest import TestCase

from shapely.geometry import Point

import common.geographic.geo_utils
import topo2vec.topography_profiler as tp

class Test(TestCase):
    @classmethod
    def setUpClass(cls):
        # actually checks almost everything...
        cls.points_inside = [Point(5.0658811, 45.0851164),
                      Point(5.058811, 45.01164)]
        cls.polygon_to_search_in = common.geographic.geo_utils.build_polygon(5, 45, 5.1, 45.1)
        tp.set_working_polygon(cls.polygon_to_search_in)

    def test_get_features(self):
        lis = tp.get_features(self.points_inside, 8)
        self.assertTupleEqual(lis.shape, (2, tp.FINAL_HPARAMS.latent_space_size))

    def test_get_all_class_points_in_polygon(self):
        final_radii = tp.FINAL_ORIGINAL_RADIIS
        points, pathches_lis = tp.get_all_class_points_in_polygon(self.polygon_to_search_in, 100, 'peaks', 8)
        logging.info(len(points))
        self.assertTupleEqual(pathches_lis.shape, (len(points), 3, 17, 17))

        points, pathches_lis = tp.get_all_class_points_in_polygon(self.polygon_to_search_in, 100, 'peaks', 8, prob_threshold=0.9)
        logging.info(len(points))
        self.assertTupleEqual(pathches_lis.shape, (len(points), 3, 17, 17))

    def test_get_top_n_similar_points_in_polygon(self):
        list_of_patches, list_of_points = tp.get_top_n_similar_points_in_polygon(self.points_inside,
                                                                                 10, self.polygon_to_search_in, 100, 8)
        self.assertEqual(len(list_of_points), 10)
        self.assertTupleEqual(list_of_patches.shape, (10, 3, 17, 17))

