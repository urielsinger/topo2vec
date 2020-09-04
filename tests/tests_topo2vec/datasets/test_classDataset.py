import logging
from unittest import TestCase

from shapely.geometry import Point, Polygon

from topo2vec.constants import N45_50_E5_15_PEAKS
from common.geographic.geo_utils import check_if_point_in_polygon
from topo2vec.datasets.class_dataset import ClassDataset


class TestClassDataset(TestCase):
    def test_init(self):
        outer_polygon = Polygon([Point(5, 49.5), Point(5, 50), Point(6, 50),
                                 Point(6, 49.5), Point(5, 49.5)])
        Radius = 10
        classification_dataset = ClassDataset(class_path=N45_50_E5_15_PEAKS, class_label=1,
                                              wanted_size=11,
                                              original_radiis=[[Radius]],
                                              outer_polygon=outer_polygon)
        self.assertEqual(classification_dataset.actual_patches.shape[2], 2 * Radius + 1)
        self.assertTrue(check_if_point_in_polygon(classification_dataset.points_locations[0],
                                                  outer_polygon))
        self.assertEqual(len(classification_dataset), len(classification_dataset.points_locations))

        Radius = 10
        classification_dataset = ClassDataset(class_path=N45_50_E5_15_PEAKS, class_label=1,
                                              wanted_size=11,
                                              original_radiis=[[Radius]],
                                              outer_polygon=outer_polygon,
                                              radii=[Radius])
        logging.info(classification_dataset.actual_patches.shape)
        self.assertEqual(classification_dataset.actual_patches.shape[2], 2 * Radius + 1)
        self.assertTrue(check_if_point_in_polygon(classification_dataset.points_locations[0],
                                                  outer_polygon))
        self.assertEqual(len(classification_dataset), len(classification_dataset.points_locations))
