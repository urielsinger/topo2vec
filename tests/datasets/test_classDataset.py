from unittest import TestCase

from shapely.geometry import Point, Polygon

from topo2vec.constants import N49_E05_RIVERS
from topo2vec.common.geographic.geo_utils import check_if_point_in_polygon
from topo2vec.datasets.class_dataset import ClassDataset
from topo2vec.modules.topography_profiler import build_polygon


class TestClassDataset(TestCase):
    def test_init(self):
        outer_polygon = build_polygon(5, 6, 49.5, 50)
        Radius = 10
        classification_dataset = ClassDataset(N49_E05_RIVERS, 1, wanted_size=11,
                                              radii=[Radius],
                                              outer_polygon=outer_polygon)
        self.assertEqual(classification_dataset.actual_patches.shape[2], 2 * Radius + 1)
        self.assertTrue(check_if_point_in_polygon(classification_dataset.points_locations[0],
                                                  outer_polygon))
        self.assertEqual(len(classification_dataset), len(classification_dataset.points_locations))
