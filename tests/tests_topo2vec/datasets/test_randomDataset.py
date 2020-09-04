from unittest import TestCase

from shapely.geometry import Point, Polygon

from topo2vec.datasets.random_dataset import RandomDataset


class TestRandomDataset(TestCase):
    def test_init(self):
        size = 10
        outer_polygon = Polygon([Point(5, 49.5), Point(5, 50), Point(6, 50),
                                 Point(6, 49.5), Point(5, 49.5)])
        random_data_set = RandomDataset(size, [[10]], outer_polygon)
        self.assertEqual(random_data_set.actual_patches.shape[0], size)
