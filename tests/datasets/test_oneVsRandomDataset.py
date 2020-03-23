from unittest import TestCase

from shapely.geometry import Polygon, Point

from topo2vec.datasets.one_vs_random_dataset import OneVsRandomDataset


class TestOneVsRandomDataset(TestCase):
    def test_init(self):
        outer_polygon = Polygon([Point(5, 49.5), Point(5, 50), Point(6, 50),
                                 Point(6, 49.5), Point(5, 49.5)])
        dataset = OneVsRandomDataset([10],11,outer_polygon)