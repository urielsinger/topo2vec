from unittest import TestCase

from shapely.geometry import Polygon, Point

from topo2vec.CONSTANTS import N49_E05_RIVERS
from topo2vec.datasets.several_classes_datasets import SeveralClassesDataset


class TestSeveralClassesDataset(TestCase):
    def test_init(self):
        outer_polygon = Polygon([Point(5, 49.5), Point(5, 50), Point(6, 50),
                                 Point(6, 49.5), Point(5, 49.5)])
        dataset = SeveralClassesDataset([10],outer_polygon, [N49_E05_RIVERS], [1])
