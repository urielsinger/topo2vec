from unittest import TestCase

from shapely.geometry import Point

from topo2vec.data_handlers.data_from_file_handler import DataFromFileHandler

ELEVATION_BASE_DIR = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/data/elevation'

class TestDataFromFileHandler(TestCase):
    @classmethod
    def setUpClass(cls):
        #actually checks almost everything...
        cls.data_from_file_handler = DataFromFileHandler(ELEVATION_BASE_DIR, (5,45,7,48))

    def test_point_to_string(self):
        point_as_string = self.data_from_file_handler.point_to_string(Point(40.56, 5.34))
        self.assertEqual(point_as_string, 'N005E040')

    def test_get_points_as_np_array(self):
        test_points = [Point(5.55, 46.2), Point(6.55, 47.2), Point(6.55, 47.2)]
        radii = [10,20]
        patches, points = self.data_from_file_handler.get_points_as_np_array(test_points, radii)
        self.assertTupleEqual(patches.shape, (len(test_points), len(radii), 21, 21))
        self.assertEqual(len(test_points), len(points))
