from unittest import TestCase

from common.list_conversions_utils import points_list_to_floats_list, floats_list_to_points_list


class TestFloats_list_to_points_list(TestCase):
    def test_floats_list_to_points_list(self):
        floats_list = [1.0, 2.0, 3.0, 4.0]
        points_list = floats_list_to_points_list(floats_list)
        self.assertSequenceEqual(floats_list, points_list_to_floats_list(points_list))
