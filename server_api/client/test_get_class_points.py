from unittest import TestCase

from shapely.geometry import Point

from server_api.client import client_lib
from server_api.client.client_lib import get_class_points


class TestGet_class_points(TestCase):
    def test_get_class_points(self):
        WORKING_POLYGON = client_lib.build_polygon(34.7, 31.3, 34.9, 31.43)
        points_list = [Point(34.75, 31.35), Point(34.76, 31.36)]

        patches, points = get_class_points(WORKING_POLYGON, 500, 'peaks')
        print(points)
