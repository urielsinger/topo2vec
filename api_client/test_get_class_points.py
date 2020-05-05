from unittest import TestCase

from shapely.geometry import Point

from api_client.client_lib import get_all_class_points_in_polygon
from api_client import client_lib

class TestGet_class_points(TestCase):
    def test_get_class_points(self):
        WORKING_POLYGON = client_lib.build_polygon(34.7, 31.3, 34.9, 31.43)
        patches, points = get_all_class_points_in_polygon(WORKING_POLYGON, 500, 'peaks')
        print(points)
