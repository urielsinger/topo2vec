from unittest import TestCase

from shapely.geometry import Point

from api_client import client_lib
from api_client.client_lib import get_top_n_similar_points_in_polygon


class TestGet_top_n_similar_points_in_polygon(TestCase):
    def test_get_top_n_similar_points_in_polygon(self):
        WORKING_POLYGON = client_lib.build_polygon(34.7, 31.3, 34.9, 31.43)
        points_list = [Point(34.75, 31.35), Point(34.76, 31.36)]

        patches, points = get_top_n_similar_points_in_polygon(points_list, 10, WORKING_POLYGON, 500)
