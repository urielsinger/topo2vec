from unittest import TestCase

from shapely.geometry import Point

from api_client import client_lib
from api_client.client_lib import get_top_n_similar_points_in_polygon, build_polygon


class TestGet_top_n_similar_points_in_polygon(TestCase):
    def test_get_top_n_similar_points_in_polygon(self):
        #should pass only if the flask server is up!
        small_polygon = build_polygon(35.3, 33.11, 35.35, 33.15)
        client_lib.set_working_polygon(small_polygon)
        points_list = [Point(35.32, 33.13), Point(35.31, 33.12)]

        patches, points = get_top_n_similar_points_in_polygon(points_list, 10, small_polygon, 500, 8)
        self.assertEqual(len(patches),len(points))
