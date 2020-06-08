from unittest import TestCase

from shapely.geometry import Point

from api_client import client_lib
from api_client.client_lib import get_latent_for_points, build_polygon


class TestGet_latent_for_points(TestCase):
    def test_get_latent_for_points(self):
        #should pass only if the flask server is up!
        small_polygon = build_polygon(35.3, 33.11, 35.35, 33.15)
        client_lib.set_working_polygon(small_polygon)
        points_list = [Point(35.32, 33.13), Point(35.31, 33.12)]
        patches, points = get_latent_for_points(points_list, 8)
        self.assertEqual(len(points), len(patches))