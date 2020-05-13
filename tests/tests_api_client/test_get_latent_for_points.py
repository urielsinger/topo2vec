from unittest import TestCase

from shapely.geometry import Point

from api_client.client_lib import get_latent_for_points


class TestGet_latent_for_points(TestCase):
    def test_get_latent_for_points(self):
        #should pass only if the flask server is up!

        points_list = [Point(34.75, 31.35), Point(34.76, 31.36)]
        patches, points = get_latent_for_points(points_list)