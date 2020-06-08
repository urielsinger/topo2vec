import logging
from unittest import TestCase

from geopy import Point

from api_client.client_lib import get_all_class_points_in_polygon, build_polygon
from api_client import client_lib

class TestGet_class_points(TestCase):
    def test_get_class_points(self):
        #should pass only if the flask server is up!
        small_polygon = build_polygon(35.3, 33.11, 35.35, 33.15)
        client_lib.set_working_polygon(small_polygon)
        points_list = [Point(35.32, 33.13), Point(35.31, 33.12)]
        points, patches = get_all_class_points_in_polygon(small_polygon, 2000, 'peaks', 0)
        logging.info(points.shape)