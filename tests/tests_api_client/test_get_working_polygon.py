import logging
from unittest import TestCase

from shapely.geometry import Polygon

from api_client.client_lib import get_working_polygon


class TestGet_working_polygon(TestCase):
    def test_get_working_polygon(self):
        poly = get_working_polygon()
        logging.info(type(poly))
        self.assertTrue(type(poly) is Polygon)