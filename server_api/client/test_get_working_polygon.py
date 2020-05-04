from unittest import TestCase

from shapely.geometry import Polygon

from server_api.client.client_lib import get_working_polygon


class TestGet_working_polygon(TestCase):
    def test_get_working_polygon(self):
        poly = get_working_polygon()
        self.assertTrue(type(poly) is Polygon)