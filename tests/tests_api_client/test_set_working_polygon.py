from unittest import TestCase

from api_client import client_lib
from api_client.client_lib import build_polygon
from topo2vec.constants import north_is_small


class TestSet_working_polygon(TestCase):
    def test_set_working_polygon(self):
        client_lib.set_working_polygon(north_is_small)
        work_poly = client_lib.get_working_polygon()
        self.assertTrue(work_poly.equals(north_is_small))