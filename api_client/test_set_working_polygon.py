from unittest import TestCase

from api_client import client_lib
from api_client.client_lib import build_polygon

class TestSet_working_polygon(TestCase):
    def test_set_working_polygon(self):
        leban = build_polygon(33.11, 35.19, 33.25, 35.47)
        client_lib.set_working_polygon(leban)
        work_poly = client_lib.get_working_polygon()
        self.assertTrue(work_poly.equals(leban))