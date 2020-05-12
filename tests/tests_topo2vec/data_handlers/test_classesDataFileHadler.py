from unittest import TestCase

from topo2vec.constants import N45_50_E5_15_PEAKS, TRAIN_HALF_SMALL
from topo2vec.data_handlers.classes_data_file_handler import ClassesDataFileHadler


class TestClassesDataFileHadler(TestCase):
    @classmethod
    def setUpClass(cls):
        #actually checks almost everything...
        cls.class_from_file_handler = ClassesDataFileHadler(N45_50_E5_15_PEAKS)

    def test_get_random_subset_in_polygon(self):
        self.class_from_file_handler.get_random_subset_in_polygon(10, TRAIN_HALF_SMALL)
