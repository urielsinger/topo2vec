from unittest import TestCase

from topo2vec.datasets.random_dataset import RandomDataset


class TestRandomDataset(TestCase):
    def test_init(self):
        size = 10
        random_sata_set = RandomDataset(size)
        self.assertEqual(random_sata_set.actual_patches.shape[0], size)
