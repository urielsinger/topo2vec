from unittest import TestCase

from topo2vec.tasks.nearest_neighbour.nn_task import NNTask


class TestNNTask(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        RADIUS = 20
        cls.nn_task = NNTask(RADIUS, 1000, 5)

    def test_run(self):
        self.nn_task.run()
        self.assertEqual(1,1)
