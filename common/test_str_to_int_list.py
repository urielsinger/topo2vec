from unittest import TestCase

from common.list_conversions_utils import str_to_int_list


class TestStr_to_int_list(TestCase):
    def test_str_to_int_list(self):
        lis = '[[1,2],[2,3]]'
        self.assertEqual(str_to_int_list(lis), [[1, 2], [2, 3]])
        lis = [[1,2],[2,3]]
        self.assertEqual(str_to_int_list(lis), [[1, 2], [2, 3]])
        lis = '[1,2,3]'
        self.assertEqual(str_to_int_list(lis), [1, 2, 3])
        lis = '[[1,2,3]]'
        self.assertEqual(str_to_int_list(lis), [[1, 2, 3]])

