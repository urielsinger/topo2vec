import os
import pickle

from topo2vec.constants import SCALES_DICT_DIR


class ScalesDict:
    def __init__(self, file_path=SCALES_DICT_DIR, file_name='ordinary'):
        self.dict = {}
        self.file_full_path = os.path.join(file_path, file_name + '.pickle')

    def add_class_size(self, class_name, class_eigen_scale):
        self.dict[class_name] = class_eigen_scale

    def init_from_dict(self, dict):
        self.dict = dict

    def load_from_file(self):
        with open(self.file_full_path, 'r') as f:
            self.dict = pickle.load(self.dict, f)
        print('succeeded to load classes scale')

    def save_to_file(self):
        with open(self.file_full_path, 'wb+') as f:
            pickle.dump(self.dict, f)

