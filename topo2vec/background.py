from topo2vec.constants import N45_50_E5_15_CLIFFS, N45_50_E5_15_RIVERS, N45_50_E5_15_PEAKS, N45_50_E5_15_STREAMS
from topo2vec.data_handlers.classes_data_file_handler import ClassesDataFileHadler
from topo2vec.data_handlers.data_from_file_handler import DataFromFileHandler

# run in background - the service for getting visualizations of lon, lats
ELEVATION_BASE_DIR = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/data/elevation'
MASK_BASE_DIR = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/data/elevation'

data_visualizer = DataFromFileHandler(ELEVATION_BASE_DIR, (5, 45, 15, 50))
visualizer = data_visualizer

class_paths = [N45_50_E5_15_CLIFFS, N45_50_E5_15_RIVERS, N45_50_E5_15_PEAKS]#], N45_50_E5_15_STREAMS]
classes_data_handlers = {}

for class_path in class_paths:
    classes_data_handlers[class_path] = ClassesDataFileHadler(class_path)

mask_visualizer = DataFromFileHandler(MASK_BASE_DIR)
