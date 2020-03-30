from topo2vec.constants import N45_50_E5_15_CLIFFS, N45_50_E5_15_RIVERS, N45_50_E5_15_PEAKS, N45_50_E5_15_STREAMS, \
    N49_E05_CLIFFS, N49_E05_RIVERS, N49_E05_PEAKS, N49_E05_STREAMS, TRAIN_HALF_LARGE, VALIDATION_HALF_LARGE, \
    VALIDATION_HALF_SMALL, TRAIN_HALF_SMALL, POINT_TO_SEARCH_SIMILAR_LARGE, POINT_TO_SEARCH_SIMILAR_SMALL
from topo2vec.data_handlers.classes_data_file_handler import ClassesDataFileHadler
from topo2vec.data_handlers.data_from_file_handler import DataFromFileHandler

# run in background - the service for getting visualizations of lon, lats
ELEVATION_BASE_DIR = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/data/elevation'
MASK_BASE_DIR = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/data/elevation'

data_visualizer = DataFromFileHandler(ELEVATION_BASE_DIR, (5, 45, 15, 50))
visualizer = data_visualizer
mask_visualizer = DataFromFileHandler(MASK_BASE_DIR)

# True to use the large area in europe data
# False for the certain place:
LOAD_CLASSES_LARGE = True

classes_data_handlers = {}

if LOAD_CLASSES_LARGE:
    # use data of the large area (5,45,15,50)
    class_paths = [N45_50_E5_15_CLIFFS, N45_50_E5_15_RIVERS, N45_50_E5_15_PEAKS]#, N45_50_E5_15_STREAMS]

    for class_path in class_paths:
        classes_data_handlers[class_path] = ClassesDataFileHadler(class_path)

    class_names = ['Cliff', 'River', 'Peak']#, 'Stream']
    TRAIN_HALF = TRAIN_HALF_LARGE
    VALIDATION_HALF = VALIDATION_HALF_LARGE
    POINT_TO_SEARCH_SIMILAR = POINT_TO_SEARCH_SIMILAR_LARGE

else:
    # use data of only the small area (5,49,6,50)
    class_paths = [N49_E05_CLIFFS, N49_E05_RIVERS, N49_E05_PEAKS, N49_E05_STREAMS]

    for class_path in class_paths:
        classes_data_handlers[class_path] = ClassesDataFileHadler(class_path)

    class_names = ['Cliff', 'River', 'Peak', 'Stream']
    TRAIN_HALF = TRAIN_HALF_SMALL
    VALIDATION_HALF = VALIDATION_HALF_SMALL
    POINT_TO_SEARCH_SIMILAR = POINT_TO_SEARCH_SIMILAR_SMALL
