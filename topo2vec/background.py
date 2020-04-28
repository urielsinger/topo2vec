from topo2vec.data_handlers.classes_data_file_handler import ClassesDataFileHadler
from topo2vec.data_handlers.data_from_file_handler import DataFromFileHandler
from topo2vec.constants import N45_50_E5_15_CLIFFS, N45_50_E5_15_RIVERS, N45_50_E5_15_PEAKS, N45_50_E5_15_STREAMS, \
    N49_E05_CLIFFS, N49_E05_RIVERS, N49_E05_PEAKS, N49_E05_STREAMS, TRAIN_HALF_LARGE, VALIDATION_HALF_LARGE, \
    VALIDATION_HALF_SMALL, TRAIN_HALF_SMALL, POINT_TO_SEARCH_SIMILAR_LARGE, POINT_TO_SEARCH_SIMILAR_SMALL, \
    CLASSES_TEST_POINTS_FOLDER, BASE_LOCATION

from pathlib import Path

# run in background - the service for getting visualizations of lon, lats
ELEVATION_BASE_DIR1 = BASE_LOCATION + 'data/elevation/big_europe'
ELEVATION_BASE_DIR2 = BASE_LOCATION + 'data/elevation'

MASK_BASE_DIR = BASE_LOCATION + 'data/elevation/45,5,50,15'
data_visualizer = DataFromFileHandler([ELEVATION_BASE_DIR1], [(5, 45, 15, 50)])
visualizer = data_visualizer
#mask_visualizer = DataFromFileHandler([MASK_BASE_DIR])

# True to use the large area in europe data
# False for the certain place:
LOAD_CLASSES_LARGE = True

classes_data_handlers = {}

if LOAD_CLASSES_LARGE:
    # use data of the large area (5,45,15,50)
    from os import listdir
    from os.path import isfile, join

    CLASSES_POINTS_FOLDER = BASE_LOCATION + 'data/overpass_classes_data'
    Path(CLASSES_POINTS_FOLDER).mkdir(parents=True, exist_ok=True)

    ## assuming the classes in data
    # and the classes in test are alphabetically in the same order!!!
    CLASS_PATHS = [str(join(CLASSES_POINTS_FOLDER, f)) for f
                   in listdir(CLASSES_POINTS_FOLDER)
                   if isfile(join(CLASSES_POINTS_FOLDER, f))]

    CLASS_NAMES = [str(f.split('_')[0]) for f
                   in listdir(CLASSES_POINTS_FOLDER)
                   if isfile(join(CLASSES_POINTS_FOLDER, f))]

    CLASS_PATHS_TEST = [str(join(CLASSES_TEST_POINTS_FOLDER, f)) for f
                        in listdir(CLASSES_TEST_POINTS_FOLDER)
                        if isfile(join(CLASSES_TEST_POINTS_FOLDER, f))]

    CLASS_NAMES_TEST = [str(f.split('_')[0]) for f
                        in listdir(CLASSES_TEST_POINTS_FOLDER)
                        if isfile(join(CLASSES_TEST_POINTS_FOLDER, f))]

    SPECIAL_CLASSES_POINTS_FOLDER = BASE_LOCATION + 'data/overpass_classes_data/for_eval'
    Path(SPECIAL_CLASSES_POINTS_FOLDER).mkdir(parents=True, exist_ok=True)

    CLASS_PATHS_SPECIAL = [str(join(SPECIAL_CLASSES_POINTS_FOLDER, f)) for f
                           in listdir(SPECIAL_CLASSES_POINTS_FOLDER)
                           if isfile(join(SPECIAL_CLASSES_POINTS_FOLDER, f))]


    CLASS_NAMES_SPECIAL = ['_'.join(f.split('_')[:-1]) for f
                           in listdir(SPECIAL_CLASSES_POINTS_FOLDER)
                           if isfile(join(SPECIAL_CLASSES_POINTS_FOLDER, f))]

    CLASS_NAMES.sort()
    CLASS_PATHS.sort()
    CLASS_NAMES_TEST.sort()
    print(CLASS_NAMES)
    CLASS_PATHS_TEST.sort()
    CLASS_PATHS_SPECIAL.sort()
    CLASS_NAMES_SPECIAL.sort()
    print(CLASS_NAMES_SPECIAL)

    TRAIN_HALF = TRAIN_HALF_LARGE
    VALIDATION_HALF = VALIDATION_HALF_LARGE
    POINT_TO_SEARCH_SIMILAR = POINT_TO_SEARCH_SIMILAR_LARGE

else:
    # use data of only the small area (5,49,6,50)
    CLASS_PATHS = [N49_E05_CLIFFS, N49_E05_RIVERS, N49_E05_PEAKS, N49_E05_STREAMS]
    CLASS_NAMES = ['Cliff', 'River', 'Peak', 'Stream']
    CLASSES_POINTS_FOLDER = BASE_LOCATION + 'data/N049E005/classes'
    TRAIN_HALF = TRAIN_HALF_SMALL
    VALIDATION_HALF = VALIDATION_HALF_SMALL
    POINT_TO_SEARCH_SIMILAR = POINT_TO_SEARCH_SIMILAR_SMALL

BUILD_CLASSES_DATA_HANDLERS = True

if BUILD_CLASSES_DATA_HANDLERS:
    for class_path in CLASS_PATHS:
        classes_data_handlers[class_path] = ClassesDataFileHadler(class_path)
