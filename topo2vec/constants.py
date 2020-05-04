from pathlib import Path

#when on server itself
BASE_LOCATION = '/home/topo2vec_kavitzky/topo2vec/'
#when inside the docker
#BASE_LOCATION = '/home/root/'

# run in background - the service for getting visualizations of lon, lats
ELEVATION_BASE_DIR1 = BASE_LOCATION + 'data/elevation/big_europe'

MASK_BASE_DIR = BASE_LOCATION + 'data/elevation/big_europe'
equatorial_circumference_of_earth = 40075016.686 #m
from shapely.geometry import Polygon, Point

N49_E05_STREAMS = BASE_LOCATION + 'data/N049E005/classes/streams.geojson'
N49_E05_CLIFFS = BASE_LOCATION + 'data/N049E005/classes/cliffs.geojson'
N49_E05_PEAKS = BASE_LOCATION + 'data/N049E005/classes/rivers.geojson'
N49_E05_RIVERS = BASE_LOCATION + 'data/N049E005/classes/peaks.geojson'

N45_50_E5_15_CLIFFS = BASE_LOCATION + 'data/overpass_classes_data/cliffs_(45,5,50,15).json'
N45_50_E5_15_PEAKS = BASE_LOCATION + 'data/overpass_classes_data/peaks_(45,5,50,15).json'
N45_50_E5_15_RIVERS = BASE_LOCATION + 'data/overpass_classes_data/rivers_(45,5,50,15).json'
N45_50_E5_15_STREAMS = BASE_LOCATION + 'data/overpass_classes_data/streams_(45,5,50,15).json'

LOGS_PATH = BASE_LOCATION + 'tb_logs/logs'
MULTICLASS_LOGS_PATH = BASE_LOCATION + 'tb_logs/multiclass'
AUTOENCODER_LOGS_PATH = BASE_LOCATION + 'tb_logs/autoencoder'
CLIFFS_TEST = BASE_LOCATION + 'data/overpass_classes_data/tests/cliffs_test_(45,10,50,15).json'
PEAKS_TEST = BASE_LOCATION + 'data/overpass_classes_data/tests/peaks_test_(45,10,50,15).json'
RIVERS_TEST = BASE_LOCATION + 'data/overpass_classes_data/tests/rivers_test_(45,10,50,15).json'
STREAMS_TEST = BASE_LOCATION + 'data/overpass_classes_data/tests/streams_test_(45,10,50,15).json'

POINT_TO_SEARCH_SIMILAR_LARGE = BASE_LOCATION + 'data/overpass_classes_data/points_search_similar/points_to_search_similar_(45,10,50,15).json'
POINT_TO_SEARCH_SIMILAR_SMALL = BASE_LOCATION + 'data/overpass_classes_data/points_search_similar/points_to_search_similar_(49,5,50,59).json'
GROUP_TO_SEARCH_SIMILAR_LONGS_LARGE = BASE_LOCATION + 'data/overpass_classes_data/points_search_similar/group_to_search_similar_longs(45,10,50,15).json'

SAVE_PATH = BASE_LOCATION + 'data/pretrained_models/basicconvnetlatent'

Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)

VALIDATION_HALF_LARGE = Polygon([Point(5, 45), Point(5, 50), Point(10, 50),
                                 Point(10, 45), Point(5, 45)])

TRAIN_HALF_LARGE = Polygon([Point(10, 50), Point(10, 45), Point(15, 45),
                            Point(15, 50), Point(10, 50)])

VALIDATION_HALF_SMALL = Polygon([Point(5, 49), Point(5, 50), Point(5.9, 50),
                                 Point(5.9, 49), Point(5, 49)])

TRAIN_HALF_SMALL = Polygon([Point(5.9, 50), Point(5.9, 49), Point(6, 49),
                            Point(6, 50), Point(5.9, 50)])

FINAL_MODEL_DIR = BASE_LOCATION + 'data/final_model'
Path(FINAL_MODEL_DIR).mkdir(parents=True, exist_ok=True)

CLASSES_TEST_POINTS_FOLDER = BASE_LOCATION + 'data/overpass_classes_data/tests'

CACHE_BASE_DIR = BASE_LOCATION + 'data/cache'

NONE_STR = 'None'

#server routes
GET_CLASS_POINTS_ROUTE = '/get_class'
GET_SIMILAR_POINTS_ROUTE = '/get_similar'