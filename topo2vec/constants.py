from shapely.geometry import Polygon, Point

N49_E05_STREAMS = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/data/N049E005/classes/streams.geojson'
N49_E05_CLIFFS = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/data/N049E005/classes/cliffs.geojson'
N49_E05_PEAKS = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/data/N049E005/classes/rivers.geojson'
N49_E05_RIVERS = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/data/N049E005/classes/peaks.geojson'

N45_50_E5_15_CLIFFS = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/data/overpass_classes_data/cliffs_(45,5,50,15).json'
N45_50_E5_15_PEAKS = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/data/overpass_classes_data/peaks_(45,5,50,15).json'
N45_50_E5_15_RIVERS = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/data/overpass_classes_data/rivers_(45,5,50,15).json'
N45_50_E5_15_STREAMS = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/data/overpass_classes_data/streams_(45,5,50,15).json'

LOGS_PATH = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/tb_logs/logs'
MULTICLASS_LOGS_PATH = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/tb_logs/multiclass'
AUTOENCODER_LOGS_PATH = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/tb_logs/autoencoder'
CLIFFS_TEST = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/data/overpass_classes_data/tests/cliffs_test_(45,10,50,15).json'
PEAKS_TEST = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/data/overpass_classes_data/tests/peaks_test_(45,10,50,15).json'
RIVERS_TEST = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/data/overpass_classes_data/tests/rivers_test_(45,10,50,15).json'
STREAMS_TEST = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/data/overpass_classes_data/tests/streams_test_(45,10,50,15).json'

POINT_TO_SEARCH_SIMILAR_LARGE = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/data/overpass_classes_data/points_search_similar/points_to_search_similar_(45,10,50,15).json'
POINT_TO_SEARCH_SIMILAR_SMALL = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/data/overpass_classes_data/points_search_similar/points_to_search_similar_(49,5,50,59).json'
GROUP_TO_SEARCH_SIMILAR_LONGS_LARGE = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/data/overpass_classes_data/points_search_similar/group_to_search_similar_longs(45,10,50,15).json'

SAVE_PATH = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/pretrained_models/basicconvnetlatent'

VALIDATION_HALF_LARGE = Polygon([Point(5, 45), Point(5, 50), Point(10, 50),
                                 Point(10, 45), Point(5, 45)])

TRAIN_HALF_LARGE = Polygon([Point(10, 50), Point(10, 45), Point(15, 45),
                            Point(15, 50), Point(10, 50)])

VALIDATION_HALF_SMALL = Polygon([Point(5, 49), Point(5, 50), Point(5.9, 50),
                                 Point(5.9, 49), Point(5, 49)])

TRAIN_HALF_SMALL = Polygon([Point(5.9, 50), Point(5.9, 49), Point(6, 49),
                            Point(6, 50), Point(5.9, 50)])

FINAL_MODEL_DIR = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/data/final_model'


CLASSES_TEST_POINTS_FOLDER = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/data/overpass_classes_data/tests'

CACHE_BASE_DIR = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/data/cache'

NONE_STR = 'None'