from topo2vec.constants import BASE_LOCATION
from topo2vec.data_handlers.data_from_file_handler import DataFromFileHandler

###########################################################################
# run in background - the service for getting visualizations of lon, lats #
###########################################################################
ELEVATION_BASE_DIR1 = BASE_LOCATION + 'data/elevation/big_europe'
ELEVATION_BASE_DIR2 = BASE_LOCATION + 'data/elevation/mid_east'
MASK_BASE_DIR = BASE_LOCATION + 'data/elevation/45,5,50,15'
boxes = [(5, 45, 15, 50), (33, 30, 37, 32), (35, 31, 36, 35)]

base_dirs = [ELEVATION_BASE_DIR1, ELEVATION_BASE_DIR2, ELEVATION_BASE_DIR2, ELEVATION_BASE_DIR2]
data_visualizer = DataFromFileHandler(base_dirs, boxes)
visualizer = data_visualizer