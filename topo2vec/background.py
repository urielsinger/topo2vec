from topo2vec.data_handlers.data_from_file_handler import DataFromFileHandler

# run in background - the service for getting visualizations of lon, lats
ELEVATION_BASE_DIR = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/data/elevation'
MASK_BASE_DIR = '/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/data/elevation'

data_visualizer = DataFromFileHandler(ELEVATION_BASE_DIR, (5, 45, 15, 50))
visualizer = data_visualizer

mask_visualizer = DataFromFileHandler(MASK_BASE_DIR)
