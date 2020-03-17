from topo2vec.data_handlers.data_from_file_handler import DataFromFileHandler


#run in background - the service for getting visualizations of lon, lats
DATA_49_05 ='/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/data/N049E005/N049E005_AVE_DSM.tif'
MASK_49_05 ='/home/morpheus/topo2vec_kavitzky/repositories/topo2vec/data/N049E005/N049E005_AVE_MSK.tif'

data_visualizer = DataFromFileHandler(DATA_49_05)
visualizer = data_visualizer

mask_visualizer = DataFromFileHandler(MASK_49_05)
