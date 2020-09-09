from topo2vec.data_handlers.classes_data_file_handler import ClassesDataFileHadler

peaks = ClassesDataFileHadler('/home/root/data/overpass_classes_data/peaks_(45,5,50,15).json', cache_load=False)

import matplotlib.pyplot as plt
plt.hist(peaks.elevations_list)
plt.show()