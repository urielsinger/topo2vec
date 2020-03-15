from topo2vec.common.geographic import geo_utils


class DatasetBuilder:
    '''
    build a dataset in a certain area
    '''
    def __init__(self, min_lon: float, min_lat: float, max_lon: float, max_lat: float):
        '''

        Args: The corners to go around:
            min_lon:
            min_lat:
            max_lon:
            max_lat:
        '''
        self.min_lon = min_lon
        self.min_lat = min_lat
        self.max_lon = max_lon
        self.max_lat = max_lat

    def get_grid_dataset(self, step_size = 0.01):
        points_list = geo_utils.sample_grid_in_range(self.min_lon, self.min_lat, self.max_lon, self.max_lat, step_size)
        return points_list

    def get_random_dataset(self, num_samples:int = 10000):
        points_list = geo_utils.sample_points_in_range(self.min_lon, self.min_lat,
                                                       self.max_lon, self.max_lat, num_samples)
        return points_list

