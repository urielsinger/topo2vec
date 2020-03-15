from abc import ABC

from shapely.geometry import Point


class DataHandler(ABC):
    '''
    An API for data handling.
    '''
    def __init__(self):
        pass
    def get_data_as_np_array(self,  lon: float, lat: float, r: int):
        '''
        get data around a certain point, in r distance to each direction
        Args:
            lon:
            lat:
            r: the distance around the center point, in pixels

        Returns:

        '''
        pass

    def get_data_as_np_array(self, center: Point, r:int):
        '''
        get data around the center point, in r distance to each direction
        Args:
             center:
            r:

        Returns:

        '''
        pass