import geopandas as gpd
from logging import warning


class GeoGraphBuilder:
    """
    A class to convert series of geometries to a graph
    """

    def __init__(self, method=None):
        self._default_method_name = "RNG"
        self.method = self._default_method_name if method is None else method

    def create_graph_from_geoseries(self, series: gpd.GeoSeries):
        if self.method == "RNG":
            return self.create_RNG_from_geoseries(series)
        else:
            warning(f"unsupported method name for GeoGraphBuilder using {self._default_method_name}")

    def create_RNG_from_geoseries(self, series):
        pass
