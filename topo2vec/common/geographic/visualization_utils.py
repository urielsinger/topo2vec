from datetime import datetime
from typing import List, Iterable, Union

from branca.colormap import linear
from branca.element import MacroElement
import folium
from folium.plugins import TimeSliderChoropleth, HeatMapWithTime
import pandas as pd
import geopandas as gpd
from jinja2 import Template
from shapely.geometry import Point
from shapely.wkt import loads
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from topo2vec.common.geographic.folium_extensions import TimeSliderPolygonHeatmap, TimeSliderDynamicMap
from topo2vec.common.geographic.geo_utils import meters2degrees

VIS_AGG_TIME = 'VIS_AGG_TIME'
VIS_GEO_SHP = 'VIS_GEO_SHP'
HOUR_AGG = 'hour'
DAY_AGG = 'day'
MINUTE_AGG = 'minute'


def _aggregate_datetime(df: pd.DataFrame, time_col: str, agg_time_col: str, agg_unit: str) -> pd.DataFrame:
    """
    This internal function is used to aggregate times into a specified time-unit.
    Args:
        df: The dataframe we want to aggregate
        time_col: The time column's name
        agg_time_col: The new aggregate time column's name
        agg_unit: The time unit for the aggregation

    Returns:
        $df with a new aggregated time column $VIS_AGG_TIME
    """
    if agg_unit is None:
        agg_unit = MINUTE_AGG

    if agg_unit == DAY_AGG:
        df[agg_time_col] = df[time_col].apply(
            lambda dt: datetime(dt.year, dt.month, dt.day))  # round to day
    elif agg_unit == HOUR_AGG:
        df[agg_time_col] = df[time_col].apply(
            lambda dt: datetime(dt.year, dt.month, dt.day, dt.hour))  # round to hour
    elif agg_unit == MINUTE_AGG:
        df[agg_time_col] = df[time_col].apply(
            lambda dt: datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute))  # round to minutes
    else:
        assert False, "agg_unit should be in [hour, minute, day]"

    return df


def heatmap_with_time(df: pd.DataFrame, geo_wkt_col: str, time_col: str, agg_unit: str = MINUTE_AGG) -> HeatMapWithTime:
    """
    A visualization utility function for visualizing a timed-geographic layer with a folium's HeatMapWithTime.
    The function aggregates the times correctly and inserts the parameters correctly into the object.
    Args:
        df: The dataframe we want to visualize
        geo_wkt_col: The geographic column's name, holding wkt objects
        time_col: The time column's name
        agg_unit: The time unit for the aggregation

    Returns:
        A HeatMapWithTime object to use with a folium map
    """
    df = df[[geo_wkt_col, time_col]].copy()
    df[VIS_GEO_SHP] = df[geo_wkt_col].apply(lambda x: loads(x))
    df = _aggregate_datetime(df, time_col, VIS_AGG_TIME, agg_unit)

    time_groupby_dict = df.groupby(VIS_AGG_TIME).groups
    time_index = sorted(list(time_groupby_dict.keys()))
    data = []
    for t in time_index:
        data.append(
            list(map(lambda shp: [shp.centroid.y, shp.centroid.x], df.iloc[time_groupby_dict[t]][VIS_AGG_TIME])))
    time_index_str = [dt.strftime('%Y-%m-%d  %H:%M') for dt in time_index]

    return HeatMapWithTime(data, index=time_index_str, auto_play=True, max_opacity=0.5)

def get_image_overlay(points: Iterable[Point], scores: np.array, step: float = 50,
                      name:str=None, return_array:bool=False) -> Union[folium.raster_layers.ImageOverlay, np.ndarray]:
    """
    Create a folium map of scores using only several known coords and their values (by interpolating).
    Args:
        points: Iterable of shaply Points
        scores: array of the coords values
        coord_range: range of the desired grid
        step: resolution of sample in meteres
        name: name for the image overlay
        return_array: if true, will return a numpy array, instead of folium overlay

    Returns:
        map_f: folium.Map - with a ImageOverlay of the scores in coord_range
    """
    # drop nans first
    not_nan_index = ~np.isnan(np.array(scores))
    scores, points = scores[not_nan_index], points[not_nan_index]

    step = meters2degrees(step)
    scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    lons = [point.coords[0][0] for point in points]
    lats = [point.coords[0][1] for point in points]
    min_lon, max_lon = np.min(lons), np.max(lons)
    min_lat, max_lat = np.min(lats), np.max(lats)
    lons_index = [int(round((lon - min_lon) / step)) for lon in lons]
    lats_index = [int(round((lat - min_lat) / step)) for lat in lats]

    cm = LinearSegmentedColormap.from_list('rgb', [(0, 1, 0, 0), (1, 1, 0, 0), (1, 0, 0, 0)], N=256, gamma=1.0)
    z = np.zeros(shape=(int((max_lon - min_lon) / step) + 1, int((max_lat - min_lat) / step) + 1))
    z[lons_index, lats_index] = scores
    if return_array:
        return z, (lons_index, lats_index)
    z = cm(z)
    z[lons_index, lats_index, -1] = 1
    image_overlay = folium.raster_layers.ImageOverlay(np.transpose(z, [1, 0, 2]),
                                                      bounds=[[min_lat, min_lon], [max_lat, max_lon]], origin='lower',
                                                      opacity=0.3, name=name)

    return image_overlay

class LatLngCopy(MacroElement):
    """
    When one clicks on a Map, it copies the latitude and longitude of the pointer to self.loc.
    """
    _template = Template(u"""
            {% macro script(this, kwargs) %}
                function latLngCopy(e) {
                    {{this._parent.get_name()}}.loc = [e.latlng.lat, e.latlng.lng]
                }
                {{this._parent.get_name()}}.on('click', latLngCopy);
                {{this._parent.get_name()}}.loc = [0, 0]
            {% endmacro %}
            """)  # noqa

    def __init__(self):
        super().__init__()
        self._name = 'LatLgCopy'
