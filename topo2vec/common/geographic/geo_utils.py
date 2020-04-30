import copy
import logging
from functools import partial
from typing import List, Union, Tuple, Iterable

import geopy
import geopy.distance
import numpy as np
import pyproj
import shapely
import shapely.ops as ops
from shapely import wkt
from shapely.geometry import Polygon, Point, LineString, MultiPolygon, MultiLineString
from shapely.geometry.base import BaseGeometry
from shapely.ops import cascaded_union
from tqdm import tqdm
import geopandas as gpd



def check_if_point_in_polygon(point:Point, polygon: Polygon) -> bool:
    '''
    Returns True iff the point is in range (roughly)
    Args:
        point: the point to check
        polygon: the polygon to check in

    Returns:

    '''
    return point.within(polygon)

def area_in_m2(poly: Polygon) -> float:
    """
    Returns the area of polygon in m^2
    Args:
        poly: The polygon in question

    Returns:
        It's area in m^2
    """
    try:
        poly_area = ops.transform(partial(pyproj.transform,
                                          pyproj.Proj('epsg:4326'),
                                          pyproj.Proj(proj='aea',
                                                      lat_1=poly.bounds[1],
                                                      lat_2=poly.bounds[3])),
                                  poly)
        return poly_area.area
    except Exception as e:
        logging.warning(e)
        logging.warning(f"Error with area_in_m2 - {poly.wkt}")
        return 0.


def distance_point_to_polygon(point: Point, poly: Polygon) -> float:
    """
    Calculates the distance in meters between point and polygon.
    First calculates the nearest point on the polygon, then calculates with geopy the distance between them in meters.
    Uses ellipsoid
    :param point: The point the Shapely.point
    :param poly: The polygon the Shapely.Polygon
    :return: The distance in meters
    """
    return distance_point_to_point(point, ops.nearest_points(poly, point)[0])


def distance_point_to_point(p1: Point, p2: Point) -> float:
    """
    Calculates the distance in meters between point and point.
    Uses geopy the distance between them in meters with ellipsoid
    :param p1: The point the Shapely.point
    :param p2: The second point the Shapely.point
    :return: The distance in meters
    """
    return geopy.distance.distance(list(p1.coords), list(p2.coords)).m


def calculate_polygons_coverage_over_area(area_polygon: Polygon, polygons: List[Polygon]) -> float:
    all_polys_union = cascaded_union(polygons)  # for now we use coverage, can use convex_hull or something else

    all_polys_union = all_polys_union.intersection(area_polygon)

    return min(all_polys_union.area / area_polygon.area, 1.0)  # 1.0 is max coverage


def distances_to_geo(geo: BaseGeometry, points: List[Point]) -> np.array:
    """
    Given a geographic object, return all distances to points
    Args:
        geo: Geographic object, in degrees
        points: List of all points in the area in degrees
    Returns:
        distances - list of all points distances to the geo object
    """
    if isinstance(geo, Polygon) or isinstance(geo, MultiPolygon):
        return np.array([distance_point_to_polygon(point, geo) for point in points])
    elif isinstance(geo, Point):  # geo is Point
        return np.array([distance_point_to_point(point, geo) for point in points])
    else:
        raise AttributeError("not correct type")


def distances_to_point(point: Point, geos: List[BaseGeometry]) -> np.array:
    """
    Given a point object, return all distances to geographic objects
    Args:
        geo: point, in degrees
        points: List of all Geographic object in the area in degrees
    Returns:
        distances - list of all geo objects distances to the point
    """
    distances = []
    for geo in geos:
        if isinstance(geo, Polygon) or isinstance(geo, MultiPolygon):
            distances.append(distance_point_to_polygon(point, geo))
        elif isinstance(geo, Point):  # geo is Point
            distances.append(distance_point_to_point(point, geo))
        else:
            raise AttributeError("not correct type")
    return distances

def sample_point_in_range(min_lon: float, min_lat: float, max_lon: float, max_lat: float,
                          seed=None) -> Point:
    if seed is not None:
        np.random.seed(seed)
    assert min_lat < max_lat and min_lon < max_lon, "min is not smaller than max"
    lat = np.random.uniform(min_lat, max_lat)
    lon = np.random.uniform(min_lon, max_lon)
    return Point(lon, lat)

def sample_points_in_range(min_lon: float, min_lat: float, max_lon: float, max_lat: float, number:int,
                          seed=None) -> Point:
    points_list = [sample_point_in_range(min_lon, min_lat, max_lon, max_lat) for i in range(number)]
    return points_list

def sample_points_in_polygon(outer_polygon: Polygon, number:int,
                             seed=None) -> Point:
    points_list = [sample_point_in_poly(outer_polygon) for i in range(number)]
    return points_list

def sample_point_in_poly(poly: Polygon, seed=None) -> Point:
    if seed is not None:
        np.random.seed(seed)
    bounds = poly.bounds
    pnt = Point(sample_point_in_range(*bounds))
    while not pnt.within(poly):
        pnt = Point(sample_point_in_range(*bounds))

    return pnt


def sample_grid_in_range(min_lon: float, min_lat: float, max_lon: float, max_lat: float,
                         step: float = 0.01) -> np.array:
    x = np.arange(min_lon, max_lon, step=step)
    y = np.arange(min_lat, max_lat, step=step)
    coords = np.stack(np.meshgrid(x, y), -1)

    return coords


def sample_grid_in_poly(poly: Polygon, step: float = 50, verbose=False) -> List[Point]:
    """
    sample Points as a grid inside a given polygon

    Args:
        poly: polygon to sample points in it
        step: step size between point sampling in meters

    Returns:
        a list of Points
    """
    step_degrees = meters2degrees(step)
    coords = sample_grid_in_range(*poly.bounds, step=step_degrees).reshape(-1, 2)

    coords = [Point(coord) for coord in tqdm(coords, desc='Sampling points', unit=' point')]
    coords_in_poly = list(filter(lambda point: point.within(poly), tqdm(coords, desc='filter in poly', unit=' point')))
    if verbose:
        print(f"Sampled {len(coords_in_poly)} points in the polygon")
    return coords_in_poly


def meters2degrees(meters: float) -> float:
    """
    convert meters to degrees
    Args:
        meters: input in meters

    Returns:
        output in degrees
    """
    return meters * 1e-5 / 1.11


def degrees2meters(degrees: float) -> float:
    """
    convert degrees to meters
    Args:
        meters: input in degrees

    Returns:
        output in meters
    """
    return degrees * 1.11 / 1e-5


def sample_points_in_poly(poly: Polygon, num_samples=1, seed=None) -> List[Point]:
    """
    returns a number of distinct points sampled from a polygon
    Args:
        poly: The area polygon from which we sample
        num_samples: The number of samples we want
        seed: Optional random seed to set

    Returns:
        List of sampled points
    """
    if seed is not None:
        np.random.seed(seed)

    pnts = []
    pnts_wkts = set()
    counter = 0
    # in order to combat duplicate samples I've put this for, running at most num_samples*50 times
    while len(pnts) < num_samples and counter < num_samples * 50:
        pnt = sample_point_in_poly(poly)
        counter += 1
        if pnt.wkt not in pnts_wkts:
            pnts.append(pnt)
            pnts_wkts.add(pnt.wkt)

    if len(pnts) < num_samples and counter == num_samples * 50:
        raise AssertionError("Can't sample distinct points")

    return pnts


def poly_outer_intersection(poly: Polygon, delete_polys: List[Polygon]) -> Polygon:
    """
    Returns the polygon without the areas contained inside $delete_polys
    :param poly: The area polygon from which we remove areas
    :param delete_polys: The areas we want to remove
    :return: The "cut-out" polygon
    """
    poly = copy.deepcopy(poly)
    for delete_poly in delete_polys:
        if poly.intersects(delete_poly):
            # If they intersect, create a new polygon that removes the area where they intersect
            poly = poly.difference(delete_poly.buffer(1e-5))
            assert not poly.intersects(delete_poly), "shouldn't intersect"
    return poly


def wkt_to_centers(wkt_array: Iterable[str]) -> List[Tuple[float, float]]:
    """
    Convert an array of geometries in wkt format to their center coordinates
    Args:
        wkt_array: An array of string wkt of geometries
    Returns:
        List of the geom's centers a lon, lat
    """
    wkt_centers = []
    for wkt_str in wkt_array:
        wkt_obj = wkt.loads(wkt_str)
        coords = wkt_obj.coords[0] if isinstance(wkt_obj, Point) else wkt_obj.centroid.coords[0]
        wkt_centers.append(coords)

    return wkt_centers


def geoms2bbox(geoms: gpd.GeoSeries) -> np.array:
    """
    get bbox that covers all geometries in geoms GeoSeries
    Args:
        geoms: GeoSeries of all geometries
    Returns:
        min_lon, min_lat, max_lon, max_lat - bbox of all geoms
    """
    bboxes = geoms.bounds
    min_lon, min_lat, max_lon, max_lat = bboxes.minx.min(), bboxes.miny.min(), bboxes.maxx.max(), bboxes.maxy.max()
    return min_lon, min_lat, max_lon, max_lat


def geom2image_projection(image: np.array, bbox: tuple, geom: shapely.geometry, color: Union[float, List],
                          fill: bool = True, line_width: int = 1) -> np.array:
    """
    Given an image and its bbox, project geom onto the image.
    Args:
        image: the image to project on
        bbox: bbox of type (min_lon, min_lat, max_lon, max_lat)
        geom: shapely geometry
        color: color of the geometry
        fill: True if one wants to fill the image, False otherwise
        line_width: width of the edge of the geometry

    Returns:
        image - after projection of geom onto it.
    """
    transform = lambda coord: [int(image.shape[1] * (coord[0] - bbox[0]) / (bbox[2] - bbox[0])),
                               int(image.shape[0] * (coord[1] - bbox[1]) / (bbox[3] - bbox[1]))]

    # get points
    if type(geom) == Point or type(geom) == LineString:
        coords = geom.coords
    elif type(geom) == Polygon:
        coords = geom.exterior.coords
        if fill:
            line_width = -1
    elif type(geom) == MultiPolygon:
        for polygon in geom:
            image = geom2image_projection(image, bbox, polygon, color, fill)
        return image
    elif type(geom) == MultiLineString:
        for linestring in geom:
            image = geom2image_projection(image, bbox, linestring, color, fill)
        return image
    else:
        raise Exception(f'get_geo_visibility doesnt support {type(geom)} geometry')

    pts = np.array([transform(coord) for coord in coords])

    # draw geom
    cv2.drawContours(image, [pts], -1, color, line_width)

    return image

def get_closest_geo(point: Point, geos: List[BaseGeometry]) -> BaseGeometry:
    """
    gets points' closest geo in geos.
    Args:
        point: the point to get the closest too
        geos: list of all geographic objects

    Returns:
        - The closest geo in geos to point
    """
    distances = distances_to_point(point, geos)
    closest_geo_index = int(np.argmin(distances))
    closest_geo = geos[closest_geo_index]

    return closest_geo
