from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import folium
from scipy import interpolate

from topo2vec import config


def interpolate_scores(coords: np.array, scores: np.array, coord_range: tuple, step: float = 0.001) -> np.array:
    """
    Given a coord_range and values for specific coords - interpolate to the rest of the grid
    Args:
        coords: array of lons and lats of points that their values are known
        scores: array of the coords values
        coord_range: range of the desired grid
        step: resolution of sample

    Returns:
        z: np.array - 2D array of the values in the entire grid of coord_range
    """
    min_lon, min_lat, max_lon, max_lat = coord_range

    x = np.arange(min_lon, max_lon, step=step)
    y = np.arange(min_lat, max_lat, step=step)
    grid_x, grid_y = np.meshgrid(x, y)
    z = interpolate.griddata(coords, scores, (x[None, :], y[:, None]), method='linear')

    return z


def visualize_predictions(coords: np.array, scores: np.array, coord_range: tuple = config.range, step: float = 0.001) -> folium.Map:
    """
    Create a folium map of scores using only several known coords and their values (by interpolating).
    Args:
        coords: array of lons and lats of points that their values are known
        scores: array of the coords values
        coord_range: range of the desired grid
        step: resolution of sample

    Returns:
        map_f: folium.Map - with a ImageOverlay of the scores in coord_range
    """
    map_f = folium.Map(location=coords[-1])
    if sum(scores > 0) != 0:
        scores[scores > 0] = scores[scores > 0] / np.max(scores[scores > 0])
    if sum(scores < 0) != 0:
        scores[scores < 0] = scores[scores < 0] / np.max(np.abs(scores[scores < 0]))
    #     scores /= np.max(np.abs(scores))
    scores = scores/2 + 0.5

    z = interpolate_scores(coords, scores, step=step, coord_range=coord_range)
    cm = LinearSegmentedColormap.from_list('rgb', [(0, 0, 1), (0, 1, 0), (1, 0, 0)], N=256, gamma=1.0)

    folium.raster_layers.ImageOverlay(z, bounds=[[coord_range[1], coord_range[0]],
                                                 [coord_range[3], coord_range[2]]],
                                      colormap=cm, opacity=0.3, origin='lower').add_to(map_f)

    for coord in coords:
        folium.CircleMarker(location=coord, radius=1, color='#000000', fill=True).add_to(map_f)

    return map_f
