from shapely.geometry import Point

from server_api.client.client_lib import get_class_points, build_polygon, get_top_n_similar_points_in_polygon

WORKING_POLYGON = build_polygon(34.7, 31.3, 34.9, 31.43)
points_list = [Point(34.75, 31.35), Point(34.76, 31.36)]

patches, points = get_class_points(WORKING_POLYGON, 500, 'peaks')
patches, points = get_top_n_similar_points_in_polygon(points_list, 10, WORKING_POLYGON, 500)

print(points)