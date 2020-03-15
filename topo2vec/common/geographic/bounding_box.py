from typing import List, Tuple

from shapely.geometry import Polygon


class BoundingBox:
    def __init__(self, minX: float, maxX: float, minY: float, maxY: float):
        self.minX = minX
        self.maxX = maxX
        self.minY = minY
        self.maxY = maxY

    def center(self) -> (float, float):
        """
        Returns: the center of the bounding box
        """
        centerX = (self.minX + self.maxX) / 2
        centerY = (self.minY + self.maxY) / 2
        return centerX, centerY

    def to_coord_list(self) -> List[Tuple[float, float]]:
        return [(self.minX, self.minY), (self.minX, self.maxY), (self.maxX, self.maxY), (self.maxX, self.minY),
                (self.minX, self.minY)]

    def to_polygon(self) -> Polygon:
        return Polygon(self.to_coord_list())

    @staticmethod
    def from_polygon(poly: Polygon):
        min_lon, min_lat, max_lon, max_lat = poly.bounds
        return BoundingBox(min_lon, max_lon, min_lat, max_lat)
