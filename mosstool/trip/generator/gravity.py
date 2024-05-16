import geopandas as gpd
import numpy as np
from scipy.spatial import distance_matrix

__all__ = ["GravityGenerator"]


class GravityGenerator:
    """
    Generate OD matrix inside the given area, based on the gravity model.
    """

    def __init__(
        self,
        Lambda: float,
        Alpha: float,
        Beta: float,
        Gamma: float,
    ):
        """
        Args:
        - pops (list[int]): The population of each area.
        - dists (np.ndarray): The distance matrix between each pair of areas.
        """
        self._lambda = Lambda
        self._alpha = Alpha
        self._beta = Beta
        self._gamma = Gamma

    def load_area(
        self,
        area: gpd.GeoDataFrame,
    ):
        """
        Load the area data.

        Args:
        - area (gpd.GeoDataFrame): The area data.
        """
        self._area = area

    def _get_one_point(self):
        """
        get one point from the shapefile
        """
        first_geometry = self._area.geometry.iloc[0]

        # for different types of geometry, get the first point
        if first_geometry.geom_type == "Polygon":
            first_point = first_geometry.exterior.coords[0]
        elif first_geometry.geom_type == "MultiPolygon":
            first_polygon = list(first_geometry.geoms)[0]
            first_point = first_polygon.exterior.coords[0]
        else:
            raise ValueError("Geometry type not supported")

        pointx, pointy = first_point[0], first_point[1]

        return pointx, pointy

    def _calculate_utm_epsg(self, longitude: float, latitude: float):
        """
        Calculate the UTM zone and corresponding EPSG code for a given longitude and latitude.

        Args:
        longitude (float): The longitude of the location.
        latitude (float): The latitude of the location.

        Returns:
        int: The EPSG code for the UTM zone.
        """
        # Determine the UTM zone from the longitude
        utm_zone = int((longitude + 180) / 6) + 1

        # Determine the hemisphere and construct the EPSG code
        if latitude >= 0:
            # Northern Hemisphere
            epsg_code = 32600 + utm_zone
        else:
            # Southern Hemisphere
            epsg_code = 32700 + utm_zone

        return epsg_code

    def cal_distance(self):
        """
        Euclidean distance matrix
        get the distance matrix for regions in the area
        based on the shapefile of the area
        """
        pointx, pointy = self._get_one_point()
        epsg = self._calculate_utm_epsg(pointx, pointy)
        area = self._area.to_crs(f"EPSG:{epsg}")
        area["centroid"] = area["geometry"].centroid  # type:ignore

        # dis
        points = area["centroid"].apply(lambda p: [p.x, p.y]).tolist()  # type:ignore
        dist_matrix = distance_matrix(points, points)

        distance = dist_matrix.astype(np.float32)

        return distance

    def generate(self, pops):  #: list[int]
        """
        Generate the OD matrix based on the gravity model.

        Args:
        - pops (list[int]): The population of each area.
        - dists (np.ndarray): The distance matrix between each pair of areas.

        Returns:
        - np.ndarray: The generated OD matrix.
        """
        dists = self.cal_distance()

        N = np.power(pops, self._alpha)
        M = np.power(pops, self._beta)
        D = np.power(dists, self._gamma)

        N = N.reshape(-1, 1).repeat(len(pops), axis=1)
        M = M.reshape(1, -1).repeat(len(pops), axis=0)

        od_matrix = self._lambda * N * M / (D + 1e-8)
        od_matrix = od_matrix.astype(np.int64)
        od_matrix[od_matrix < 0] = 0
        for i in range(od_matrix.shape[0]):
            od_matrix[i, i] = 0

        return od_matrix
