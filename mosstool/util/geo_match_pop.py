"""
Geojson/shapefile format matches population
"""

import logging
from copy import deepcopy
from functools import partial
from math import asin, ceil, cos, floor, radians, sin, sqrt
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Tuple, Union, cast

import numpy as np
import rasterio
from geojson import FeatureCollection
from geopandas.geodataframe import GeoDataFrame
from shapely.geometry import MultiPolygon, Point, Polygon

from ..map._map_util.aoiutils import geo_coords
from ..map._map_util.const import *

__all__ = ["geo2pop"]



def _gps_distance(
    LON1: Union[float, Tuple[float, float]],
    LAT1: Union[float, Tuple[float, float]],
    LON2: Optional[float] = None,
    LAT2: Optional[float] = None,
):
    """
    GPS distance
    """
    if LON2 == None:  # input is [lon1,lat1], [lon2,lat2]
        lon1, lat1 = cast(Tuple[float, float], LON1)
        lon2, lat2 = cast(Tuple[float, float], LAT1)
    else:  # input is lon1, lat1, lon2, lat2
        assert LAT2 != None, "LON2 and LAT2 should be both None or both not None"
        LON1 = cast(float, LON1)
        LAT1 = cast(float, LAT1)
        lon1, lat1, lon2, lat2 = LON1, LAT1, LON2, LAT2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return float(2 * asin(sqrt(a)) * 6371393)


def _get_idx_range_in_bbox(min_x, max_x, min_y, max_y, xy_bound, mode="loose"):
    """
    Get all pixel_idx in a longitude and latitude bbox. For processing at the boundary, there are two modes: "loose" and "tight"
    """
    x_left, y_upper, x_step, y_step = xy_bound
    i_min = (min_x - x_left) / x_step
    i_max = (max_x - x_left) / x_step
    if i_min > i_max:  # in case x_step < 0
        i_min, i_max = i_max, i_min

    j_min = (min_y - y_upper) / y_step
    j_max = (max_y - y_upper) / y_step
    if j_min > j_max:  # in case y_step < 0
        j_min, j_max = j_max, j_min

    if mode == "loose":
        i_min, i_max = floor(i_min), ceil(i_max)
        j_min, j_max = floor(j_min), ceil(j_max)
    elif mode == "tight":
        i_min, i_max = ceil(i_min), floor(i_max)
        j_min, j_max = ceil(j_min), floor(j_max)
    return i_min, i_max, j_min, j_max


def _get_pixel_info(band, x_left, y_upper, x_step, y_step, bbox, padding=20):
    """
    Get the information of each WorldPop_pixel within the latitude and longitude range: {idx(i,j) : (Point(lon, lat), population)}
    Original data pixel size: lon_step: 0.0008333333, lat_step: -0.0008333333, ~ 100m * 100m
    """

    min_lon, max_lon, min_lat, max_lat = bbox
    i_min, i_max, j_min, j_max = _get_idx_range_in_bbox(
        min_x=min_lon,
        max_x=max_lon,
        min_y=min_lat,
        max_y=max_lat,
        xy_bound=(x_left, y_upper, x_step, y_step),
        mode="tight",
    )
    i_min -= padding  # Get more data outside the bbox range to avoid key_error when a small amount of aoi occurs outside the bbox
    i_max += padding
    j_min -= padding
    j_max += padding

    return {
        (i, j): (
            Point((x_left + i * x_step, y_upper + j * y_step)),
            max(
                0, band[0, j][i]
            ),  # The population of 0 in the original data is represented by -99999
        )
        for i in range(i_min, i_max + 1)
        for j in range(j_min, j_max + 1)
    }


def _upsample_pixels_unit(arg):
    """
    The original pixel is about 100 * 100, now it is divided equally into (100/n) * (100/n)
    The population is not evenly distributed among all cells, but is evenly divided by the cells covered by aoi.
    """
    global all_geos
    (n_upsample, x_step, y_step), ((i, j), (point, pop)) = arg
    ni, nj = n_upsample * i, n_upsample * j
    x, y = point.x, point.y
    range_n = list(range(n_upsample))
    x_next, y_next = x + n_upsample * x_step, y + n_upsample * y_step
    pixels = []
    for u in range_n:
        for v in range_n:
            p = Point(x + x_step * u, y + y_step * v)
            # Determine whether p is covered by a rectangular frame
            for geo in all_geos:
                poly, (x_min, y_min, x_max, y_max) = geo
                if x_next > x_min and x < x_max and y_next > y_min and y < y_max:
                    if p.within(poly):
                        pixels.append([(ni + u, nj + v), [p, 1]])
                        break
            else:
                pixels.append([(ni + u, nj + v), [p, 0]])
    n_covered = sum([x[1][1] for x in pixels])
    if (
        n_covered > 0
    ):  # The population is evenly distributed among the cells covered by the rectangular frame
        pop_new = pop / n_covered
        for x in pixels:
            if x[1][1] > 0:
                x[1][1] = pop_new
    else:
        pop_new = pop / n_upsample**2
        for x in pixels:
            x[1][1] = pop_new
    return pixels


def _upsample_pixels_idiot_unit(arg):
    """
    The original pixel is about 100 * 100, now it is divided equally into (100/n) * (100/n)
    The population is divided equally among the cells
    """
    (n_upsample, x_step, y_step), ((i, j), (point, pop)) = arg
    ni, nj = n_upsample * i, n_upsample * j
    x, y = point.x, point.y
    pop_new = pop / n_upsample**2
    range_n = list(range(n_upsample))
    return [
        [(ni + u, nj + v), [Point(x + x_step * u, y + y_step * v), pop_new]]
        for u in range_n
        for v in range_n
    ]


def _get_poly_pop_unit(arg, geo_item):
    """
    Estimate the polygonal box: take the population sum of the pixels falling inside it
    If aoi is too small so that no pixel falls within it, take the population of the pixel where it is located and multiply it by the area ratio of aoi to pixel.
    """
    global pixel_idx2point_pop
    (
        x_left,
        y_upper,
        x_step,
        y_step,
        xy_gps_scale2,
        pixel_area,
    ) = arg
    idx, (poly, (min_x, min_y, max_x, max_y)) = geo_item
    i_min, i_max, j_min, j_max = _get_idx_range_in_bbox(
        min_x=min_x,
        min_y=min_y,
        max_x=max_x,
        max_y=max_y,
        xy_bound=(x_left, y_upper, x_step, y_step),
    )
    # Get the population sum of the pixels falling within it
    total_pop = 0
    has_inside_pixel = False
    for i in range(i_min, i_max + 1):
        for j in range(j_min, j_max + 1):
            try:
                point, pop = pixel_idx2point_pop[(i, j)]
                if point.within(poly):
                    has_inside_pixel = True
                    total_pop += pop
            except KeyError:
                pass
    if has_inside_pixel:
        return (idx, total_pop)

    # If aoi is too small so that no pixel falls within it, take the population of the pixel where it is located.
    x, y = geo_coords(poly.centroid)[0]
    try:
        t = (
            pixel_idx2point_pop[
                (int((x - x_left) / x_step), int((y - y_upper) / y_step))
            ][-1]
            * poly.area
            * xy_gps_scale2
            / pixel_area
        )  # the area for calculate should be poly.area * xy_gps_scale2
        total_pop = t
    except KeyError:
        total_pop = 0
    return (idx, total_pop)


def _get_geo_pop(
    geos,
    workers,
    x_left,
    y_upper,
    x_step,
    y_step,
    xy_gps_scale2,
    pixel_area,
):
    geos_dict = {i: d for i, d in enumerate(geos)}
    arg = (
        x_left,
        y_upper,
        x_step,
        y_step,
        xy_gps_scale2,
        pixel_area,
    )
    _get_poly_pop_unit_with_arg = partial(_get_poly_pop_unit, arg)
    result = []
    list_geos_items = list(geos_dict.items())
    for i in range(0, len(list_geos_items), MAX_BATCH_SIZE):
        list_geos_items_batch = list_geos_items[i : i + MAX_BATCH_SIZE]
        with Pool(processes=workers) as pool:
            result += pool.map(
                _get_poly_pop_unit_with_arg,
                list_geos_items_batch,
                chunksize=min(ceil(len(list_geos_items_batch) / workers), 200),
            )

    idx2pop = {i: pop for (i, pop) in result}
    return idx2pop


def geo2pop(
    geo_data: Union[GeoDataFrame, FeatureCollection],
    pop_tif_path: str,
    upsample_factor: int = 4,
    pop_in_aoi_factor: float = 0.7,
) -> Union[GeoDataFrame, FeatureCollection]:
    """
    Args:
    - geo_data (GeoDataFrame | FeatureCollection): polygon geo files.
    - pop_tif_path (str): path to population tif file.
    - upsample_factor (int): scaling factor for dividing the raw population data grid.
    - pop_in_aoi_factor (float): the proportion of the total population within the AOI.

    Returns:
    - geo_data (GeoDataFrame | FeatureCollection): geo files with population.
    """

    global pixel_idx2point_pop, all_geos
    geo_data = deepcopy(geo_data)
    orig_geo_data = deepcopy(geo_data)
    geo_type = type(geo_data)
    all_geos = []
    all_coords_lonlat = []
    if not geo_type in [GeoDataFrame, FeatureCollection]:
        logging.warning(f"Unsupported data type {geo_type}")
        return geo_data
    elif geo_type == GeoDataFrame:
        geo_data = cast(GeoDataFrame, geo_data)
        for _, polygon in enumerate(geo_data.geometry):
            all_geos.append((polygon, polygon.bounds))
            all_coords_lonlat.extend([c[:2] for c in geo_coords(polygon)])
    elif geo_type == FeatureCollection:
        geo_data = cast(FeatureCollection, geo_data)
        for feature in geo_data["features"]:
            if not feature["geometry"]["type"] == "Polygon":
                raise ValueError("bad geometry type: " + feature)
            if "properties" not in feature:
                raise ValueError("no properties in feature: " + feature)
            coords = np.array(
                feature["geometry"]["coordinates"][0], dtype=np.float64
            )  # inner poly is unsupported
            lonlat_coords = [c[:2] for c in coords]
            polygon = Polygon(lonlat_coords)
            all_geos.append((polygon, polygon.bounds))
            all_coords_lonlat.extend(lonlat_coords)
    all_coords_lonlat = np.array(all_coords_lonlat)
    if len(all_coords_lonlat) < 2:
        logging.warning("No polygons to add pop!")
        return geo_data

    min_lon, min_lat = np.min(all_coords_lonlat, axis=0)
    max_lon, max_lat = np.max(all_coords_lonlat, axis=0)
    bbox = (min_lon, max_lon, min_lat, max_lat)
    lon_cen, lat_cen = (min_lon + max_lon) / 2, (min_lat + max_lat) / 2
    workers = cpu_count()
    # Read raw data
    raster = rasterio.open(pop_tif_path)  # world_population_dataset
    band = raster.read()
    raster_transform = raster.meta["transform"]
    x_step, y_step = raster_transform[0], raster_transform[4]  # lon_step lat_step
    x_left, y_upper = raster_transform[2], raster_transform[5]  # min_lon max_lat
    x_len = _gps_distance(lon_cen, lat_cen, lon_cen + x_step, lat_cen)
    y_len = _gps_distance(lon_cen, lat_cen, lon_cen, lat_cen + y_step)
    pixel_area = x_len * y_len  # The area of one pixel in the original data
    xy_gps_scale = (
        x_len / abs(x_step) + y_len / abs(y_step)
    ) / 2  # Conversion ratio between latitude and longitude and m
    xy_gps_scale2 = xy_gps_scale**2
    logging.info(f"original pixel size:{int(x_len)}*{int(y_len)}")
    # Preprocessed population data
    logging.info("Getting pixel info")
    pixel_idx2point_pop = _get_pixel_info(
        band=band,
        x_left=x_left,
        y_upper=y_upper,
        x_step=x_step,
        y_step=y_step,
        bbox=bbox,
        padding=20,
    )
    total_pop = sum([x[-1] for x in pixel_idx2point_pop.values()])
    logging.info(f"Original total population: {total_pop}")
    # Population data pixel upsampling
    n_upsample = upsample_factor
    # The first upsampling: the original grid population is evenly divided by the cells covered by aoi
    x_step /= n_upsample
    y_step /= n_upsample
    pixel_area /= n_upsample**2
    logging.info(f"Up-sampling pixel: {n_upsample} cut the pixel length")
    list_pixel2pop = list(pixel_idx2point_pop.items())
    results = []
    for i in range(0, len(list_pixel2pop), MAX_BATCH_SIZE):
        list_pixel2pop_batch = list_pixel2pop[i : i + MAX_BATCH_SIZE]
        with Pool(processes=workers) as pool:
            results += pool.map(
                _upsample_pixels_unit,
                [((n_upsample, x_step, y_step), d) for d in list_pixel2pop_batch],
                chunksize=min(
                    ceil(len(list_pixel2pop_batch) / workers),
                    1000,
                ),
            )
    pixel_idx2point_pop = {k: v for x in results for k, v in x}

    # The second upsampling: the original grid population is divided equally by all cells
    x_step /= n_upsample
    y_step /= n_upsample
    pixel_area /= n_upsample**2
    logging.info(f"Up-sampling pixel: {n_upsample} cut the pixel length")
    list_pixel2pop = list(pixel_idx2point_pop.items())
    results = []
    for i in range(0, len(list_pixel2pop), MAX_BATCH_SIZE):
        list_pixel2pop_batch = list_pixel2pop[i : i + MAX_BATCH_SIZE]
        with Pool(processes=workers) as pool:
            results += pool.map(
                _upsample_pixels_idiot_unit,
                [((n_upsample, x_step, y_step), d) for d in list_pixel2pop_batch],
                chunksize=min(
                    ceil(len(list_pixel2pop_batch) / workers),
                    1000,
                ),
            )
    pixel_idx2point_pop = {k: v for x in results for k, v in x}
    logging.info(f"Adding populations")
    idx2pop = _get_geo_pop(
        all_geos,
        workers,
        x_left,
        y_upper,
        x_step,
        y_step,
        xy_gps_scale2,
        pixel_area,
    )
    geo_total_pop = sum(idx2pop.values())
    # Post-processing, multiply by the proportional coefficient so that the total population within aoi is total_pop * pop_in_aoi_factor
    multiplier = total_pop * pop_in_aoi_factor / geo_total_pop
    for i in range(len(all_geos)):
        idx2pop[i] = ceil(multiplier * idx2pop[i])

    # Returns
    if geo_type == GeoDataFrame:
        # add population column
        geo_data = cast(GeoDataFrame, orig_geo_data)
        geo_data = geo_data.assign(population=[idx2pop[i] for i in range(len(geo_data.geometry))]) # type:ignore 
    elif geo_type == FeatureCollection:
        # add properties.population
        geo_data = cast(FeatureCollection, orig_geo_data)
        for i, feature in enumerate(geo_data["features"]):
            assert feature["geometry"]["type"] == "Polygon"
            assert "properties" in feature
            feature["properties"]["population"] = idx2pop[i]
    return geo_data
