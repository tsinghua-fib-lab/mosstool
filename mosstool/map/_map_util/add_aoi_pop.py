import logging
from math import asin, ceil, cos, floor, radians, sin, sqrt
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple, Union, cast

import pyproj
import rasterio
from shapely.geometry import MultiPolygon, Point, Polygon

from .._map_util.aoiutils import geo_coords
from .const import *

__all__ = ["add_aoi_pop"]




def _gps_distance(
    LON1: Union[float, Tuple[float, float]],
    LAT1: Union[float, Tuple[float, float]],
    LON2: Optional[float] = None,
    LAT2: Optional[float] = None,
):
    """
    Distance between GPS points (m)
    """
    if LON2 == None:  # The input is [lon1,lat1], [lon2,lat2]
        lon1, lat1 = cast(Tuple[float, float], LON1)
        lon2, lat2 = cast(Tuple[float, float], LAT1)
    else:  # The input is lon1, lat1, lon2, lat2
        assert LAT2 != None, "LON2 and LAT2 should be both None or both not None"
        LON1 = cast(float, LON1)
        LAT1 = cast(float, LAT1)
        lon1, lat1, lon2, lat2 = LON1, LAT1, LON2, LAT2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return float(2 * asin(sqrt(a)) * 6371393)


def _get_idx_range_in_bbox(min_x, max_x, min_y, max_y, mode="loose"):
    """
    Get all pixel_idx in a longitude and latitude bbox. For processing at the boundary, there are two modes: "loose" and "tight".
    """
    global x_left, y_upper, x_step, y_step
    i_min = (min_x - x_left) / x_step
    i_max = (max_x - x_left) / x_step
    if i_min > i_max:  # in case x_step < 0
        t = i_min
        i_min = i_max
        i_max = t

    j_min = (min_y - y_upper) / y_step
    j_max = (max_y - y_upper) / y_step
    if j_min > j_max:  # in case y_step < 0
        t = j_min
        j_min = j_max
        j_max = t

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
    global aois_poly_global, n_upsample, x_step, y_step
    (i, j), (point, pop) = arg
    ni, nj = n_upsample * i, n_upsample * j
    x, y = point.x, point.y
    range_n = list(range(n_upsample))
    x_next, y_next = x + n_upsample * x_step, y + n_upsample * y_step
    pixels = []
    for u in range_n:
        for v in range_n:
            p = Point(x + x_step * u, y + y_step * v)
            # Determine whether p is covered by aoi
            for aoi in aois_poly_global:
                x_min, y_min, x_max, y_max = aoi["bound"]
                if x_next > x_min and x < x_max and y_next > y_min and y < y_max:
                    if p.within(aoi["poly"]):
                        pixels.append([(ni + u, nj + v), [p, 1]])
                        break
            else:
                pixels.append([(ni + u, nj + v), [p, 0]])
    n_covered = sum([x[1][1] for x in pixels])
    if (
        n_covered > 0
    ):  # The population is evenly distributed among the cells covered by aoi
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
    global n_upsample, x_step, y_step
    (i, j), (point, pop) = arg
    ni, nj = n_upsample * i, n_upsample * j
    x, y = point.x, point.y
    pop_new = pop / n_upsample**2
    range_n = list(range(n_upsample))
    return [
        [(ni + u, nj + v), [Point(x + x_step * u, y + y_step * v), pop_new]]
        for u in range_n
        for v in range_n
    ]


def _get_aoi_point_pop_unit(aoi):
    """
    To estimate the population of a single point aoi (essentially poi): take the population of the pixel where it is located, and multiply it by the area ratio of aoi to pixel
    """
    global pixel_idx2point_pop, x_left, y_upper, x_step, y_step, aoi_point_area, pixel_area
    x, y = aoi["lonlat"]
    try:
        t = (
            pixel_idx2point_pop[
                (int((x - x_left) / x_step), int((y - y_upper) / y_step))
            ][-1]
            * aoi_point_area
            / pixel_area
        )
        aoi["external"]["population"] = t
    except KeyError:
        aoi["external"]["population"] = 0
    return aoi


def _get_aoi_poly_pop_unit(aoi):
    """
    Estimate the population of polygon aoi: take the population sum of pixels falling inside it
    If aoi is too small so that no pixel falls within it, take the population of the pixel where it is located and multiply it by the area ratio of aoi to pixel.
    """
    global pixel_idx2point_pop, x_left, y_upper, x_step, y_step, xy_gps_scale2, pixel_area
    poly = aoi["poly"]
    min_x, min_y, max_x, max_y = aoi["bound"]
    i_min, i_max, j_min, j_max = _get_idx_range_in_bbox(
        min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y
    )  # Get the population sum of the pixels falling within it
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
        aoi["external"]["population"] = total_pop
        return aoi, HAS_INSIDE_PIXEL

    # If aoi is too small so that no pixel falls within it, take the population of the pixel where it is located.
    x, y = geo_coords(poly.centroid)[0][:2]
    try:
        t = (
            pixel_idx2point_pop[
                (int((x - x_left) / x_step), int((y - y_upper) / y_step))
            ][-1]
            * poly.area
            * xy_gps_scale2
            / pixel_area
        )  # The square meter area of poly should be poly.area * xy_gps_scale2
        aoi["external"]["population"] = t
    except KeyError:
        aoi["external"]["population"] = 0
        # print("warning: aoi outside of bbox, please get the pixel info with larger padding")
    return aoi, NO_INSIDE_PIXEL


def _get_aoi_pop(aois_point, aois_poly, workers):
    aois_point_result = []
    for i in range(0, len(aois_point), MAX_BATCH_SIZE):
        aois_point_batch = aois_point[i : i + MAX_BATCH_SIZE]
        with Pool(processes=workers) as pool:
            aois_point_result += pool.map(
                _get_aoi_point_pop_unit,
                aois_point_batch,
                chunksize=min(ceil(len(aois_point_batch) / workers), 1000),
            )
    aois_poly_result = []
    for i in range(0, len(aois_poly), MAX_BATCH_SIZE):
        aois_poly_batch = aois_poly[i : i + MAX_BATCH_SIZE]
        with Pool(processes=workers) as pool:
            aois_poly_result += pool.map(
                _get_aoi_poly_pop_unit,
                aois_poly_batch,
                chunksize=min(ceil(len(aois_poly_batch) / workers), 200),
            )
    aois_point, aois_poly = aois_point_result, aois_poly_result
    aois_poly, flags = zip(*aois_poly)
    logging.info(
        f"proportion of polygon aoi without inside pixel:{len([x for x in flags if x == NO_INSIDE_PIXEL]) / len(flags)}",
    )
    # In the polygon Aoi, no pixel falls inside the proportion
    # n_upsample=2,  0.453
    # n_upsample=3,  0.293
    # n_upsample=4,  0.183
    # n_upsample=5,  0.110
    # n_upsample=6,  0.061
    # n_upsample=7,  0.034
    # n_upsample=8,  0.021
    # n_upsample=15  0.002
    return aois_point + list(aois_poly)


# The imaginary area of a single point aoi (from poi), based on the nearest pixel_pop * aoi_default_area / pixel_area as its pop
aoi_point_area = 50
# The proportion of people staying in aoi among the total population
pop_in_aoi_portion = 0.7
HAS_INSIDE_PIXEL = 0
NO_INSIDE_PIXEL = 1


def add_aoi_pop(
    aois: Union[List, Dict],
    max_longitude: float,
    min_longitude: float,
    max_latitude: float,
    min_latitude: float,
    proj_str: str,
    upsample_factor: int = 4,
    workers: int = 32,
    tif_path: Optional[str] = None,
):
    """Add the population field to the AOI external"""
    if tif_path is None:
        logging.warning("No world-pop data!")
        return
    min_lon, max_lon, min_lat, max_lat = (
        min_longitude,
        max_longitude,
        min_latitude,
        max_latitude,
    )
    bbox = (min_lon, max_lon, min_lat, max_lat)
    lon_cen, lat_cen = (min_lon + max_lon) / 2, (min_lat + max_lat) / 2
    global pixel_idx2point_pop, aois_poly_global, x_left, y_upper, x_step, y_step, pixel_area, aoi_point_area, xy_gps_scale2, n_upsample
    # Preprocess AOI data
    logging.info("Pre-processing aois")
    has_pop = False
    AOIS_TYPE = type(aois)
    if type(aois) == dict:
        aois = list(aois.values())
    for a in aois:
        if "population" in a["external"] and a["external"]["population"] > 0:
            has_pop = True
    if has_pop:
        logging.info("AOI already has pop")
        return
    projector = pyproj.Proj(proj_str)
    if AOIS_TYPE == list:
        aois_point = [aoi for aoi in aois if len(aoi["coords"]) == 1]
        aois_poly = [aoi for aoi in aois if len(aoi["coords"]) > 1]
        for aoi in aois_poly:
            coords_xy = np.array([c[:2] for c in aoi["coords"]])
            lons, lats = projector(coords_xy[:, 0], coords_xy[:, 1], inverse=True)
            aoi["poly"] = Polygon(list(zip(lons, lats)))
            aoi["bound"] = aoi["poly"].bounds
        for aoi in aois_point:
            x, y = aoi["coords"][0][:2]
            aoi["lonlat"] = projector(x, y, inverse=True)
    else:
        aois_point = [aoi for aoi in aois if len(aoi["positions"]) == 1]
        aois_poly = [aoi for aoi in aois if len(aoi["positions"]) > 1]
        for aoi in aois_poly:
            coords_xy = np.array([(c["x"], c["y"]) for c in aoi["positions"]])
            lons, lats = projector(coords_xy[:, 0], coords_xy[:, 1], inverse=True)
            aoi["poly"] = Polygon(list(zip(lons, lats)))
            aoi["bound"] = aoi["poly"].bounds
        for aoi in aois_point:
            x, y = aoi["positions"][0]["x"], aoi["positions"][0]["y"]
            aoi["lonlat"] = projector(x, y, inverse=True)
    aois_poly_global = aois_poly
    # Read raw data
    raster = rasterio.open(tif_path)  # world_population_dataset
    band = raster.read()
    raster_transform = raster.meta["transform"]
    x_step = raster_transform[0]  # lon_step
    y_step = raster_transform[4]  # lat_step
    x_left = raster_transform[2]  # min_lon
    y_upper = raster_transform[5]  # max_lat
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
                list_pixel2pop_batch,
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
                list_pixel2pop_batch,
                chunksize=min(
                    ceil(len(list_pixel2pop_batch) / workers),
                    1000,
                ),
            )
    pixel_idx2point_pop = {k: v for x in results for k, v in x}

    # Calculate aoi population
    logging.info("Calculating aoi population")
    aois_with_pop = _get_aoi_pop(
        aois_point=aois_point, aois_poly=aois_poly, workers=workers
    )
    aoi_total_pop = sum([aoi["external"]["population"] for aoi in aois_with_pop])

    # Post-processing, directly multiply by a certain ratio to make the total population in aoi total_pop * pop_in_aoi_portion
    multiplier = total_pop * pop_in_aoi_portion / aoi_total_pop
    if AOIS_TYPE == list:
        for aoi in aois_with_pop:
            aoi["external"]["population"] = ceil(
                multiplier * aoi["external"]["population"]
            )
            if len(aoi["coords"]) > 1:
                del aoi["poly"]
                del aoi["bound"]
            if len(aoi["coords"]) == 1:
                del aoi["lonlat"]
        return aois_with_pop
    else:
        for aoi in aois_with_pop:
            aoi["external"]["population"] = ceil(
                multiplier * aoi["external"]["population"]
            )
            if len(aoi["positions"]) > 1:
                del aoi["poly"]
                del aoi["bound"]
            if len(aoi["positions"]) == 1:
                del aoi["lonlat"]
        return {a["id"]: a for a in aois_with_pop}
