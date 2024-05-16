"""Fetch the subway bus data from transitland"""

import logging
from collections import defaultdict
from math import asin, cos, radians, sin, sqrt
from typing import List, Optional, Tuple, Union, cast

import numpy as np
import pyproj
import requests
from shapely.geometry import LineString, MultiPoint, Point, Polygon

from .._util.line import clip_line, offset_lane

__all__ = [
    "TransitlandPublicTransport",
]



def cut(line: LineString, points: List[Point], projstr: str):
    """
    Split routes based on stations
    Args:
    - line (LineString([(lat lon), (lat lon),]))
    - points (list[Point(lat lon),])
    """
    res = []
    projector = pyproj.Proj(projstr)
    coords_xy = [list(projector(*c)) for c in line.coords]
    points_xy = [Point(list(projector(p.x, p.y))) for p in points]
    line_i = LineString(coordinates=coords_xy)
    if len(points_xy) >= 2:
        p_start, p_end = points_xy[0], points_xy[-1]
        line_i = clip_line(line_i, p_start, p_end)
    for ii in range(len(points_xy) - 1):
        p_0 = points_xy[ii]
        p_1 = points_xy[ii + 1]
        res.append(clip_line(line_i, p_0, p_1))
    res = [offset_lane(l,-10) for l in res] # Translate the road
    return [[list(projector(*c, inverse=True)) for c in line.coords] for line in res]


def merge_geo(coord, projstr, square_length=350):
    SQUARE_SIDE_LENGTH = square_length # The side length of the rectangle formed by a single point POI
    projector = pyproj.Proj(projstr)
    if len(coord) > 1:
        geo = MultiPoint(coord).minimum_rotated_rectangle
        # Expand geo into a large square
        if isinstance(geo, Polygon):
            min_x = min([p[0] for p in geo.exterior.coords])  # type: ignore
            min_y = min([p[1] for p in geo.exterior.coords])  # type: ignore
            max_x = max([p[0] for p in geo.exterior.coords])  # type: ignore
            max_y = max([p[1] for p in geo.exterior.coords])  # type: ignore
        else:
            min_x = min([p[0] for p in geo.coords])  # type: ignore
            min_y = min([p[1] for p in geo.coords])  # type: ignore
            max_x = max([p[0] for p in geo.coords])  # type: ignore
            max_y = max([p[1] for p in geo.coords])  # type: ignore
        bottom_left = (min_x, min_y)
        top_left = (min_x, max_y)
        top_right = (max_x, max_y)
        bottom_right = (max_x, min_y)
        geo = [bottom_left, top_left, top_right, bottom_right, bottom_left]
        geo_xy = Polygon(
            [
                list(projector(*p))
                for p in [bottom_left, top_left, top_right, bottom_right, bottom_left]
            ]
        )
        if geo_xy.area < SQUARE_SIDE_LENGTH * SQUARE_SIDE_LENGTH:
            x, y = projector(0.5 * (min_x + max_x), 0.5 * (min_y + max_y))
            half_side = SQUARE_SIDE_LENGTH / 2
            bottom_left = (x - half_side, y - half_side)
            top_left = (x - half_side, y + half_side)
            top_right = (x + half_side, y + half_side)
            bottom_right = (x + half_side, y - half_side)
            geo = [
                list(projector(*p, inverse=True))
                for p in [bottom_left, top_left, top_right, bottom_right, bottom_left]
            ]
    else:
        geo = Point(coord)
        center = geo
        x, y = projector(center.x, center.y)
        half_side = SQUARE_SIDE_LENGTH / 2
        bottom_left = (x - half_side, y - half_side)
        top_left = (x - half_side, y + half_side)
        top_right = (x + half_side, y + half_side)
        bottom_right = (x + half_side, y - half_side)
        geo = [
            list(projector(*p, inverse=True))
            for p in [bottom_left, top_left, top_right, bottom_right, bottom_left]
        ]
    return geo


def get_sta_dis(sta1, sta2):
    x1, y1 = sta1["geo_xy"][0]
    x2, y2 = sta2["geo_xy"][0]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def gps_distance(
    LON1: Union[float, Tuple[float, float]],
    LAT1: Union[float, Tuple[float, float]],
    LON2: Optional[float] = None,
    LAT2: Optional[float] = None,
):
    """
    Distance between GPS points (m)
    """
    if LON2 == None: # The input is [lon1,lat1], [lon2,lat2]
        lon1, lat1 = cast(Tuple[float, float], LON1)
        lon2, lat2 = cast(Tuple[float, float], LAT1)
    else: # The input is lon1, lat1, lon2, lat2
        assert LAT2 != None, "LON2 and LAT2 should be both None or both not None"
        LON1 = cast(float, LON1)
        LAT1 = cast(float, LAT1)
        lon1, lat1, lon2, lat2 = LON1, LAT1, LON2, LAT2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return float(2 * asin(sqrt(a)) * 6371393)


class TransitlandPublicTransport:
    """
    Process Transitland raw data to Public Transport data as geojson format files
    """

    def __init__(
        self,
        proj_str: str,
        transitland_ak: str,
        max_longitude: float,
        min_longitude: float,
        max_latitude: float,
        min_latitude: float,
    ):
        self.MAX_SEARCH_RADIUS = 10000 # API’s maximum search radius
        self.MAX_STOP_NUM = 100 #The maximum number of sites for a single API search
        self.MAX_STOP_DISGATE = 15 # Anything beyond this distance is not considered a stop for this line.
        self.SUBWAY_SAME_STA_DIS = 800 # The merge distance threshold of subway stations with the same name
        self.BUS_SAME_STA_DIS = 30
        MIN_UNIT_NUM = 16 # Minimum number of search blocks
        lon1, lat1 = min_longitude, min_latitude
        lon2, lat2 = max_longitude, max_latitude
        self.lon_center = (lon1 + lon2) / 2
        self.lat_center = (lat1 + lat2) / 2
        self.ak = transitland_ak
        self.projstr = proj_str
        self.unit = max(
            MIN_UNIT_NUM,
            int(gps_distance(lon1, lat1, lon2, lat2) / self.MAX_SEARCH_RADIUS / 2 + 1),
        ) # unit-1 * unit-1 large raster
        self.lon_partition = [
            round(x, 6) for x in list(np.linspace(lon1, lon2, self.unit))
        ] # Used to calculate search radius
        self.lat_partition = [
            round(y, 6) for y in list(np.linspace(lat1, lat2, self.unit))
        ]
    def fetch_raw_stops(self):
        url = "https://transit.land/api/v2/rest/stops"
        stops_without_routes = []
        logging.info("Fetching raw stops from Transitland")
        for i in range(0, self.unit - 1): # Horizontal grid index
            for j in range(0, self.unit - 1): # Vertical grid index
                lon_1, lat_1 = self.lon_partition[i], self.lat_partition[j]
                lon_2, lat_2 = self.lon_partition[i + 1], self.lat_partition[j + 1]
                radius = min(
                    gps_distance(lon_1, lat_1, lon_2, lat_2) / 2 + 50,
                    self.MAX_SEARCH_RADIUS,
                )
                params = {
                    "include_alerts": "false",
                    "limit": self.MAX_STOP_NUM,
                    "format": "json",
                    "lon": (lon_1 + lon_2) / 2,
                    "lat": (lat_1 + lat_2) / 2,
                    "radius": radius,
                    "apikey": self.ak,
                }
                response = requests.get(url=url, params=params)
                if response:
                    res = response.json()
                    stops_without_routes.extend(res["stops"])
                else:
                    print("BAD", i, j)
        one_stop_ids = set(s["onestop_id"] for s in stops_without_routes)
        stops_with_route = []
        # interface address
        for one_stop_id in one_stop_ids:
            params = {
                "include_alerts": "false",
                "format": "json",
                "stop_key": one_stop_id,
                "apikey": self.ak,
            }
            response = requests.get(url=url, params=params)
            if response:
                res = response.json()
                stops_with_route.extend(res["stops"])
        self.stops_with_route = stops_with_route

    def fetch_raw_lines(self):
        GTFS_stop_id2route_ids = defaultdict(list)
        for s in self.stops_with_route:
            if "id" not in s:
                continue
            gtfs_id = s["id"]
            for d in s["route_stops"]:
                if "id" in d["route"]:
                    GTFS_stop_id2route_ids[gtfs_id].append(d["route"]["id"])

        GTFS_route_id2route_info = {}
        url = "https://transit.land/api/v2/rest/routes"
        for _, v in GTFS_stop_id2route_ids.items():
            for gtfs_route_id in v:
                if gtfs_route_id in GTFS_route_id2route_info:
                    continue
                params = {
                    "include_alerts": "false",
                    "format": "json",
                    "route_key": gtfs_route_id,
                    "include_geometry": "true",
                    "apikey": self.ak,
                }
                response = requests.get(url=url, params=params)
                if response:
                    res = response.json()
                    GTFS_route_id2route_info[gtfs_route_id] = res
                else:
                    GTFS_route_id2route_info[gtfs_route_id] = {}
        self.GTFS_route_id2route_info = GTFS_route_id2route_info

    def process_raw_data(self):
        projector = pyproj.Proj(self.projstr)
        all_routes = {}
        stops_from_routes = {}
        for gtfs_route_id, v in self.GTFS_route_id2route_info.items():
            route = v["routes"][0]
            route_type = route["route_type"]
            long_name = route["route_long_name"]
            short_name = route["route_short_name"]
            route_geo = route["geometry"]["coordinates"]
            route_geo_xy = [[projector(*c) for c in l] for l in route_geo]
            rstops = []
            for st in route["route_stops"]:
                st_name = st["stop"].get("stop_name", "")
                st_id = st["stop"].get("id", "")  # gtfs_id
                st_loc = [st["stop"]["geometry"]["coordinates"]]  #  [[lon,lat]]
                kk = (gtfs_route_id, st_name)
                rstops.append(kk)
                if kk in stops_from_routes:
                    continue
                else:
                    stops_from_routes[kk] = {
                        "name": st_name,
                        "geo": st_loc,
                        "geo_xy": [projector(*c) for c in st_loc],
                        "gtfs_id": st_id,
                    }
            all_routes[gtfs_route_id] = {
                "geo": route_geo,
                "geo_xy": route_geo_xy,
                "long_name": long_name,
                "short_name": short_name,
                "stops": list(set(rstops)),
                "route_type": route_type,
            }
        split_routes = {}
        for gtfs_route_id, gtfs_route in all_routes.items():
            geo_xy = gtfs_route["geo_xy"]
            sublines = []
            for i, coords in enumerate(geo_xy):
                stop_points = []
                same_name_stops = defaultdict(list)
                for k in gtfs_route["stops"]:
                    _, st_name = k
                    same_name_stops[st_name].append(
                        {"point": Point(*stops_from_routes[k]["geo_xy"][0]), "key": k}
                    )  # k sl_id,st_name
                line = LineString(coords)
                for _, v in same_name_stops.items():
                    stop_points.append(min(v, key=lambda x: x["point"].distance(line)))
                stop_points = sorted(
                    stop_points, key=lambda x: line.project(x["point"])
                )
                added_stops = set()
                subline_stops = []
                for p in stop_points:
                    _, st_name = p["key"]
                    if st_name in added_stops:
                        continue
                    else:
                        if line.distance(p["point"]) < self.MAX_STOP_DISGATE:
                            added_stops.add(st_name)
                            subline_stops.append(p["key"])
                if len(subline_stops) >= 2:
                    coords_lonlat = gtfs_route["geo"][i]
                    split_geo = cut(
                        line=LineString(coords_lonlat),
                        points=[
                            Point(*stops_from_routes[k]["geo"][0])
                            for k in subline_stops
                        ],
                        projstr=self.projstr,
                    )
                    sublines.append(
                        {
                            "geo": coords_lonlat,
                            "geo_xy": coords,
                            "long_name": gtfs_route["long_name"],
                            "short_name": gtfs_route["short_name"],
                            "stops": subline_stops,
                            "split_geo": split_geo,
                        }
                    )
            if len(sublines) >= 1:
                split_routes[gtfs_route_id] = {
                    "sublines": sublines,
                    "route_type": gtfs_route["route_type"],
                }
        bus_routes = {}
        bus_stations = {}
        subway_routes = {}
        subway_stations = {}
        for k, v in split_routes.items():
            if v["route_type"] == 1:  # metro
                subway_routes[k] = v
                for sl in v["sublines"]:
                    for st in sl["stops"]:
                        subway_stations[st] = stops_from_routes[st]
            elif v["route_type"] == 3:  # bus
                bus_routes[k] = v
                for sl in v["sublines"]:
                    for st in sl["stops"]:
                        bus_stations[st] = stops_from_routes[st]
        self.bus_routes = bus_routes
        self.bus_stations = bus_stations
        self.subway_routes = subway_routes
        self.subway_stations = subway_stations

    def merge_raw_data(self):
        name2ss = defaultdict(list)  # subway stations
        merged_subway_stations = {}
        name2bs = defaultdict(list)  # bus stations
        merged_bus_stations = {}
        for k, v in self.subway_stations.items():
            st_name = v["name"]
            if st_name in name2ss:
                added_in_old_sts = False
                for sts in name2ss[st_name]:
                    if all(
                        get_sta_dis(st, v) < self.SUBWAY_SAME_STA_DIS
                        for (kk, st) in sts
                    ):
                        sts.append((k, v))
                        added_in_old_sts = True
                        break
                if not added_in_old_sts:
                    name2ss[st_name].append([(k, v)])
            else:
                name2ss[st_name].append([(k, v)])
        old_key2new_key = {}
        for st_name, ss in name2ss.items():
            for i, sts in enumerate(ss):
                new_key = st_name + f"_{i}"
                coords = set()
                for old_key, v in sts:
                    old_key2new_key[old_key] = new_key
                    coords.add(tuple(v["geo"][0]))
                merged_geo = merge_geo(list(coords), projstr=self.projstr)
                merged_subway_stations[new_key] = {
                    "name": st_name,
                    "geo": merged_geo,
                }
        for k, v in self.bus_stations.items():
            st_name = v["name"]
            if st_name in name2bs:
                added_in_old_sts = False
                for sts in name2bs[st_name]:
                    if all(
                        get_sta_dis(st, v) < self.BUS_SAME_STA_DIS for (kk, st) in sts
                    ):
                        sts.append((k, v))
                        added_in_old_sts = True
                        break
                if not added_in_old_sts:
                    name2bs[st_name].append([(k, v)])
            else:
                name2bs[st_name].append([(k, v)])
        for st_name, ss in name2bs.items():
            for i, sts in enumerate(ss):
                new_key = st_name + f"_{i}"
                coords = set()
                for old_key, v in sts:
                    old_key2new_key[old_key] = new_key
                    coords.add(tuple(v["geo"][0]))
                merged_geo = merge_geo(list(coords), self.projstr, square_length=50)
                merged_bus_stations[new_key] = {
                    "name": st_name,
                    "geo": merged_geo,
                }
        for k, v in self.bus_routes.items():
            for sl in v["sublines"]:
                sl["merged_stops"] = []
                for st in sl["stops"]:
                    new_key = old_key2new_key[st]
                    sl["merged_stops"].append(new_key)
        for k, v in self.subway_routes.items():
            for sl in v["sublines"]:
                sl["merged_stops"] = []
                for st in sl["stops"]:
                    new_key = old_key2new_key[st]
                    sl["merged_stops"].append(new_key)
        self.old_key2new_key = old_key2new_key
        self.merged_subway_stations = merged_subway_stations
        self.merged_bus_stations = merged_bus_stations

    def get_output_data(self):
        self.fetch_raw_stops()
        self.fetch_raw_lines()
        self.process_raw_data()
        self.merge_raw_data()
        merged_subway_stations = self.merged_subway_stations
        merged_bus_stations = self.merged_bus_stations
        old_key2new_key = self.old_key2new_key
        all_sta_key2int_id = {}
        for ii, (k, v) in enumerate(merged_bus_stations.items()):
            all_sta_key2int_id[(k, "BUS")] = ii
        for jj, (k, v) in enumerate(merged_subway_stations.items()):
            all_sta_key2int_id[(k, "SUBWAY")] = len(merged_bus_stations) + jj
        transport_data = {
            "stations": [],
            "lines": [],
        }
        ii = 0
        all_sta_key2sl_ids = defaultdict(list)
        for k, v in self.subway_routes.items():
            line_data = {"name": str(k), "type": "SUBWAY", "sublines": []}
            for sl in v["sublines"]:
                sl_stop_start_name, sl_stop_end_name = (
                    merged_subway_stations[old_key2new_key[sl["stops"][0]]]["name"],
                    merged_subway_stations[old_key2new_key[sl["stops"][-1]]]["name"],
                )
                line_data["sublines"].append(
                    {
                        "id": ii,
                        "name": sl.get(
                            "long name", f"{sl_stop_start_name}->{sl_stop_end_name}"
                        ),
                        "geo": sl["split_geo"],
                        "stations": [
                            all_sta_key2int_id[(old_key2new_key[i], "SUBWAY")]
                            for i in sl["stops"]
                        ],
                        "schedules": [],
                    }
                )
                for kk in sl["stops"]:
                    all_sta_key2sl_ids[old_key2new_key[kk]].append(ii)
                ii += 1
            transport_data["lines"].append(line_data)
        jj = ii
        for k, v in self.bus_routes.items():
            line_data = {"name": str(k), "type": "BUS", "sublines": []}
            for sl in v["sublines"]:
                sl_stop_start_name, sl_stop_end_name = (
                    merged_bus_stations[old_key2new_key[sl["stops"][0]]]["name"],
                    merged_bus_stations[old_key2new_key[sl["stops"][-1]]]["name"],
                )
                line_data["sublines"].append(
                    {
                        "id": jj,
                        "name": sl.get(
                            "long name", f"{sl_stop_start_name}->{sl_stop_end_name}"
                        ),
                        "geo": sl["split_geo"],
                        "stations": [
                            all_sta_key2int_id[(old_key2new_key[i], "BUS")]
                            for i in sl["stops"]
                        ],
                        "schedules": [],
                    }
                )
                for kk in sl["stops"]:
                    all_sta_key2sl_ids[old_key2new_key[kk]].append(jj)
                jj += 1
            transport_data["lines"].append(line_data)
        for ii, (k, v) in enumerate(merged_bus_stations.items()):
            transport_data["stations"].append(
                {
                    "id": ii,
                    "name": v["name"],
                    "geo": [list(v["geo"])],
                    "type": "BUS",
                    "subline_ids": all_sta_key2sl_ids[k],
                }
            )
        for jj, (k, v) in enumerate(merged_subway_stations.items()):
            transport_data["stations"].append(
                {
                    "id": jj + ii,
                    "name": v["name"],
                    "geo": [list(v["geo"])],
                    "type": "SUBWAY",
                    "subline_ids": all_sta_key2sl_ids[k],
                }
            )
        return transport_data
