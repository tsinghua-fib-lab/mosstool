"""Fetch the subway bus data from transitland"""

import logging
import random
from collections import defaultdict
from datetime import datetime
from math import asin, cos, radians, sin, sqrt
from typing import Any, Optional, Union, cast

import numpy as np
import pyproj
import requests
from shapely.geometry import (LineString, MultiPoint, MultiPolygon, Point,
                              Polygon)

from .._util.line import (clip_line, connect_split_lines, merge_near_xy_points,
                          offset_lane)
from .gtfs_converter import convert_gtfs_to_format

__all__ = [
    "TransitlandPublicTransport",
]


def geo_coords(geo):
    if isinstance(geo, Polygon):
        return list(geo.exterior.coords)
    elif isinstance(geo, MultiPolygon):
        all_coords = []
        for p_geo in geo.geoms:
            all_coords.extend(geo_coords(p_geo))
        return all_coords
    else:
        return list(geo.coords)


def extract_schedules_from_tags(tags: dict[str, Any]) -> list[float]:
    # Extract schedules from tags
    if not "interval" in tags:
        return []
    if not "opening_hours" in tags:
        return []

    def get_schedules(opening_hours: str, interval: str) -> list[float]:
        def _date_time_to_seconds(date_time) -> float:
            return date_time.hour * 3600 + date_time.minute * 60

        list_opening_hours = opening_hours.split(" ")[-1].split(",")
        schedules: list[float] = []
        for time_format in ["%H:%M:%S", "%H:%M", "%M"]:
            try:
                time_interval = _date_time_to_seconds(
                    datetime.strptime(interval, time_format)
                )
                break
            except:
                pass
        else:
            return []
        time_interval = max(1, time_interval)
        for oh in list_opening_hours:
            for time_format in ["%H:%M:%S", "%H:%M", "%M"]:
                try:
                    start, end = oh.split("-")
                    # special case
                    if start == "24:00":
                        start = "23:59"
                    if end == "24:00":
                        end = "23:59"
                    start = _date_time_to_seconds(datetime.strptime(start, time_format))
                    end = _date_time_to_seconds(datetime.strptime(end, time_format))
                    # start-end with time interval
                    while start < end:
                        schedules.append(start)
                        start += time_interval
                    break
                except:
                    pass
        return schedules

    return get_schedules(tags["opening_hours"], tags["interval"])


def _get_headers(referer_url):
    first_num = random.randint(55, 76)
    third_num = random.randint(0, 3800)
    fourth_num = random.randint(0, 140)
    os_type = [
        "(Windows NT 6.1; WOW64)",
        "(Windows NT 10.0; WOW64)",
        "(X11; Linux x86_64)",
        "(Macintosh; Intel Mac OS X 10_14_5)",
    ]
    chrome_version = "Chrome/{}.0.{}.{}".format(first_num, third_num, fourth_num)

    ua = " ".join(
        [
            "Mozilla/5.0",
            random.choice(os_type),
            "AppleWebKit/537.36",
            "(KHTML, like Gecko)",
            chrome_version,
            "Safari/537.36",
        ]
    )
    headers = {"User-Agent": ua, "Referer": referer_url}
    return headers


def cut(
    line: LineString,
    points: list[Point],
    projstr: str,
    reverse_line: Optional[LineString] = None,
) -> list:
    """
    Split routes based on stations.

    Args:
    - line (LineString([(lat lon), (lat lon),]))
    - points (list[Point(lat lon),])
    """
    res = []
    projector = pyproj.Proj(projstr)
    coords_xy = [list(projector(*c)) for c in line.coords]
    points_xy = [Point(list(projector(p.x, p.y))) for p in points]
    line_i = LineString(coordinates=coords_xy)
    reverse_line_i = (
        LineString(coordinates=[list(projector(*c)) for c in reverse_line.coords[::-1]])
        if reverse_line is not None
        else None
    )
    if len(points_xy) >= 2:
        p_start, p_end = points_xy[0], points_xy[-1]
        line_i = clip_line(line_i, p_start, p_end)
    for ii in range(len(points_xy) - 1):
        cut_res = []
        p_0 = points_xy[ii]
        p_1 = points_xy[ii + 1]
        cut_res.append(clip_line(line_i, p_0, p_1))
        cut_res.append(
            clip_line(reverse_line_i, p_0, p_1)
            if reverse_line_i is not None
            else LineString()
        )
        cut_res = [r for r in cut_res if r]
        best_cut_res = cast(
            LineString,
            min(cut_res, key=lambda l: np.abs(l.length - p_0.distance(p_1))).simplify(
                5
            ),
        )
        reverse_best_cut_res = LineString(best_cut_res.coords[::-1])
        if Point(reverse_best_cut_res.coords[0]).distance(p_0) < Point(
            reverse_best_cut_res.coords[-1]
        ).distance(p_0) or Point(reverse_best_cut_res.coords[-1]).distance(p_1) < Point(
            reverse_best_cut_res.coords[-1]
        ).distance(
            p_0
        ):
            res.append(reverse_best_cut_res)
        else:
            res.append(best_cut_res)
    res = [offset_lane(l, -10) for l in res]  # Translate the road
    return [[list(projector(*c, inverse=True)) for c in line.coords] for line in res]


def _output_data_filter(output_data: dict, proj_str: str, sta_dis_gate: float):
    MAX_STRAIGHT_LENGTH = 3000  # meter
    projector = pyproj.Proj(proj_str)
    pt_lines = output_data["lines"]
    pt_stations = {"BUS": {}, "SUBWAY": {}}
    for sta in output_data["stations"]:
        pt_stations[sta["type"]][sta["id"]] = sta
    # update split_geo
    for _, pt_data in enumerate(pt_lines):
        for sl in pt_data["sublines"]:
            pt_type = pt_data["type"]
            new_split_geo = []
            for geo, cur_sta_id, next_sta_id in zip(
                sl["geo"], sl["stations"][:-1], sl["stations"][1:]
            ):
                line_xy = LineString([projector(*c) for c in geo])
                cur_station, next_station = (
                    pt_stations[pt_type][cur_sta_id],
                    pt_stations[pt_type][next_sta_id],
                )
                cur_lon, cur_lat = MultiPoint(cur_station["geo"]).centroid.coords[0][:2]
                cur_p_xy = Point(projector(cur_lon, cur_lat))
                next_lon, next_lat = MultiPoint(next_station["geo"]).centroid.coords[0][
                    :2
                ]
                next_p_xy = Point(projector(next_lon, next_lat))
                sta_dis = cur_p_xy.distance(next_p_xy)
                if line_xy.length < 0.8 * sta_dis or (
                    line_xy.distance(cur_p_xy) > 3 * sta_dis_gate
                    and line_xy.distance(next_p_xy) > 3 * sta_dis_gate
                ):
                    new_geo = [[cur_lon, cur_lat], [next_lon, next_lat]]
                else:
                    new_geo = geo
                    if cur_p_xy.distance(line_xy) > sta_dis_gate:
                        new_geo = [[cur_lon, cur_lat]] + new_geo
                    if next_p_xy.distance(line_xy) > sta_dis_gate:
                        new_geo = new_geo + [[next_lon, next_lat]]
                new_split_geo.append(new_geo)
            sl["geo"] = new_split_geo
    # delete bad sublines
    sta_id2del_slid = defaultdict(set)
    bad_sl_ids_set = set()
    for _, pt_data in enumerate(pt_lines):
        for sl in pt_data["sublines"]:
            pt_type = pt_data["type"]
            sl_id = sl["id"]
            cur_sta_idx = 0
            for geo in sl["geo"]:
                cur_sta_idx += 1
                line_xy = LineString([projector(*c) for c in geo])
                line_xy = line_xy.simplify(5)
                straight_line = LineString([line_xy.coords[ii] for ii in [0, -1]])
                if (
                    line_xy.length > MAX_STRAIGHT_LENGTH
                    and 0.8 < straight_line.length / line_xy.length < 1.2
                ) or straight_line.length / line_xy.length < 1 / 5:
                    break
            for sta_id in sl["stations"][cur_sta_idx:]:
                sta_id2del_slid[sta_id].add(sl_id)
            sl["stations"] = sl["stations"][:cur_sta_idx]
            sl["geo"] = sl["geo"][: cur_sta_idx - 1]
            if not len(sl["stations"]) >= 2:
                bad_sl_ids_set.add(sl_id)
    # update output_data
    ## lines
    new_lines = []
    for _, pt_data in enumerate(pt_lines):
        pt_data["sublines"] = [
            sl for sl in pt_data["sublines"] if sl["id"] not in bad_sl_ids_set
        ]
        if len(pt_data["sublines"]) >= 1:
            new_lines.append(pt_data)
    output_data["lines"] = new_lines
    ## stations
    new_stations = []
    for sta in output_data["stations"]:
        sta["subline_ids"] = [
            slid
            for slid in sta["subline_ids"]
            if slid not in sta_id2del_slid[sta["id"]]
        ]
        if len(sta["subline_ids"]) >= 1:
            new_stations.append(sta)
    output_data["stations"] = new_stations


def merge_geo(coord, proj_str, square_length=350):
    SQUARE_SIDE_LENGTH = (
        square_length  # The side length of the rectangle formed by a single point POI
    )
    projector = pyproj.Proj(proj_str)
    if len(coord) > 1:
        geo = MultiPoint(coord).minimum_rotated_rectangle
        # Expand geo into a large square
        coords = np.array(geo_coords(geo))
        min_x, min_y = coords.min(axis=0)
        max_x, max_y = coords.max(axis=0)
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
    LON1: Union[float, tuple[float, float]],
    LAT1: Union[float, tuple[float, float]],
    LON2: Optional[float] = None,
    LAT2: Optional[float] = None,
):
    """
    Distance between GPS points (m)
    """
    if LON2 == None:  # The input is [lon1,lat1], [lon2,lat2]
        lon1, lat1 = cast(tuple[float, float], LON1)
        lon2, lat2 = cast(tuple[float, float], LAT1)
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


class TransitlandPublicTransport:
    """
    Process Transitland raw data to Public Transport data as geojson format files
    """

    def __init__(
        self,
        proj_str: str,
        max_longitude: Optional[float] = None,
        min_longitude: Optional[float] = None,
        max_latitude: Optional[float] = None,
        min_latitude: Optional[float] = None,
        transitland_ak: Optional[str] = None,
        proxies: Optional[dict[str, str]] = None,
        wikipedia_name: Optional[str] = None,
        from_osm: bool = False,
        gtfs_dir: Optional[str] = None,
    ):
        self.proxies = proxies
        self.wikipedia_name = wikipedia_name
        self.from_osm = from_osm
        self.gtfs_dir = gtfs_dir
        if transitland_ak is None:
            self.from_osm = True
            logging.info("No transitland ak provided! Fetching from OSM")
        self.MAX_SEARCH_RADIUS = 10000  # API’s maximum search radius
        self.MIN_PT_LENGTH = 200  # minimum length of PT route
        self.MAX_STOP_NUM = 100  # The maximum number of sites for a single API search
        self.MAX_STOP_DISGATE = (
            50  # Anything beyond this distance is not considered a stop for this line.
        )
        self.OSM_STOP_DISGATE = 15
        self.SUBWAY_SAME_STA_DIS = (
            800  # The merge distance threshold of subway stations with the same name
        )
        self.BUS_SAME_STA_DIS = 30
        MIN_UNIT_NUM = 16  # Minimum number of search blocks
        self.projstr = proj_str
        if not gtfs_dir:
            assert min_longitude is not None and max_longitude is not None and min_latitude is not None and max_latitude is not None, "min_longitude, max_longitude, min_latitude, max_latitude should be provided if gtfs_dir is not provided"
            self.bbox = (
                min_latitude,
                min_longitude,
                max_latitude,
                max_longitude,
            )
            lon1, lat1 = min_longitude, min_latitude
            lon2, lat2 = max_longitude, max_latitude
            self.lon_center = (lon1 + lon2) / 2
            self.lat_center = (lat1 + lat2) / 2
            self.ak = transitland_ak
            self.unit = max(
                MIN_UNIT_NUM,
                int(gps_distance(lon1, lat1, lon2, lat2) / self.MAX_SEARCH_RADIUS / 2 + 1),
            )  # unit-1 * unit-1 large raster
            self.lon_partition = [
                round(x, 6) for x in list(np.linspace(lon1, lon2, self.unit))
            ]  # Used to calculate search radius
            self.lat_partition = [
                round(y, 6) for y in list(np.linspace(lat1, lat2, self.unit))
            ]

    def _query_raw_data_from_osm(self):
        """
        Get raw data from OSM API
        OSM query language: https://wiki.openstreetmap.org/wiki/Overpass_API/Language_Guide
        Can be run and visualized in real time at https://overpass-turbo.eu/
        """
        logging.info("Fetching raw stops from OpenStreetMap")
        bbox_str = ",".join(str(i) for i in self.bbox)
        query_header = f"[out:json][timeout:120][bbox:{bbox_str}];"
        area_wikipedia_name = self.wikipedia_name
        if area_wikipedia_name is not None:
            query_header += f'area[wikipedia="{area_wikipedia_name}"]->.searchArea;'
        osm_data = None
        for _ in range(3):  # retry 3 times
            try:
                query_body_raw = [
                    (
                        "way",
                        "[route=bus]",
                    ),
                    (
                        "rel",
                        "[route=bus]",
                    ),
                    (
                        "way",
                        "[route=trolleybus]",
                    ),
                    (
                        "rel",
                        "[route=trolleybus]",
                    ),
                    (
                        "way",
                        "[route=subway]",
                    ),
                    (
                        "rel",
                        "[route=subway]",
                    ),
                    (
                        "node",
                        "[highway=bus_stop]",
                    ),
                ]
                query_body = ""
                for obj, args in query_body_raw:
                    area = (
                        "(area.searchArea)" if area_wikipedia_name is not None else ""
                    )
                    query_body += obj + area + args + ";"
                query_body = "(" + query_body + ");"
                query = query_header + query_body + "(._;>;);" + "out body;"
                logging.info(f"{query}")
                osm_data = requests.get(
                    "http://overpass-api.de/api/interpreter?data=" + query,
                    proxies=self.proxies,
                ).json()["elements"]
                break
            except Exception as e:
                logging.warning(f"Exception when querying OSM data {e}")
                logging.warning("No response from OSM, Please try again later!")
        if osm_data is None:
            raise Exception("No BUS response from OSM!")
        self._osm_data = osm_data

    def _process_raw_data_from_osm(self):
        _nodes = {}
        _ways = {}
        _rels = {}
        _station_node_ids = set()
        # highway=bus_stop, public_transport=platform,
        projector = pyproj.Proj(self.projstr)
        _osm_dict = defaultdict(list)
        for d in self._osm_data:
            data_type = d["type"]
            _osm_dict[data_type].append(d)
        # stations
        for d in _osm_dict["node"]:
            d_id = d["id"]
            node_name = d_id
            if "tags" in d:
                d_tags = d["tags"]
                if "name" in d_tags:
                    node_name = d_tags["name"]
                elif "wikipedia" in d_tags:
                    node_name = d_tags["wikipedia"]
                elif "name:en" in d_tags:
                    node_name = d_tags["name:en"]
                if (
                    d_tags.get("highway", "") in {"bus_stop", "subway", "station"}
                    or d_tags.get("public_transport", "")
                    in {"platform", "station", "bus", "subway"}
                    or d_tags.get("bus", "") in {"yes"}
                    or d_tags.get("amenity", "")
                    in {"bus_station", "subway_station", "bus", "subway"}
                    or "station" in d_tags
                ):
                    _station_node_ids.add(d_id)
            _nodes[d_id] = {
                "pos": (
                    d["lon"],
                    d["lat"],
                ),
                "pos_xy": list(projector(d["lon"], d["lat"])),
                "name": str(node_name),
                "orig_node": d,
            }
        # lines
        for d in _osm_dict["way"]:
            if "nodes" in d:
                way_nodes = [_nodes[nid] for nid in d["nodes"]]
                pos = [n["pos"] for n in way_nodes]
                pos_xy = [n["pos_xy"] for n in way_nodes]
                way_name = f"{way_nodes[0]['name']}->{way_nodes[-1]['name']}"
                d_tags = d.get("tags", {})
                if "name" in d_tags:
                    way_name = d_tags["name"]
                elif "wikipedia" in d_tags:
                    way_name = d_tags["wikipedia"]
                elif "name:en" in d_tags:
                    way_name = d_tags["name:en"]
                tags = d_tags
                _ways[d["id"]] = {
                    "pos": pos,
                    "pos_xy": pos_xy,
                    "name": str(way_name),
                    "tags": tags,
                    "node_ids": d["nodes"],
                }
        # relations
        for d in _osm_dict["relation"]:
            rel_way_ids = [mm["ref"] for mm in d["members"] if mm["type"] == "way"]
            rel_node_ids = [mm["ref"] for mm in d["members"] if mm["type"] == "node"]
            d_id = d["id"]
            d_tags = d.get("tags", {})
            rel_name = str(d_id)
            if "name" in d_tags:
                rel_name = d_tags["name"]
            elif "description" in d_tags:
                rel_name = d_tags["description"]
            if not rel_way_ids and not rel_node_ids:
                continue
            _rels[d_id] = {
                "way_ids": rel_way_ids,
                "node_ids": rel_node_ids,
                "name": rel_name,
                "tags": d_tags,
            }
        GTFS_format_data = {}
        ref_id2routes = defaultdict(list)
        for rel_id, rel in _rels.items():
            ref_id = rel.get("tags", {}).get("ref")
            rel_way_ids = rel["way_ids"]
            rel_ways = [_ways[wid] for wid in rel_way_ids]
            rel_way_shapely = [
                cast(LineString, LineString(way["pos_xy"]).simplify(5))
                for way in rel_ways
            ]
            rel_node_ids = [node_id for way in rel_ways for node_id in way["node_ids"]]
            station_ids = rel["node_ids"]
            # station nodes
            if len(station_ids) <= 2:
                station_ids = [nid for nid in rel_node_ids if nid in _station_node_ids]
            for idx in range(len(station_ids) - 1):
                cur_id = station_ids[idx]
                cur_pos = _nodes[cur_id]["pos"]
                next_id = station_ids[idx + 1]
                next_pos = _nodes[next_id]["pos"]
            # way positions
            rel_coords = []
            station_pos_xy = [_nodes[nid]["pos_xy"] for nid in station_ids]
            if len(station_pos_xy) < 2:
                continue
            merged_station_pos_xy = merge_near_xy_points(station_pos_xy)
            rel_sta_ids_dict = defaultdict(list)
            for coords, nid in zip(merged_station_pos_xy, station_ids):
                rel_sta_ids_dict[tuple(coords)].append(nid)
            rel_station_ids = list(
                {rel_sta_ids_dict[tuple(coords)][0] for coords in merged_station_pos_xy}
            )
            if len(rel_station_ids) <= 2 or len(rel_way_shapely) < 1:
                continue
            straight_station_line = LineString(
                [_nodes[rel_station_ids[idx]]["pos_xy"] for idx in [0, -1]]
            )
            rel_station_ids = sorted(
                rel_station_ids,
                key=lambda sta_id: straight_station_line.project(
                    Point(_nodes[sta_id]["pos_xy"])
                ),
            )
            rel_coords_xy = connect_split_lines(
                lines=rel_way_shapely,
                start_point=Point(_nodes[rel_station_ids[-1]]["pos_xy"]),
            )  # xy
            # no enough way shapes
            if len(rel_coords_xy) <= 2:
                continue
            # add mid point for rel_coords
            for idx, (cur_pos_xy, next_pos_xy) in enumerate(
                zip(rel_coords_xy[:-1], rel_coords_xy[1:])
            ):
                cur_pos = projector(*cur_pos_xy, inverse=True)  # lon,lat
                next_pos = projector(*next_pos_xy, inverse=True)  # lon,lat
                avg_pos = [(cur_pos[j] + next_pos[j]) / 2 for j in range(len(cur_pos))]
                rel_coords.append(cur_pos)
                rel_coords.append(avg_pos)
                if idx == len(rel_coords_xy) - 2:
                    rel_coords.append(next_pos)
            if any(
                [
                    "railway" in way["tags"]
                    and way["tags"]["railway"] == "subway"
                    or "name" in way["tags"]
                    and any(
                        k in way["tags"]["name"]
                        for k in ["subway", "Subway", "metro", "地铁"]
                    )
                    for way in rel_ways
                ]
            ):
                route_type = 1  # subway
            else:
                route_type = 3  # bus
            route_coords = [rel_coords]
            if ref_id is None:
                route_coords += [rel_coords[::-1]]
            # store schedules in route
            route = {
                "route_type": route_type,
                "route_long_name": rel["name"],
                "route_short_name": rel["name"],
                "geometry": {
                    "coordinates": route_coords,
                },
                "schedules": extract_schedules_from_tags(rel.get("tags", {})),
                "route_stops": [
                    {
                        "stop": {
                            "stop_name": _nodes[node_id]["name"],
                            "id": node_id,
                            "geometry": {
                                "coordinates": _nodes[node_id]["pos"],
                            },
                        }
                    }
                    for node_id in rel_station_ids
                ],
            }
            if ref_id is None:
                GTFS_format_data[rel_id] = {
                    "routes": [route],
                    "schedules": route.get("schedules", []),
                }
            else:
                ref_id2routes[ref_id].append(route)
        for ref_id, routes in ref_id2routes.items():
            main_route = routes[0]
            for r in routes[1:]:
                main_route["geometry"]["coordinates"] += r["geometry"]["coordinates"]
            route_schedules = [r.get("schedules", []) for r in routes]
            route_schedules = [r for r in route_schedules if r]
            GTFS_format_data[ref_id] = {
                "routes": [main_route],
                "schedules": route_schedules[0] if route_schedules else [],
            }
        self.GTFS_route_id2route_info = GTFS_format_data

    def _fetch_raw_stops(self):
        url = "https://transit.land/api/v2/rest/stops"
        stops_without_routes = []
        logging.info("Fetching raw stops from Transitland")
        for i in range(0, self.unit - 1):  # Horizontal grid index
            for j in range(0, self.unit - 1):  # Vertical grid index
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
                response = requests.get(
                    url=url, params=params, headers=_get_headers(url)
                )
                if response:
                    res = response.json()
                    stops_without_routes.extend(res["stops"])
                else:
                    # no lines in this area
                    continue
        one_stop_ids = set(s["onestop_id"] for s in stops_without_routes)
        stops_with_route = []
        # interface address
        for one_stop_id in one_stop_ids:
            for _ in range(3):
                try:
                    params = {
                        "include_alerts": "false",
                        "format": "json",
                        "stop_key": one_stop_id,
                        "apikey": self.ak,
                    }
                    response = requests.get(
                        url=url, params=params, headers=_get_headers(url)
                    )
                    if response:
                        res = response.json()
                        stops_with_route.extend(res["stops"])
                    break
                except:
                    continue
        self.stops_with_route = stops_with_route

    def _fetch_raw_lines(self):
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
                for _ in range(3):
                    try:
                        if (
                            gtfs_route_id in GTFS_route_id2route_info
                            and len(GTFS_route_id2route_info[gtfs_route_id]) > 0
                        ):
                            continue
                        params = {
                            "include_alerts": "false",
                            "format": "json",
                            "route_key": gtfs_route_id,
                            "include_geometry": "true",
                            "apikey": self.ak,
                        }
                        response = requests.get(
                            url=url, params=params, headers=_get_headers(url)
                        )
                        if response:
                            res = response.json()
                            GTFS_route_id2route_info[gtfs_route_id] = res
                        else:
                            GTFS_route_id2route_info[gtfs_route_id] = {}
                        break
                    except:
                        continue
        self.GTFS_route_id2route_info = GTFS_route_id2route_info

    def process_raw_data(self):
        projector = pyproj.Proj(self.projstr)
        all_routes = {}
        stops_from_routes = {}
        for gtfs_route_id, v in self.GTFS_route_id2route_info.items():
            if len(v.get("routes", [])) == 0:
                continue
            route = v["routes"][0]
            route_type = route["route_type"]
            long_name = route["route_long_name"]
            short_name = route["route_short_name"]
            route_geo = route["geometry"]["coordinates"]
            route_geo_xy = [[projector(*c) for c in l] for l in route_geo]
            if len(route_geo) == 0:
                continue
            rstops = []
            for st in route["route_stops"]:
                st_name = st["stop"].get("stop_name", "")
                st_id = st["stop"].get("id", "")  # gtfs_id
                if not len(st["stop"]["geometry"]["coordinates"]) == 2:
                    continue
                st_loc = [st["stop"]["geometry"]["coordinates"]]  #  [[lon,lat]]
                stop_key = (gtfs_route_id, st_name)
                rstops.append(stop_key)
                if stop_key in stops_from_routes:
                    continue
                else:
                    stops_from_routes[stop_key] = {
                        "name": st_name,
                        "geo": st_loc,
                        "geo_xy": [projector(*c) for c in st_loc],
                        "gtfs_id": st_id,
                    }
            if len(rstops) == 0:
                continue
            all_routes[gtfs_route_id] = {
                "geo": route_geo,
                "geo_xy": route_geo_xy,
                "long_name": long_name,
                "short_name": short_name,
                "stops": list(set(rstops)),
                "route_type": route_type,
                "schedules": v.get("schedules", []),
            }
        split_routes = {}
        for gtfs_route_id, gtfs_route in all_routes.items():
            try:
                geo_xy = gtfs_route["geo_xy"]
                sublines = []
                for i, coords in enumerate(geo_xy):
                    stop_points = []
                    same_name_stops = defaultdict(list)
                    for k in gtfs_route["stops"]:
                        _, st_name = k
                        same_name_stops[st_name].append(
                            {
                                "point": Point(*stops_from_routes[k]["geo_xy"][0]),
                                "key": k,
                            }
                        )  # k sl_id,st_name
                    line = LineString(coords)
                    line = line.simplify(5)
                    for _, v in same_name_stops.items():
                        stop_points.append(
                            min(v, key=lambda x: x["point"].distance(line))
                        )
                    stop_points = sorted(
                        stop_points, key=lambda x: line.project(x["point"])
                    )
                    # stop_points = sorted(
                    #     stop_points, key=lambda x: LineString([coords[0],coords[-1]]).project(x["point"])
                    # )
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
                        other_coords_lonlats = [
                            ccs
                            for ii, ccs in enumerate(gtfs_route["geo"])
                            if not ii == i
                        ]
                        if len(other_coords_lonlats) > 0:
                            reverse_line = LineString(other_coords_lonlats[0])
                        else:
                            reverse_line = None
                        split_geo = cut(
                            line=LineString(coords_lonlat),
                            reverse_line=reverse_line,
                            points=[
                                Point(*stops_from_routes[k]["geo"][0])
                                for k in subline_stops
                            ],
                            projstr=self.projstr,
                        )
                        # too short
                        if (
                            np.sum(
                                [
                                    LineString([projector(*c) for c in lonlats]).length
                                    for lonlats in split_geo
                                ]
                            )
                            < self.MIN_PT_LENGTH
                        ):
                            continue
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
                        "schedules": gtfs_route.get("schedules", []),
                    }
            except:
                continue
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
            elif v["route_type"] in {3, 11}:  # bus, trolleybus
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
                        get_sta_dis(st, v) < self.SUBWAY_SAME_STA_DIS - 1e99
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
                merged_geo = merge_geo(list(coords), proj_str=self.projstr)
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
                        get_sta_dis(st, v) < self.BUS_SAME_STA_DIS - 1e99
                        for (kk, st) in sts
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
        if self.gtfs_dir:
            # from GTFS data
            self.GTFS_route_id2route_info = convert_gtfs_to_format(self.gtfs_dir)
        else:
            if self.from_osm:
                self._query_raw_data_from_osm()
                self._process_raw_data_from_osm()
            else:
                self._fetch_raw_stops()
                self._fetch_raw_lines()
        self.process_raw_data()
        self.merge_raw_data()
        merged_subway_stations = self.merged_subway_stations
        merged_bus_stations = self.merged_bus_stations
        old_key2new_key = self.old_key2new_key
        all_sta_key2int_id = {}
        _all_bus_sta_ids = set({-1})
        for ii, (k, v) in enumerate(merged_bus_stations.items()):
            all_sta_key2int_id[(k, "BUS")] = ii
            _all_bus_sta_ids.add(ii)
        _subway_sta_id_start = max(_all_bus_sta_ids) + 1
        for jj, (k, v) in enumerate(merged_subway_stations.items()):
            all_sta_key2int_id[(k, "SUBWAY")] = _subway_sta_id_start + jj
        transport_data = {
            "stations": [],
            "lines": [],
        }
        _bus_sl_id = -1
        all_sta_key2sl_ids = defaultdict(list)
        for k, v in self.subway_routes.items():
            line_data = {"name": str(k), "type": "SUBWAY", "sublines": []}
            for sl in v["sublines"]:
                _bus_sl_id += 1
                sl_stop_start_name, sl_stop_end_name = (
                    merged_subway_stations[old_key2new_key[sl["stops"][0]]]["name"],
                    merged_subway_stations[old_key2new_key[sl["stops"][-1]]]["name"],
                )
                line_data["sublines"].append(
                    {
                        "id": _bus_sl_id,
                        "name": sl.get(
                            "long_name", f"{sl_stop_start_name}->{sl_stop_end_name}"
                        ),
                        "geo": sl["split_geo"],
                        "stations": [
                            all_sta_key2int_id[(old_key2new_key[i], "SUBWAY")]
                            for i in sl["stops"]
                        ],
                        "schedules": v.get("schedules", []),
                    }
                )
                for kk in sl["stops"]:
                    all_sta_key2sl_ids[old_key2new_key[kk]].append(_bus_sl_id)
            transport_data["lines"].append(line_data)
        _subway_sl_id = _bus_sl_id
        for k, v in self.bus_routes.items():
            line_data = {"name": str(k), "type": "BUS", "sublines": []}
            for sl in v["sublines"]:
                _subway_sl_id += 1
                sl_stop_start_name, sl_stop_end_name = (
                    merged_bus_stations[old_key2new_key[sl["stops"][0]]]["name"],
                    merged_bus_stations[old_key2new_key[sl["stops"][-1]]]["name"],
                )
                line_data["sublines"].append(
                    {
                        "id": _subway_sl_id,
                        "name": sl.get(
                            "long_name", f"{sl_stop_start_name}->{sl_stop_end_name}"
                        ),
                        "geo": sl["split_geo"],
                        "stations": [
                            all_sta_key2int_id[(old_key2new_key[i], "BUS")]
                            for i in sl["stops"]
                        ],
                        "schedules": v.get("schedules", []),
                    }
                )
                for kk in sl["stops"]:
                    all_sta_key2sl_ids[old_key2new_key[kk]].append(_subway_sl_id)
            transport_data["lines"].append(line_data)
        for ii, (k, v) in enumerate(merged_bus_stations.items()):
            transport_data["stations"].append(
                {
                    "id": all_sta_key2int_id[(k, "BUS")],
                    "name": v["name"],
                    "geo": list(v["geo"]),
                    "type": "BUS",
                    "subline_ids": all_sta_key2sl_ids[k],
                }
            )
        for jj, (k, v) in enumerate(merged_subway_stations.items()):
            transport_data["stations"].append(
                {
                    "id": all_sta_key2int_id[(k, "SUBWAY")],
                    "name": v["name"],
                    "geo": list(v["geo"]),
                    "type": "SUBWAY",
                    "subline_ids": all_sta_key2sl_ids[k],
                }
            )
        # output check
        _output_data_filter(transport_data, self.projstr, self.MAX_STOP_DISGATE)
        return transport_data
