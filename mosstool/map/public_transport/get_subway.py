"""Fetch the subway data from amap"""

import json
import sys
from collections import defaultdict
from math import atan2

import numpy as np
import pyproj
import requests
from coord_convert.transform import gcj2wgs
from lxml import etree
from shapely.geometry import LineString, MultiPoint, Point, Polygon

from .._map_util.aoiutils import geo_coords

__all__ = [
    "AmapSubway",
]


class AmapSubway:
    """
    Process amap raw data to Public Transport data as geojson format files
    """

    def __init__(
        self,
        city_name_en_us: str,
        proj_str: str,
        amap_ak: str,
    ):
        self.city_name_en_us = city_name_en_us
        self.amap_ak = amap_ak
        self.projector = pyproj.Proj(proj_str)
        self.subway_lines = {}
        self.square_length = 350

    def _fetch_raw_data(self):
        page_url = "http://map.amap.com/subway/index.html?&1100"
        data_url = "http://map.amap.com/service/subway?srhdata="
        header = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36"
        }

        def fetch_city(url, header):
            r = requests.get(url, header)
            html = r.content
            element = etree.HTML(html)  # type:ignore
            options = element.xpath("//a[contains(@class, 'city')]")
            all_city_names = []
            for option in options:
                city = {
                    "id": option.get("id"),
                    "name": option.get("cityname"),
                    "text": option.text,
                }
                all_city_names.append(city["name"])
                if city["name"] == self.city_name_en_us:
                    return city, all_city_names
            return {}, all_city_names

        city, all_city_names = fetch_city(page_url, header)
        if not city:
            sys.exit(
                f"Unavailable city name {self.city_name_en_us}, should be in {all_city_names}"
            )
        url = data_url + "{}_drw_{}.json".format(city["id"], city["name"])
        json_str = requests.get(url).text
        raw_data = json.loads(json_str)
        for line in raw_data["l"]:
            line_id = line["ls"]
            line_name = line["ln"]
            sub_line_amap_ids = [lid for lid in line["li"].split("|")]
            self.subway_lines[line_id] = {
                "name": line_name,
                "sub_line_amap_ids": sub_line_amap_ids,
            }

    def _fetch_amap_lines(self):
        def get_amap_response(amap_id):
            url = "https://restapi.amap.com/v3/bus/lineid?parameters"
            params = {
                "id": amap_id,
                "output": "json",
                "key": self.amap_ak,
                "extensions": "all",
            }
            amap_info = []
            response = requests.get(url=url, params=params)
            if response:
                res = response.json()
                if res["status"] == 1 or res["status"] == "1":
                    amap_info = res
            return amap_info

        def offset_lane_lonlat(coords, distance):
            def offset_lane(line: LineString, distance: float) -> LineString:
                offset_line = line.offset_curve(distance)
                if offset_line:
                    return offset_line  # type: ignore
                line_vec = np.array(line.coords[-1]) - np.array(line.coords[-2])
                line_angle = atan2(*line_vec)
                if distance < 0:
                    vec = np.array(
                        [np.cos(line_angle - np.pi / 2), np.sin(line_angle - np.pi / 2)]
                    )
                else:
                    vec = np.array(
                        [np.cos(line_angle + np.pi / 2), np.sin(line_angle + np.pi / 2)]
                    )
                offset_vec = vec / np.linalg.norm(vec) * np.abs(distance)
                coords_xy = [c + offset_vec for c in line.coords]
                return LineString(coords_xy)

            coords_xy = [list(self.projector(*c)) for c in coords]
            line = offset_lane(LineString(coords_xy), distance)
            return [list(self.projector(*c, inverse=True)) for c in line.coords]

        subway_stations = {}
        subway_lines = {}

        def get_coords(ss: str):
            coords = []
            for c in ss.split(";"):
                lon, lat = c.split(",")
                coords.append((np.float64(lon), np.float64(lat)))
            return coords

        amap_id2info = {}
        for k, v in self.subway_lines.items():
            lname = v["name"]
            sublines = []
            sub_line_amap_ids = v["sub_line_amap_ids"]
            for amap_id in sub_line_amap_ids:
                sl = {}
                info = get_amap_response(amap_id)
                if not info:
                    continue
                amap_id2info[amap_id] = info
                sline = info["buslines"][0]  # type:ignore
                bstops = sline["busstops"]
                sl_name = sline["name"]
                sl["stops"] = []
                sl["stop_names"] = []
                line_str = sline["polyline"]
                for st in bstops:
                    st_id = st["id"]
                    st_name = st["name"]
                    if not st_id:
                        st_id = st_name
                    sl["stops"].append((sl_name, st_id))
                    sl["stop_names"].append(st_name)
                    st_loc = gcj2wgs(*(get_coords(st["location"])[0]))
                    subway_stations[(sl_name, st_id)] = {
                        "name": st_name,
                        "geo": st_loc,
                        "sub_line": sl_name,
                    }
                sl["name"] = sl_name
                split_str = []
                added_set = set()
                path_str = bstops[0]["location"]
                i = 1
                for coord_str in line_str.split(";"):
                    stop_str = bstops[i]["location"]
                    path_str += ";" + coord_str
                    added_set.add(coord_str)
                    if stop_str == coord_str:
                        if not path_str.split(";")[0] == path_str.split(";")[-1]:
                            split_str.append(path_str)
                            i += 1
                            if i >= len(bstops):
                                break
                        path_str = coord_str
                sl["geo"] = [[gcj2wgs(*c) for c in get_coords(l)] for l in split_str]
                sl["offset_geo"] = [
                    offset_lane_lonlat(coords, -20) for coords in sl["geo"]
                ]
                sl["split_str"] = split_str
                sl["all_geo"] = [gcj2wgs(*c) for c in get_coords(line_str)]
                sublines.append(sl)
            subway_lines[k] = {
                "name": lname,
                "sublines": sublines,
            }
        return (subway_stations, subway_lines)

    def _merge_stations(self, subway_stations):
        def merge_geo(coord):
            if len(coord) > 1:
                geo = MultiPoint(coord).minimum_rotated_rectangle
                # Extend geo to a large square
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
                        list(self.projector(*p))
                        for p in [
                            bottom_left,
                            top_left,
                            top_right,
                            bottom_right,
                            bottom_left,
                        ]
                    ]
                )
                if geo_xy.area < self.square_length * self.square_length:
                    x, y = self.projector(0.5 * (min_x + max_x), 0.5 * (min_y + max_y))
                    half_side = self.square_length / 2
                    bottom_left = (x - half_side, y - half_side)
                    top_left = (x - half_side, y + half_side)
                    top_right = (x + half_side, y + half_side)
                    bottom_right = (x + half_side, y - half_side)
                    geo = [
                        list(self.projector(*p, inverse=True))
                        for p in [
                            bottom_left,
                            top_left,
                            top_right,
                            bottom_right,
                            bottom_left,
                        ]
                    ]
            else:
                geo = Point(coord)
                center = geo
                x, y = self.projector(center.x, center.y)
                half_side = self.square_length / 2
                bottom_left = (x - half_side, y - half_side)
                top_left = (x - half_side, y + half_side)
                top_right = (x + half_side, y + half_side)
                bottom_right = (x + half_side, y - half_side)
                geo = [
                    list(self.projector(*p, inverse=True))
                    for p in [
                        bottom_left,
                        top_left,
                        top_right,
                        bottom_right,
                        bottom_left,
                    ]
                ]
            return geo

        self.name2sts = {}  # subway name->info
        for k, v in subway_stations.items():
            sl_name, _ = k
            st_name = v["name"]
            st_loc = v["geo"]
            if st_name in self.name2sts:
                st = self.name2sts[st_name]
                st["sub_line"].append(sl_name)
                st["geo"].add(st_loc)
            else:
                self.name2sts[st_name] = {
                    "name": st_name,
                    "sub_line": [sl_name],
                    "geo": {st_loc},
                }
        for k, v in self.name2sts.items():
            v["geo"] = merge_geo(list(v["geo"]))

    def get_output_data(self):
        self._fetch_raw_data()
        (subway_stations, subway_lines) = self._fetch_amap_lines()
        self._merge_stations(subway_stations)
        sta2id = {k: i for i, (k, _) in enumerate(self.name2sts.items())}
        subway_data = {
            "stations": [],
            "lines": [],
        }
        ii = 0
        sl2id = {}
        sta2sl = defaultdict(list)
        for k, v in subway_lines.items():
            line_data = {"name": k, "type": "SUBWAY", "sublines": []}
            for sl in v["sublines"]:
                if "stop_names" not in sl:
                    continue
                line_data["sublines"].append(
                    {
                        "id": ii,
                        "name": sl["name"],
                        "geo": sl["geo"],
                        "stations": [sta2id[i] for i in sl["stop_names"]],
                        "schedules": [],
                    }
                )
                for kk in sl["stop_names"]:
                    sta2sl[kk].append(ii)
                sl2id[sl["name"]] = ii
                ii += 1
            if len(line_data["sublines"]) == 0:
                continue
            subway_data["lines"].append(line_data)
        for ii, (k, v) in enumerate(self.name2sts.items()):
            subway_data["stations"].append(
                {
                    "id": ii,
                    "name": v["name"],
                    "geo": list(v["geo"]),
                    "type": "SUBWAY",
                    "subline_ids": sta2sl[k],
                }
            )
        return subway_data
