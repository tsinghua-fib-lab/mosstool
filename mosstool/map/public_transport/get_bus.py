"""Fetch the bus data from amap"""

import random
from collections import defaultdict

import Levenshtein
import numpy as np
import requests
from bs4 import BeautifulSoup
from coord_convert.transform import gcj2wgs

__all__ = [
    "AmapBus",
]


# UA pool
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


class AmapBus:
    """
    Process amap raw data to Public Transport data as geojson format files
    """

    def __init__(
        self,
        city_name_en_us: str,
        city_name_zh_cn: str,
        bus_heads: str,
        amap_ak: str,
    ):
        self.city_name_en_us = city_name_en_us
        self.city_name_zh_cn = city_name_zh_cn
        self.bus_heads = bus_heads.split("_")
        self.amap_ak = amap_ak
        self.bus_lines = {}
        self.all_sta_names = set()
        self.sta2bv = {}  # used to find line
        self.sl2amap = (
            {}
        )  # Contains line latitude and longitude and all site information

    def _fetch_raw_data(self):
        url = f"https://{self.city_name_en_us}.8684.cn"
        # bus_heads = [str(i) for i in range(1, 10)] + ['B', 'C', 'D', 'F']
        for bus in self.bus_heads:
            bus_single_url = (
                url + "/list" + bus
            )  # corresponds to the bus at the beginning
            resp = requests.get(bus_single_url, headers=_get_headers(url))
            bus_main_html = BeautifulSoup(resp.text, "html.parser")
            try:
                bus_route_list = bus_main_html.find(
                    "div", class_="list clearfix"
                ).find_all(  # type:ignore
                    "a"
                )
            except:
                continue
            route_href = (
                []
            )  # Only access the link of the route -/x_322e21c5, the complete url needs to be spliced
            # url + route_href[0] = https://beijing.8684.cn/x_322e21c5
            for single_route in bus_route_list:
                route_href.append(single_route.get("href"))

            for href in route_href:
                route_url = url + href
                bus_detail = requests.get(
                    route_url, headers=_get_headers(bus_single_url)
                )
                bus_detail_html = BeautifulSoup(bus_detail.text, "html.parser")
                try:
                    bus_info = bus_detail_html.find("div", class_="info")
                    detail = bus_info.get_text("#").split("#")[:6]  # type:ignore
                    route_total = bus_detail_html.find_all("div", "bus-excerpt mb15")
                    bus_lzlist = bus_detail_html.find_all("div", "bus-lzlist mb15")
                    line_name = detail[0]
                    sub_lines = {}
                    for route, bus_ls in zip(route_total, bus_lzlist):
                        trip = route.find("div", "trip").get_text()
                        start, end = trip.split(
                            "—"
                        )  # Get the names of the starting and ending stations
                        li_list = [
                            li.get_text() for li in bus_ls.find_all("a")
                        ]  # Get the passing site
                        tmp = [
                            li
                            for idx, li in enumerate(li_list[1:-1])
                            if li != start and li != end
                        ]  # Remove the first and last two sites
                        tmp = (
                            [li_list[0]] + tmp + [li_list[-1]]
                        )  # Add the starting station and the ending station
                        sub_lines[trip] = {
                            "stations": tmp,
                        }
                        self.all_sta_names.update(set(tmp))
                    self.bus_lines[line_name] = {
                        "name": line_name,
                        "sub_lines": sub_lines,
                        "details": detail,
                    }
                except:
                    continue

    def _fetch_amap_positions(self):
        url = "https://restapi.amap.com/v3/place/text"
        amap_api_res = {}
        for sta in self.all_sta_names:
            params = {
                "keywords": f"{sta}公交站",
                "city": self.city_name_zh_cn,
                "types": "150700",  # bus stations
                "city_limit": "true",
                "output": "json",
                "key": self.amap_ak,
            }
            response = requests.get(url=url, params=params)
            if response:
                res = response.json()
                amap_api_res[sta] = res
        for sta, v in amap_api_res.items():
            pois = v["pois"]
            for p in pois:
                if "id" in p and p["id"].startswith("BV"):
                    self.sta2bv[sta] = p["id"]
                    break

    def _fetch_amap_lines(self):
        # fetch info according to BVcode, mainly for line id
        bv2amap = {}
        url = "https://restapi.amap.com/v3/bus/stopid?parameters"
        for _, BVcode in self.sta2bv.items():
            params = {
                "id": BVcode,
                "city": self.city_name_zh_cn,
                "output": "json",
                "key": self.amap_ak,
            }
            response = requests.get(url=url, params=params)
            if response:
                res = response.json()
                if res["status"] == 1 or res["status"] == "1":
                    bv2amap[BVcode] = res
        sl2amap_id = {}
        bus_amap_info = {}
        for _, lines in self.bus_lines.items():
            for sl_name, sl in lines["sub_lines"].items():
                has_bv_stas = [s for s in sl["stations"] if s in self.sta2bv]
                if len(has_bv_stas) <= 1:
                    continue
                BVcodes = [self.sta2bv[s] for s in has_bv_stas]
                line_ids_sets = set()
                for bvcode in BVcodes:
                    if not bvcode in bv2amap:
                        continue
                    id_sets = set()
                    if len(bv2amap[bvcode].get("busstops", [])) < 1:
                        continue
                    for bl in bv2amap[bvcode]["busstops"][0]["buslines"]:
                        id_sets.add(bl["id"])
                        if bl["id"] not in bus_amap_info:
                            bus_amap_info[bl["id"]] = bl
                    if len(line_ids_sets) == 0:
                        line_ids_sets = id_sets
                    else:
                        line_ids_sets.intersection(id_sets)
                candidate_lids = []
                for lid in line_ids_sets:
                    # normal binary lines
                    info = bus_amap_info[lid]
                    start_stop = info["start_stop"]
                    end_stop = info["end_stop"]
                    if "-" in sl_name and len(sl_name.split("-")) >= 2:
                        sl_start, sl_end = sl_name.split("-")[0], sl_name.split("-")[-1]
                    else:
                        sl_start, sl_end = (
                            sl_name[: len(sl_name) // 2],
                            sl_name[len(sl_name) // 2 :],
                        )
                    candidate_lids.append(
                        (
                            lid,
                            Levenshtein.distance(sl_start, start_stop)
                            + Levenshtein.distance(sl_end, end_stop),
                        )
                    )
                if len(candidate_lids) > 0:
                    lid = min(candidate_lids, key=lambda x: x[1])
                    sl2amap_id[sl_name] = lid
        url = "https://restapi.amap.com/v3/bus/lineid?parameters"
        for sl_name, lid in sl2amap_id.items():
            params = {
                "id": lid,
                "output": "json",
                "key": self.amap_ak,
                "extensions": "all",
            }
            response = requests.get(url=url, params=params)
            if response:
                res = response.json()
                if res["status"] == 1 or res["status"] == "1":
                    self.sl2amap[sl_name] = res

    def _process_amap(self):
        bus_stations = {}
        bus_lines = {}

        def get_coords(coords_str: str):
            coords = []
            try:
                for c in coords_str.split(";"):
                    lon, lat = c.split(",")
                    coords.append((np.float64(lon), np.float64(lat)))
            except:
                pass
            return coords

        for k, v in self.bus_lines.items():
            sublines = []
            for sl_name, sl in v["sub_lines"].items():
                if not sl_name in self.sl2amap:
                    continue
                info = self.sl2amap[sl_name]
                if len(info.get("buslines", [])) < 1:
                    continue
                bline = info["buslines"][0]
                bstops = bline["busstops"]
                if len(bstops) < 1:
                    continue
                if not all(
                    "location" in st and len(get_coords(st["location"])) > 0
                    for st in bstops
                ):
                    continue
                sl["stops"] = []
                line_str = bline["polyline"]
                for st in bstops:
                    st_id = st["id"]
                    st_name = st["name"]
                    if not st_id:
                        st_id = st_name
                    sl["stops"].append((st_name, st_id))
                    st_loc = gcj2wgs(*(get_coords(st["location"])[0]))
                    bus_stations[(st_name, st_id)] = {
                        "name": st_name,
                        "geo": st_loc,
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
                sl["split_str"] = split_str
                sl["all_geo"] = [gcj2wgs(*c) for c in get_coords(line_str)]
                sublines.append(sl)
            if len(sublines) == 0:
                continue
            bus_lines[k] = {
                "name": k,
                "sublines": sublines,
            }
        return (bus_stations, bus_lines)

    def get_output_data(self):
        self._fetch_raw_data()
        self._fetch_amap_positions()
        self._fetch_amap_lines()
        bus_stations, bus_lines = self._process_amap()
        sta2id = {k: i for i, (k, _) in enumerate(bus_stations.items())}
        bus_data = {
            "stations": [],
            "lines": [],
        }
        ii = 0
        sl2id = {}
        sta2sl = defaultdict(list)
        for k, v in bus_lines.items():
            line_data = {"name": k, "type": "BUS", "sublines": []}
            for sl in v["sublines"]:
                line_data["sublines"].append(
                    {
                        "id": ii,
                        "name": k + " " + sl["name"],  # line name +subline name
                        "geo": sl["geo"],
                        "stations": [sta2id[i] for i in sl["stops"]],
                        "schedules": [],
                    }
                )
                for kk in sl["stops"]:
                    sta2sl[kk].append(ii)
                sl2id[sl["name"]] = ii
                ii += 1
            bus_data["lines"].append(line_data)
        for ii, (k, v) in enumerate(bus_stations.items()):
            bus_data["stations"].append(
                {
                    "id": ii,
                    "name": v["name"],
                    "geo": [list(v["geo"])],
                    "type": "BUS",
                    "subline_ids": sta2sl[k],
                }
            )
        return bus_data
