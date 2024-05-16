"""Convert AOI POI for aoi_matcher"""

import logging
from typing import Dict, Optional, Tuple
from xml.dom.minidom import parse

import numpy as np
import pyproj
from geojson import FeatureCollection

from .._util.line import offset_lane

__all__ = [
    "convert_aoi",
    "convert_poi",
    "convert_sumo_aoi_poi",
    "convert_sumo_stops",
]


def _shape2geo(shape_str):
    """Tool function for reading SUMO files"""
    shape_coords = shape_str.split(" ")
    res = []
    for coord in shape_coords:
        x, y = coord.split(",")
        res.append((np.float64(x), np.float64(y)))
    if len(res) == 2:
        res.append((res[0][0], res[1][1]))
    return res


def convert_aoi(
    geos: FeatureCollection, public_transport_data: Dict[str, dict], projstr: str
) -> Tuple[list, list]:
    aois = []
    stops = []
    projector = pyproj.Proj(projstr)
    if geos["features"] is not None:
        for i, feature in enumerate(geos["features"]):
            if not feature["geometry"]["type"] == "Polygon":
                logging.warning(f"Invalid geometry {feature}")
                continue
            osm_tags = feature["properties"].get("osm_tags", [])
            aoi_id = feature["properties"].get("id", len(geos["features"]) + i)
            coords = feature["geometry"]["coordinates"][
                0
            ]  # Inner shape is not supported yet
            aoi_type = feature["properties"].get("type", "")
            aois.append(
                {
                    "id": aoi_id,
                    "coords": [projector(*c) for c in coords],
                    "external": {
                        "population": 0,
                        "inner_poi": [],
                        "inner_poi_catg": [],
                        "osm_tags": osm_tags,
                        "land_types": {},
                    },
                }
            )
    for stype, stas in public_transport_data["stations"].items():
        for sta_id, sta in stas.items():
            stops.append(
                {
                    "id": sta_id,
                    "coords": sta["geo"],
                    "external": {
                        "population": 0,
                        "stop_id": sta_id,
                        "station_type": stype,
                        "capacity": sta["capacity"],
                        "subline_lane_matcher": sta[
                            "subline_lane_matcher"
                        ],  # for subway
                        "subline_geos": sta["subline_geos"],  # for bus
                        "name": sta["name"],
                        "inner_poi": [],
                        "inner_poi_catg": [],
                        "osm_tags": [],
                        "land_types": {},
                    },
                }
            )
    return (aois, stops)


def convert_poi(geos: FeatureCollection, projstr: str) -> list:
    pois = []
    projector = pyproj.Proj(projstr)
    if geos["features"] is not None:
        for feature in geos["features"]:
            if not feature["geometry"]["type"] == "Point":
                logging.warning(f"Invalid geometry {feature}")
                continue
            poi_id = feature["properties"]["id"]
            coords = feature["geometry"]["coordinates"]
            catg = feature["properties"].get("catg", "")
            name = feature["properties"].get("name", "")
            if catg not in ["261200", "271021", "271022", "271212", "271213"]:
                pois.append(
                    {
                        "id": poi_id,
                        "coords": [projector(*coords)],
                        "external": {
                            "name": name,
                            "catg": catg,
                        },
                    }
                )
            # The meaning of the above catgs
            # 261200 道路名
            # 271021 271022 公交线路 地铁线路
            # 271212 271213 道路出入口 路口
            # 详见https://lbs.qq.com/service/webService/webServiceGuide/webServiceAppendix
    return pois


def convert_sumo_aoi_poi(
    projstr: str,
    id2uid: dict,
    map_lanes: dict,
    poly_path: Optional[str] = None,
    max_longitude: Optional[float] = None,
    min_longitude: Optional[float] = None,
    max_latitude: Optional[float] = None,
    min_latitude: Optional[float] = None,
) -> Tuple[list, list]:
    aois = []
    pois = []
    projector = pyproj.Proj(projstr)

    def in_boundary(geo):
        return any(
            l is None
            for l in [min_longitude, max_longitude, min_latitude, max_latitude]
        ) or any(
            [
                min_longitude < x < max_longitude and min_latitude < y < max_latitude
                for (x, y) in geo
            ]
        )

    if poly_path is not None:
        poly_dom_tree = parse(poly_path)  # Original AOI POI file path
        poly_root_node = poly_dom_tree.documentElement
        for poly in poly_root_node.getElementsByTagName("poly"):
            poly_id = poly.getAttribute("id")
            poly_type = poly.getAttribute("type") if poly.hasAttribute("type") else ""
            geo_flag = (
                True
                if poly.hasAttribute("geo") and poly.getAttribute("geo") == "1"
                else False
            )
            if not geo_flag:
                continue
            poly_shape = poly.getAttribute("shape")
            geo = _shape2geo(poly_shape)
            if in_boundary(geo):
                if len(geo) == 1:
                    pois.append(
                        {
                            "id": poly_id,
                            "coords": [projector(lon, lat) for (lon, lat) in geo],
                            "external": {
                                "name": "",  # SUMO's POI does not have a name field
                                "catg": poly_type,
                            },
                        }
                    )
                elif len(geo) >= 3:
                    aois.append(
                        {
                            "id": poly_id,
                            "coords": [projector(lon, lat) for (lon, lat) in geo],
                            "external": {
                                "population": 0,
                                "osm_tags": [],
                                "inner_poi": [],
                                "inner_poi_catg": [],
                                "land_types": {},
                            },
                        }
                    )

                else:
                    logging.warning(f"Bad Poly {poly_id}")
        for poi in poly_root_node.getElementsByTagName("poi"):
            poi_id = poi.getAttribute("id")
            poi_type = poi.getAttribute("type")
            if poi.hasAttribute("lon") and poi.hasAttribute("lat"):
                lon = np.float64(poi.getAttribute("lon"))
                lat = np.float64(poi.getAttribute("lat"))
                gps_loc = [lon, lat]
            elif poi.hasAttribute("x") and poi.hasAttribute("y"):
                x = np.float64(poi.getAttribute("x"))
                y = np.float64(poi.getAttribute("y"))
                gps_loc = list(projector(x, y, inverse=True))

            elif poi.hasAttribute("lane"):
                lane_id = id2uid[poi.getAttribute("lane")]
                loc_line = map_lanes[lane_id]["geo"]
                pos = np.float64(poi.getAttribute("pos"))
                pos_lateral = (
                    float(poi.getAttribute("posLat"))
                    if poi.hasAttribute("posLat")
                    else 0.0
                )
                ratio_pos = pos / loc_line.length
                x, y = (
                    offset_lane(loc_line, pos_lateral)
                    .interpolate(ratio_pos, normalized=True)
                    .coords[0]
                )
                gps_loc = list(projector(x, y, inverse=True))
            else:
                logging.warning(f"Incomplete poi {poi_id}!")
                continue
            if in_boundary([gps_loc]):
                pois.append(
                    {
                        "id": poi_id,
                        "coords": [projector(lon, lat) for (lon, lat) in [gps_loc]],
                        "external": {
                            "name": "",  # SUMO's POI does not have a name field
                            "catg": poi_type,
                        },
                    }
                )

    return (aois, pois)


def convert_sumo_stops(
    projstr: str,
    id2uid: dict,
    map_lanes: dict,
    additional_path: Optional[str] = None,
    max_longitude: Optional[float] = None,
    min_longitude: Optional[float] = None,
    max_latitude: Optional[float] = None,
    min_latitude: Optional[float] = None,
) -> list:
    stops = []
    projector = pyproj.Proj(projstr)

    def in_boundary(geo):
        return any(
            l is None
            for l in [min_longitude, max_longitude, min_latitude, max_latitude]
        ) or any(
            [
                min_longitude < x < max_longitude and min_latitude < y < max_latitude
                for (x, y) in geo
            ]
        )

    if additional_path is not None:
        add_dom_tree = parse(additional_path)
        add_root_node = add_dom_tree.documentElement
        for bus_stop in add_root_node.getElementsByTagName("busStop"):
            bus_stop_id = bus_stop.getAttribute("id")
            bus_stop_name = (
                bus_stop.getAttribute("name") if bus_stop.hasAttribute("name") else ""
            )
            stop_lid = id2uid[bus_stop.getAttribute("lane")]
            stop_line = map_lanes[stop_lid]["geo"]
            start_pos = (
                np.float64(bus_stop.getAttribute("startPos"))
                if bus_stop.hasAttribute("startPos")
                else 0.1 * stop_line.length
            )
            if start_pos < 0:
                start_pos += stop_line.length
            end_pos = (
                np.float64(bus_stop.getAttribute("endPos"))
                if bus_stop.hasAttribute("endPos")
                else 0.9 * stop_line.length
            )
            if end_pos < 0:
                end_pos += stop_line.length
            stop_s = (end_pos + start_pos) / 2
            stop_s = np.clip(stop_s, 0.1 * stop_line.length, 0.9 * stop_line.length)
            x, y = stop_line.interpolate(stop_s).coords[0]
            gps_loc = list(projector(x, y, inverse=True))
            if in_boundary([gps_loc]):
                # TODO: add convert for SUMO PT
                stops.append(
                    {
                        "id": bus_stop_id,
                        "coords": [projector(*c) for c in [gps_loc]],
                        "external": {
                            "name": bus_stop_name,
                            "population": 0,
                            "capacity": 0,
                            "stop_id": bus_stop_id,
                            "subline_lane_matcher": [],  # for subway
                            "subline_geos": [],  # for bus
                            "population": 0,
                            "catg": "",
                            "inner_poi": [bus_stop_id],
                            "inner_poi_catg": [""],
                        },
                    }
                )
    return stops
