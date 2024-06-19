"""Check the field type and value of map format"""

import logging
import sys
from collections import defaultdict

from geojson import FeatureCollection

from ...type import LaneType
from .._map_util.const import *

__all__ = ["geojson_format_check", "output_format_check"]


class _FormatCheckHandler(logging.Handler):
    def __init__(self):
        logging.Handler.__init__(self)
        self.messages = []

    def emit(self, record):
        self.messages.append(self.format(record))

    def trigger_warnings(self):
        if self.messages:
            raise ValueError("Mapbuilder terminated due to warning messages above.")
        else:
            pass


def geojson_format_check(
    topo: FeatureCollection,
) -> None:
    logging.basicConfig(level=logging.INFO)
    handler = _FormatCheckHandler()
    logger = logging.getLogger()
    logger.addHandler(handler)
    topo_dict = {k: [] for k in ("MultiPoint", "LineString")}
    for feature in topo["features"]:
        if "geometry" not in feature:
            logging.warning(f"No 'geometry' at {feature}")
            continue
        if "type" not in feature["geometry"]:
            logging.warning(f"No 'geometry.type' at {feature}")
            continue
        if "properties" not in feature:
            logging.warning(f"No 'properties' at {feature}")
            continue
        feature_type = feature["geometry"]["type"]
        if feature_type not in topo_dict:
            logging.warning(
                f"Bad geometry type at {feature}, should be 'MultiPoint' or 'LineString'"
            )
        else:
            topo_dict[feature_type].append(feature)
    road_ids = set()
    junc_ids = set()
    out_way_id2junc_id = defaultdict(
        list
    )  # Check if the pre-junction of the road is unique
    in_way_id2junc_id = defaultdict(
        list
    )  # Check if the suc-junction of the road is unique
    # Check road
    for feature in topo_dict["LineString"]:
        # id
        feature_id = None
        if "id" in feature or "id" in feature["properties"]:
            feature_id = (
                feature["id"] if "id" in feature else feature["properties"]["id"]
            )
            if type(feature_id) == int:
                if feature_id in road_ids:
                    logging.warning(f"duplicated 'id' {feature_id}")
                road_ids.add(feature_id)
            else:
                logging.warning(f"ID {feature_id} is not int type")
        else:
            logging.warning(f"No 'id' at {feature}")
        # properties.lanes
        lanes = None
        if "lanes" in feature["properties"]:
            lanes = feature["properties"]["lanes"]
            if type(lanes) == int:
                if lanes <= 0:
                    logging.warning(f"lanes at {feature_id} is <= 0")
            else:
                logging.warning(f"lanes at {feature_id} is not int type")
        else:
            logging.warning(f"No 'lanes' at {feature}")
        # properties.highway
        highway = None
        if "highway" in feature["properties"]:
            highway = feature["properties"]["highway"]
            if type(highway) == str:
                pass
            else:
                logging.warning(f"highway at {feature_id} is not string type")
        else:
            logging.warning(f"No 'properties.highway' at {feature}")
        # properties.max_speed
        max_speed = None
        if "max_speed" in feature["properties"]:
            max_speed = feature["properties"]["max_speed"]
            if type(max_speed) == float:
                pass
            else:
                logging.warning(
                    f"properties.max_speed at {feature_id} is not float type"
                )
            if max_speed <= 0:
                logging.warning(f"max_speed at {feature_id} is <= 0")
        else:
            logging.warning(f"No 'properties.max_speed' at {feature}")
        # properties.name
        fname = None
        if "name" in feature["properties"]:
            fname = feature["properties"]["name"]
            if type(fname) == str:
                pass
            else:
                logging.warning(f"name at {feature_id} is not string type")
        else:
            logging.warning(f"No 'properties.name' at {feature}")
        # properties.turn
        fturn = None
        if "turn" in feature["properties"]:
            fturn = feature["properties"]["turn"]
            if type(fturn) == list:
                if not all(
                    s
                    in [
                        "A",
                        "L",
                        "S",
                        "R",
                    ]
                    for t in fturn
                    for s in t
                ):
                    logging.warning(
                        f"properties.turn at {feature_id} should be 'A','L','S' or 'R'"
                    )
                if not len(fturn) == lanes:
                    logging.warning(
                        f"properties.turn at {feature_id} has different length from 'properties.lanes'"
                    )
            else:
                logging.warning(f"properties.turn at {feature_id} is not list type")
        # properties.width
        lane_width = None
        if "width" in feature["properties"] or "lanewidth" in feature["properties"]:
            lane_width = feature["properties"].get(
                "width", feature["properties"].get("lanewidth", None)
            )
            if type(lane_width) == float:
                if lane_width <= 0:
                    logging.warning(f"width at {feature_id} is <= 0")
            else:
                logging.warning(f"width at {feature_id} is not float type")
        # properties.walk_lane_width
        walk_lane_width = None
        if "walk_lane_width" in feature["properties"]:
            walk_lane_width = feature["properties"]["walk_lane_width"]
            if type(walk_lane_width) == float:
                if walk_lane_width <= 0:
                    logging.warning(f"walk lane width at {feature_id} is <= 0")
            else:
                logging.warning(f"walk lane width at {feature_id} is not float type")
        if lane_width is not None:
            walk_lane_width = lane_width
        # properties.walk_lane_offset
        walk_lane_offset = None
        if "walk_lane_offset" in feature["properties"]:
            walk_lane_offset = feature["properties"]["walk_lane_offset"]
            if type(walk_lane_offset) == float:
                if (
                    lane_width is not None
                    and walk_lane_width is not None
                    and (lane_width + walk_lane_width) * 0.5 + walk_lane_offset <= 0
                ):
                    logging.warning(
                        f"(width + walk_lane_width) * 0.5 + walk_lane_offset at {feature_id} is <= 0, should be > 0"
                    )
            else:
                logging.warning(f"walk_lane_offset at {feature_id} is not float type")
        # geometry.coordinates
        coords = None
        if "coordinates" in feature["geometry"]:
            coords = feature["geometry"]["coordinates"]
            if type(coords) == list:
                if not all(len(c) in (2, 3) for c in coords):
                    logging.warning(
                        f"length of points in geometry.coordinates at {feature_id} is not 2 (lng, lat), 3 (lng, lat, z)"
                    )
                if not all(type(p) == float for c in coords for p in c):
                    logging.warning(
                        f"x, y of point in geometry.coordinates at {feature_id} is not float type"
                    )
            else:
                logging.warning(
                    f"geometry.coordinates at {feature_id} is not list type"
                )
        else:
            logging.warning(f"No 'geometry.coordinates' at {feature}")
    if len(road_ids) == 0:
        logging.warning("No well-formed roads!")

    # Check junction
    for feature in topo_dict["MultiPoint"]:
        # id
        feature_id = None
        if "id" in feature or "id" in feature["properties"]:
            feature_id = (
                feature["id"] if "id" in feature else feature["properties"]["id"]
            )
            if type(feature_id) == int:
                if feature_id in junc_ids:
                    logging.warning(f"duplicated 'id' {feature_id}")
                junc_ids.add(feature_id)
            else:
                logging.warning(f"ID {feature_id} is not int type")
        else:
            logging.warning(f"No 'id' at {feature}")
        # properties.in_ways
        in_ways = None
        len_in_ways = 0
        if "in_ways" in feature["properties"]:
            in_ways = feature["properties"]["in_ways"]
            if type(in_ways) == list:
                if not all(type(w) == int for w in in_ways):
                    logging.warning(
                        f"properties.in_ways at {feature_id} should be list of int"
                    )
                if not all(w in road_ids for w in in_ways):
                    logging.warning(
                        f"properties.in_ways at {feature_id} has id that is not in roads"
                    )
                for w in in_ways:
                    in_way_id2junc_id[w].append(feature_id)
                len_in_ways = len(in_ways)
            else:
                logging.warning(f"properties.in_ways at {feature_id} is not list type")
        else:
            logging.warning(f"No 'properties.in_ways' at {feature}")
        # properties.out_ways
        out_ways = None
        len_out_ways = 0
        if "out_ways" in feature["properties"]:
            out_ways = feature["properties"]["out_ways"]
            if type(out_ways) == list:
                if not all(type(w) == int for w in out_ways):
                    logging.warning(
                        f"properties.out_ways at {feature_id} should be list of int"
                    )
                if not all(w in road_ids for w in out_ways):
                    logging.warning(
                        f"properties.out_ways at {feature_id} has id that is not in roads"
                    )
                for w in out_ways:
                    out_way_id2junc_id[w].append(feature_id)
                len_out_ways = len(out_ways)
            else:
                logging.warning(f"properties.out_ways at {feature_id} is not list type")
        else:
            logging.warning(f"No 'properties.out_ways' at {feature}")
        if not len_in_ways + len_out_ways > 0:
            logging.warning(
                f"'properties.in_ways' and 'properties.out_ways' should have at least one road id at {feature_id}"
            )
        # geometry.coordinates
        coords = None
        if "coordinates" in feature["geometry"]:
            coords = feature["geometry"]["coordinates"]
            if type(coords) == list:
                if not all(len(c) in (2, 3) for c in coords):
                    logging.warning(
                        f"length of points in geometry.coordinates at {feature_id} is not 2 (lng, lat) or 3 (lng, lat ,z)"
                    )
                if not all(type(p) == float for c in coords for p in c):
                    logging.warning(
                        f"x, y of point in geometry.coordinates at {feature_id} is not float type"
                    )
            else:
                logging.warning(
                    f"geometry.coordinates at {feature_id} is not list type"
                )
        else:
            logging.warning(f"No 'geometry.coordinates' at {feature}")
    if len(junc_ids) == 0:
        logging.warning("No well-formed junctions!")
    # Check connections
    for in_way_id, feature_ids in in_way_id2junc_id.items():
        if len(feature_ids) > 1:
            logging.warning(
                f"Road {in_way_id}'s successor is not unique: {feature_ids}"
            )
    for out_way_id, feature_ids in out_way_id2junc_id.items():
        if len(feature_ids) > 1:
            logging.warning(
                f"Road {out_way_id}'s predecessor is not unique: {feature_ids}"
            )
    handler.trigger_warnings()


def output_format_check(output_map: dict, output_lane_length_check: bool):
    logging.basicConfig(level=logging.INFO)
    handler = _FormatCheckHandler()
    logger = logging.getLogger()
    logger.addHandler(handler)
    for class_name in [
        "header",
        "lanes",
        "roads",
        "junctions",
        "aois",
        "pois",
    ]:
        if not class_name in output_map:
            logging.warning(f"No class:{class_name} in output map!")
    # Check connections
    lanes = output_map["lanes"]
    dict_lanes = {l["id"]: l for l in lanes}
    roads = output_map["roads"]
    juncs = output_map["junctions"]
    lane_type_dict = {
        LaneType.LANE_TYPE_DRIVING: "drive",
        LaneType.LANE_TYPE_WALKING: "walk",
    }
    road_id2lane_ids = defaultdict(list)
    for road in roads:
        road_id = road["id"]
        for lane_id in road["lane_ids"]:
            road_id2lane_ids[road_id].append(lane_id)
            road_lane = dict_lanes[lane_id]
            pre_parent_ids = set()
            suc_parent_ids = set()
            # the predecessors should be within the same junction
            for pre in road_lane["predecessors"]:
                pre_lane = dict_lanes[pre["id"]]
                pre_parent_ids.add(pre_lane["parent_id"])
            if len(pre_parent_ids) > 1:
                logging.warning(
                    f"road {road_id} has predecessors in more than one junction {pre_parent_ids}"
                )
            elif any(pid < JUNC_START_ID for pid in pre_parent_ids):
                logging.warning(
                    f"road {road_id} has predecessors in road {pre_parent_ids}"
                )
            # the successors should be within the same junction
            for suc in road_lane["successors"]:
                suc_lane = dict_lanes[suc["id"]]
                suc_parent_ids.add(suc_lane["parent_id"])
            if len(suc_parent_ids) > 1:
                logging.warning(
                    f"road {road_id} has successors in more than one junction {suc_parent_ids}"
                )
            elif any(pid < JUNC_START_ID for pid in suc_parent_ids):
                logging.warning(
                    f"road {road_id} has successors in road {suc_parent_ids}"
                )
    junc_id2lane_ids = defaultdict(list)
    for junc in juncs:
        junc_id = junc["id"]
        for lane_id in junc["lane_ids"]:
            junc_id2lane_ids[junc_id].append(lane_id)
    all_lane_ids = set(l["id"] for l in lanes)
    for lane in lanes:
        lane_id = lane["id"]
        lane_type = lane_type_dict.get(lane["type"], "unspecified")
        parent_id = lane["parent_id"]
        if parent_id in road_id2lane_ids:
            lane_ids = road_id2lane_ids[parent_id]
            if not lane_id in lane_ids:
                logging.warning(
                    f"{lane_type} lane {lane_id} not in parent road {parent_id}"
                )
        elif parent_id in junc_id2lane_ids:
            lane_ids = junc_id2lane_ids[parent_id]
            if not lane_id in lane_ids:
                logging.warning(
                    f"{lane_type} lane {lane_id} not in parent junction {parent_id}"
                )
        else:
            logging.warning(
                f"Parent {parent_id} of {lane_type} lane {lane_id} not in output_map"
            )
        # Check if predecessors and successors exist
        for pre in lane["predecessors"]:
            pre_lane_id = pre["id"]
            if not pre_lane_id in all_lane_ids:
                logging.warning(
                    f"{lane_type} lane {lane_id} has predecessor {pre_lane_id} not in lane data"
                )
        for suc in lane["successors"]:
            suc_lane_id = suc["id"]
            if not suc_lane_id in all_lane_ids:
                logging.warning(
                    f"{lane_type} lane {lane_id} has successor {suc_lane_id} not in lane data"
                )
    # Check values
    if output_lane_length_check:
        for lane in lanes:
            lane_id = lane["id"]
            length = lane["length"]
            lane_type = lane_type_dict.get(lane["type"], "unspecified")
            parent_id = lane["parent_id"]
            if parent_id < JUNC_START_ID:
                continue
            if length > MAX_JUNC_LANE_LENGTH:
                logging.warning(
                    f"Junction {lane_type} lane {lane_id} is too long ({length} m), please check input GeoJSON file!"
                )
    handler.trigger_warnings()
