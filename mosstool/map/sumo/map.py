import logging
from multiprocessing import cpu_count
import time
from collections import defaultdict
from math import atan2
from typing import Optional
from xml.dom.minidom import parse

import numpy as np
import pycityproto.city.map.v2.map_pb2 as mapv2
import shapely.ops as ops
from shapely.geometry import LineString, MultiPoint, Point, Polygon

from .._map_util.aoi_matcher import add_sumo_aoi_to_map
from .._map_util.aoiutils import generate_sumo_aoi_poi
from .._map_util.const import *
from .._map_util.convert_aoi import convert_sumo_aoi_poi, convert_sumo_stops
from .._map_util.gen_traffic_light import convert_traffic_light
from .._util.angle import abs_delta_angle
from .._util.line import (
    connect_line_string,
    get_line_angle,
    get_start_vector,
)

__all__ = ["MapConverter"]


class MapConverter:
    def __init__(
        self,
        net_path: str,
        default_lane_width: float = 3.2,
        green_time: float = 60.0,
        yellow_time: float = 5.0,
        poly_path: Optional[str] = None,
        additional_path: Optional[str] = None,
        traffic_light_path: Optional[str] = None,
        traffic_light_min_direction_group: int = 3,
        merge_aoi: bool = False,
        workers: int = cpu_count(),
    ):
        """
        Args:
        - net_path (str): The path to the SUMO road net file.
        - default_lane_width (float): default lane width
        - green_time (float): green time
        - yellow_time (float): yellow time
        - poly_path (str): The path to the SUMO aois and pois file.
        - additional_path (str): The path to the SUMO additional file.
        - traffic_light_path (str): The path to the SUMO traffic-light file.
        - traffic_light_min_direction_group (int): minimum number of lane directions for traffic-light generation
        - merge_aoi (bool): merge nearby aois
        - workers (int): number of workers
        """
        self._lane_width = default_lane_width
        self._green_time = green_time
        self._yellow_time = yellow_time
        self._poly_path = poly_path  # Original AOI POI path
        self._additional_path = additional_path  # Additional file path, take "busStop"
        self._traffic_light_path = traffic_light_path  # Traffic control file path
        self._traffic_light_min_direction_group = traffic_light_min_direction_group
        self._merge_aoi = merge_aoi
        self._workers = workers
        logging.info(f"Reading net file from {net_path}")
        dom_tree = parse(net_path)
        # Get the root node
        root_node = dom_tree.documentElement
        loc_node = root_node.getElementsByTagName("location")[
            0
        ]  # SUMO road network has only one location node
        self._projstr = loc_node.getAttribute("projParameter")
        x_offset, y_offset = loc_node.getAttribute("netOffset").split(",")
        self._net_offset = (np.float64(x_offset), np.float64(y_offset))
        self._orig_boundary = [
            np.float64(s) for s in loc_node.getAttribute("origBoundary").split(",")
        ]
        # Read SUMO .net.xml file
        # At least the following four Elements need to exist to convert a complete map, so it is directly obtained without judging whether it exists.
        self._types = root_node.getElementsByTagName("type")
        self._edges = root_node.getElementsByTagName("edge")
        self._juncs = root_node.getElementsByTagName("junction")
        self._conns = root_node.getElementsByTagName("connection")

        def shape2array(shape_str):
            shape_coords = shape_str.split(" ")
            x_offset, y_offset = self._net_offset  # Projection offset
            res = []
            for coord in shape_coords:
                x, y = coord.split(",")
                res.append((np.float64(x) - x_offset, np.float64(y) - y_offset))
            return res

        def get_lane_type(allow, disallow):
            """Determine sidewalks/roadways based on traffic restrictions"""
            if "pedestrian" in disallow:
                return mapv2.LANE_TYPE_DRIVING
            if "pedestrian" in allow and not any(
                v in allow
                for v in [
                    "bus",
                    "delivery",
                    "motorcycle",
                ]
            ):
                return mapv2.LANE_TYPE_WALKING
            return mapv2.LANE_TYPE_DRIVING

        def get_turn_type(dir):
            """Determine the turn type based on the dir of the connection"""
            if dir == "s":
                return mapv2.LANE_TURN_STRAIGHT
            if dir == "t":
                return mapv2.LANE_TURN_AROUND
            if dir == "l" or dir == "L":
                return mapv2.LANE_TURN_LEFT
            if dir == "r" or dir == "R":
                mapv2.LANE_TURN_RIGHT
            return mapv2.LANE_TURN_STRAIGHT

        # uid starting point
        self._lane_uid = LANE_START_ID
        self._road_uid = ROAD_START_ID
        self._junc_uid = JUNC_START_ID
        self._map_lanes = {}
        self._map_roads = {}
        self._map_aois = {}
        self._map_pois = {}
        self._internal_lanes = (
            {}
        )  # The original id (str) lane of the internal edge is currently reserved only to see the speed limit.
        self._junc_conns = (
            {}
        )  # (original from id(str), original internal id(str))->connected edge
        # Original from edge id(str)->[] Original to edge id(str)
        self._plain_conns = defaultdict(list)
        self._tjunctions = {}
        self._map_junctions = {}
        self._id2uid = {}  # Original id(str)->uid(int)
        self._etype2data = {}  # edge type -> {allowed type not allowed type}
        # Output Data
        self._output_lanes = {}
        self._output_roads = {}
        self._output_junctions = {}
        self._output_aois = {}
        self._output_pois = {}
        logging.info("Reading types")
        for t in self._types:
            tid = t.getAttribute("id")
            if t.hasAttribute("allow"):
                allow = t.getAttribute("allow")
            else:
                allow = ""
            if t.hasAttribute("disallow"):
                disallow = t.getAttribute("disallow")
            else:
                disallow = ""
            if t.hasAttribute("width"):
                width = np.float64(t.getAttribute("width"))
            else:
                width = self._lane_width
            self._etype2data[tid] = {
                "allow": allow,
                "disallow": disallow,
                "width": width,
            }
        logging.info("Reading edges")
        for edge in self._edges:
            eid = edge.getAttribute("id")
            # if (eid[0] == ":"):
            if (
                edge.hasAttribute("function")
                and edge.getAttribute("function") == "internal"
            ):
                # internal edge
                lanes = edge.getElementsByTagName("lane")
                for lane in lanes:
                    lid = lane.getAttribute("id")
                    speed = np.float64(lane.getAttribute("speed"))
                    self._internal_lanes[lid] = {
                        "max_speed": speed,
                    }
            else:
                name = edge.getAttribute("name") if edge.hasAttribute("name") else ""
                etype = edge.getAttribute("type")
                if not any(t in etype for t in SUMO_WAY_LEVELS):
                    # Only handle edges of the specified level
                    continue
                e_allow = self._etype2data[etype]["allow"]
                e_disallow = self._etype2data[etype]["disallow"]
                lane_width = self._etype2data[etype]["width"]
                lanes = edge.getElementsByTagName("lane")
                lane_ids = []
                for lane in lanes:
                    lid = lane.getAttribute("id")
                    speed = np.float64(lane.getAttribute("speed"))
                    shape = lane.getAttribute("shape")
                    l_allow = (
                        lane.getAttribute("allow") if lane.hasAttribute("allow") else ""
                    )
                    l_disallow = (
                        lane.getAttribute("disallow")
                        if lane.hasAttribute("disallow")
                        else ""
                    )
                    lane_type = get_lane_type(
                        allow=e_allow + l_allow, disallow=e_disallow + l_disallow
                    )
                    line = LineString(shape2array(shape))
                    if line.length > 3 * lane_width:
                        line = ops.substring(line, lane_width, line.length - lane_width)
                    else:
                        line = ops.substring(
                            line, line.length / 3, line.length - line.length / 3
                        )
                    line = line.simplify(0.1)
                    self._id2uid[lid] = self._lane_uid
                    lane_ids.append(self._lane_uid)
                    self._map_lanes[self._lane_uid] = {
                        "id": lid,  # original id
                        "geo": line,
                        "in": [],
                        "out": [],
                        "left_lane_ids": [],
                        "right_lane_ids": [],
                        "max_speed": speed,
                        "parent_id": self._road_uid,
                        "width": lane_width,
                        "type": lane_type,
                        "turn": mapv2.LANE_TURN_STRAIGHT,
                    }
                    self._lane_uid += 1
                self._id2uid[eid] = self._road_uid
                self._map_roads[self._road_uid] = {
                    "lane_ids": lane_ids,
                    "name": name,
                    "highway": etype.split(".")[-1],
                }
                self._road_uid += 1
        logging.info("Reading junctions")
        tjunc_id = 0  # Because some junctions filter out some levels of roads, it may not be possible to generate junctions that meet the requirements. This part needs to be deleted.
        for junc in self._juncs:
            jid = junc.getAttribute("id")
            # if (jid[0] == ":"):
            if junc.hasAttribute("type") and junc.getAttribute("type") == "internal":
                # internal junction
                # ATTENTION: internal junction indicates junction traffic rules and is not processed.
                pass
            else:
                inc_lids = junc.getAttribute("incLanes").split(" ")  # inc=incoming
                int_lids = junc.getAttribute("intLanes").split(" ")  # int=internal
                shape = junc.getAttribute("shape")
                multi_point = MultiPoint(shape2array(shape))
                x_center, y_center = multi_point.centroid.coords[0]
                if any(lid in self._id2uid.keys() for lid in inc_lids):
                    self._tjunctions[tjunc_id] = {
                        "id": jid,
                        "lane_ids": [],  # Subsequent processing of internal connections
                        "inc_lids": inc_lids,
                        "int_lids": int_lids,
                        "center": {
                            "x": x_center,
                            "y": y_center,
                        },
                    }
                    tjunc_id += 1
        logging.info("Reading connections")
        for conn in self._conns:
            if conn.hasAttribute("via"):
                # is the connection inside the junction
                lid = conn.getAttribute("via")
                dir = conn.getAttribute("dir")
                from_edge = conn.getAttribute("from")
                from_offset = conn.getAttribute("fromLane")  # Subscript offset
                to_edge = conn.getAttribute("to")
                to_offset = conn.getAttribute("toLane")  # Subscript offset
                from_lid = from_edge + "_" + from_offset
                to_lid = to_edge + "_" + to_offset
                self._junc_conns[(from_lid, lid)] = {
                    "to_lid": to_lid,
                    "turn_type": get_turn_type(dir),
                }
            else:
                # is the road connection outside the junction. dir is not useful for road network construction and is not used here.
                from_edge = conn.getAttribute("from")
                from_offset = conn.getAttribute("fromLane")  # Subscript offset
                to_edge = conn.getAttribute("to")
                to_offset = conn.getAttribute("toLane")  # Subscript offset
                from_lid = from_edge + "_" + from_offset
                to_lid = to_edge + "_" + to_offset
                self._plain_conns[from_lid].append(to_lid)

        logging.info("Reading Complete")

    def _connect_lane(
        self, in_uid, out_uid, orig_lid, lane_turn, lane_type, junc_id, max_speed
    ):
        """
        Connect two lanes
        """
        in_lane = self._map_lanes[in_uid]["geo"]
        out_lane = self._map_lanes[out_uid]["geo"]
        # Lanes are connected with Bezier curves and sidewalks are connected with straight lines
        conn_lane = (
            connect_line_string(in_lane, out_lane)
            if lane_type != mapv2.LANE_TYPE_WALKING
            else LineString([in_lane.coords[-1], out_lane.coords[0]])
        )
        # Add new lane
        self._map_lanes[self._lane_uid] = conn_lane
        # Add the connection relationship of the new lane
        lane_conn_in = [
            {"id": in_uid, "type": mapv2.LANE_CONNECTION_TYPE_TAIL},
        ]
        lane_conn_out = [
            {"id": out_uid, "type": mapv2.LANE_CONNECTION_TYPE_HEAD},
        ]
        self._map_lanes[self._lane_uid] = {
            "id": orig_lid,
            "geo": conn_lane,
            "in": lane_conn_in,
            "out": lane_conn_out,
            "left_lane_ids": [],
            "right_lane_ids": [],
            "max_speed": max_speed,
            "parent_id": junc_id,
            "width": self._lane_width,
            "type": lane_type,
            "turn": lane_turn,
        }
        # Do not record the mapping relationship of junction lane
        # if orig_lid:
        # self._id2uid[orig_lid] = self._lane_uid
        # Add the connection relationship of in_lane
        self._map_lanes[in_uid]["out"].append(
            {"id": self._lane_uid, "type": mapv2.LANE_CONNECTION_TYPE_HEAD}
        )
        # Add the connection relationship of out_lane
        self._map_lanes[out_uid]["in"].append(
            {"id": self._lane_uid, "type": mapv2.LANE_CONNECTION_TYPE_TAIL}
        )
        self._lane_uid += 1
        return self._lane_uid - 1

    def _create_junctions(self):
        """Reconstruct the junction based on the original SUMO road network connection relationship"""
        for _, junc in self._tjunctions.items():
            jid = junc["id"]
            inc_lids = junc["inc_lids"]
            int_lids = junc["int_lids"]
            junc_center = junc["center"]
            junc_lane_uids = []
            to_lids = set()
            # Normal lane connection relationship
            for from_lid in inc_lids:
                for lid in int_lids:
                    if (
                        not (from_lid, lid) in self._junc_conns.keys()
                        or not from_lid in self._id2uid.keys()
                        or not lid in self._internal_lanes.keys()
                    ):
                        continue
                    else:
                        junc_conn = self._junc_conns[(from_lid, lid)]
                        to_lid = junc_conn["to_lid"]
                        if not to_lid in self._id2uid.keys():
                            continue
                        to_lids.add(to_lid)
                        turn_type = junc_conn["turn_type"]
                        from_uid = self._id2uid[from_lid]
                        from_lane = self._map_lanes[from_uid]
                        to_uid = self._id2uid[to_lid]
                        to_lane = self._map_lanes[to_uid]
                        if not from_lane["type"] == to_lane["type"]:
                            # If the two roads are not roadways/pedestrians at the same time, they will not be connected.
                            continue
                        junc_lane_uids.append(
                            self._connect_lane(
                                in_uid=from_uid,
                                out_uid=to_uid,
                                orig_lid=lid,
                                lane_turn=turn_type,
                                lane_type=from_lane["type"],
                                junc_id=self._junc_uid,
                                max_speed=self._internal_lanes[lid]["max_speed"],
                            )
                        )
            # SUMO Two-way road comes with head-to-tail U-turn, also added here

            for from_lid in inc_lids:
                for to_lid in to_lids:
                    if (
                        from_lid == to_lid
                        or not from_lid.split("-")[-1] == to_lid.split("-")[-1]
                        or not from_lid in self._id2uid.keys()
                        or not to_lid in self._id2uid.keys()
                    ):
                        continue
                    else:
                        turn_type = mapv2.LANE_TURN_AROUND  # around
                        from_uid = self._id2uid[from_lid]
                        from_lane = self._map_lanes[from_uid]
                        to_uid = self._id2uid[to_lid]
                        to_lane = self._map_lanes[to_uid]
                        if not from_lane["type"] == to_lane["type"]:
                            # If the two roads are not roadways/pedestrians at the same time, they will not be connected.
                            continue
                        junc_lane_uids.append(
                            self._connect_lane(
                                in_uid=from_uid,
                                out_uid=to_uid,
                                orig_lid=None,
                                lane_turn=turn_type,
                                lane_type=from_lane["type"],
                                junc_id=self._junc_uid,
                                max_speed=from_lane["max_speed"],
                            )
                        )
            if junc_lane_uids:
                self._map_junctions[self._junc_uid] = {
                    "lane_ids": junc_lane_uids,
                    "center": junc_center,
                }
                self._id2uid[jid] = self._junc_uid
                self._junc_uid += 1

    def _add_lane_conn(self):
        """
        Add connections outside the junction
        """
        for from_lid, to_lids in self._plain_conns.items():
            if not from_lid in self._id2uid.keys():
                continue
            else:
                from_uid = self._id2uid[from_lid]
                from_lane = self._map_lanes[from_uid]
                if from_lane["parent_id"] >= 3_0000_0000:
                    continue
            for to_lid in to_lids:
                if not to_lid in self._id2uid.keys():
                    continue
                from_uid = self._id2uid[from_lid]
                from_lane = self._map_lanes[from_uid]
                to_uid = self._id2uid[to_lid]
                to_lane = self._map_lanes[to_uid]
                if to_lane["parent_id"] >= 3_0000_0000:
                    continue
                from_lane["out"].append(
                    {"id": to_uid, "type": mapv2.LANE_CONNECTION_TYPE_HEAD}
                )
                to_lane["in"].append(
                    {"id": from_uid, "type": mapv2.LANE_CONNECTION_TYPE_TAIL}
                )

    def _add_junc_lane_overlaps(self):
        """
        add overlap for junction lanes
        """
        for _, junc in self._output_junctions.items():
            junc_poly_lanes = [
                (l, LineString([[n["x"], n["y"]] for n in l["center_line"]["nodes"]]))
                for l in (self._output_lanes[lid] for lid in junc["lane_ids"])
            ]
            for ii, (lane_i, poly_i) in enumerate(junc_poly_lanes):
                angle_i = get_line_angle(poly_i)
                start_vec_i = get_start_vector(poly_i)
                for lane_j, poly_j in junc_poly_lanes[ii + 1 :]:
                    if lane_i["predecessors"][0] == lane_j["predecessors"][0]:
                        continue
                    intersections = poly_i.intersection(poly_j)
                    if isinstance(intersections, MultiPoint):
                        intersections = [p for p in intersections.geoms]
                    elif isinstance(intersections, Point):
                        intersections = [intersections]
                    else:
                        continue
                    # Priority rules:
                    # 0. yield to crosswalks
                    # 1. yield to road on the right
                    # 2. turning roads yield to non-turning road
                    # 3. right turns yields to left turns on opposite direction
                    self_first = True
                    both_false = False
                    angle_j = get_line_angle(poly_j)
                    start_vec_j = get_start_vector(poly_j)
                    abs_delta_angle_i_j = abs_delta_angle(angle_i, angle_j)
                    while True:
                        if lane_i["turn"] == mapv2.LANE_TURN_STRAIGHT:
                            if lane_j["turn"] == mapv2.LANE_TURN_STRAIGHT:
                                if (
                                    abs_delta_angle(abs_delta_angle_i_j, np.pi / 2)
                                    < np.pi / 4  # Close to 90 degree
                                ):
                                    if (
                                        np.cross(start_vec_i, start_vec_j) < 0
                                    ):  # i is about 90 degrees to the right of j
                                        #   ⬆
                                        #   ⬆ j
                                        #   ⬆
                                        # ➡➡➡➡ i
                                        #   ⬆
                                        self_first = False
                                else:
                                    both_false = True
                            break
                        if lane_j["turn"] == mapv2.LANE_TURN_STRAIGHT:
                            self_first = False
                            break
                        if abs_delta_angle_i_j > np.pi * 5 / 6:
                            if (
                                lane_i["turn"] == mapv2.LANE_TURN_RIGHT
                                and lane_j["turn"] == mapv2.LANE_TURN_LEFT
                            ):
                                self_first = False
                                break
                            if (
                                lane_j["turn"] == mapv2.LANE_TURN_RIGHT
                                and lane_i["turn"] == mapv2.LANE_TURN_LEFT
                            ):
                                break
                        # default
                        both_false = True
                        break
                    if both_false:
                        self_first = other_first = False
                    else:
                        other_first = not self_first
                    lane_i["overlaps"].extend(
                        {
                            "self": {"lane_id": lane_i["id"], "s": poly_i.project(p)},
                            "other": {"lane_id": lane_j["id"], "s": poly_j.project(p)},
                            "self_first": self_first,
                        }
                        for p in intersections
                    )
                    lane_j["overlaps"].extend(
                        {
                            "other": {"lane_id": lane_i["id"], "s": poly_i.project(p)},
                            "self": {"lane_id": lane_j["id"], "s": poly_j.project(p)},
                            "self_first": other_first,
                        }
                        for p in intersections
                    )

    def _add_driving_lane_group(self):
        """Clustering lane groups from the lanes at the junction"""
        # Shallow copy
        lanes = {lid: d for lid, d in self._output_lanes.items()}
        roads = {rid: d for rid, d in self._output_roads.items()}
        juncs = {jid: d for jid, d in self._output_junctions.items()}

        # Mapping from road id to angle
        road_start_angle = {}
        road_end_angle = {}
        for road in roads.values():
            l = lanes[road["lane_ids"][len(road["lane_ids"]) // 2]]
            nodes = l["center_line"]["nodes"]
            start_angle = atan2(
                nodes[1]["y"] - nodes[0]["y"], nodes[1]["x"] - nodes[0]["x"]
            )
            end_angle = atan2(
                nodes[-1]["y"] - nodes[-2]["y"], nodes[-1]["x"] - nodes[-2]["x"]
            )
            road_start_angle[road["id"]] = start_angle
            road_end_angle[road["id"]] = end_angle
        # Main logic
        for j in juncs.values():
            # (in road id, out road id) -> [lane id]
            lane_groups = defaultdict(list)
            for lid in j["lane_ids"]:
                l = lanes[lid]
                if l["type"] != mapv2.LANE_TYPE_DRIVING:  # driving
                    # Only process traffic lanes
                    continue
                pres = l["predecessors"]
                assert len(pres) == 1
                pre_lane = lanes[pres[0]["id"]]
                sucs = l["successors"]
                assert len(sucs) == 1
                suc_lane = lanes[sucs[0]["id"]]
                key = (pre_lane["parent_id"], suc_lane["parent_id"])
                lane_groups[key].append(lid)
            # Write into junction
            # message JunctionLaneGroup {
            # # The entrance road to this lane group
            # int32 in_road_id = 1;
            # # Entrance angle of this lane group (radians)
            # double in_angle = 2;
            # # The exit road of this lane group
            # int32 out_road_id = 3;
            # # The exit angle of this lane group (radians)
            # double out_angle = 4;
            # # Lanes included in this lane group
            # repeated int32 lane_ids = 5;
            # # The steering attributes of this lane group
            # LaneTurn turn = 6;
            # }
            j["driving_lane_groups"] = [
                {
                    "in_road_id": k[0],
                    "in_angle": road_end_angle[k[0]],
                    "out_road_id": k[1],
                    "out_angle": road_start_angle[k[1]],
                    "lane_ids": v,
                }
                for k, v in lane_groups.items()
            ]
            # Check whether the turns of all lanes in each group are the same, and set the turn of the lane group
            for group in j["driving_lane_groups"]:
                lane_ids = group["lane_ids"]
                turns = [lanes[lid]["turn"] for lid in lane_ids]
                # Turn the U-turn into a left turn
                turns = [
                    mapv2.LANE_TURN_LEFT if t == mapv2.LANE_TURN_AROUND else t
                    for t in turns
                ]
                # Check if all turns are the same
                assert all(t == turns[0] for t in turns)
                group["turn"] = turns[0]

    def _add_traffic_light(self):
        lanes = {lid: d for lid, d in self._output_lanes.items()}
        roads = {rid: d for rid, d in self._output_roads.items()}
        juncs = {jid: d for jid, d in self._output_junctions.items()}
        convert_traffic_light(
            lanes,
            roads,
            juncs,
            self._id2uid,
            self._green_time,
            self._yellow_time,
            self._traffic_light_min_direction_group,
            self._traffic_light_path,
        )

    def _get_output_map(self):
        """Post-processing converts map data into output format"""
        # output aoi
        logging.info("Make output aois")
        self._output_aois = {uid: d for uid, d in self._map_aois.items()}
        # output poi
        logging.info("Make output pois")
        self._output_pois = {uid: d for uid, d in self._map_pois.items()}
        # output road
        logging.info("Make output roads")

        def sort_lane_ids(lane_ids):
            """
            Sort from left to right according to lane space position
            """
            if not lane_ids:
                return []

            def get_in_angle(line):
                def in_get_vector(line):
                    return np.array(line.coords[-1]) - np.array(
                        line.coords[4 * len(line.coords) // 5 - 1]
                    )

                v = in_get_vector(line)
                return np.arctan2(v[1], v[0])

            in_lanes = {lid: self._map_lanes[lid]["geo"] for lid in lane_ids}
            in_angle = np.mean([get_in_angle(lane) for _, lane in in_lanes.items()])
            vec = np.array([np.cos(in_angle + np.pi / 2), np.sin(in_angle + np.pi / 2)])
            in_vecs = [
                {
                    "lid": lid,
                    "vec": np.array(l.coords[0]),
                }
                for lid, l in in_lanes.items()
            ]
            sorted_lids = [
                d["lid"]
                for d in sorted(in_vecs, key=lambda x: np.dot(x["vec"], vec))[::-1]
            ]  # The larger the inner product, the closer it is to the left
            return sorted_lids

        for rid, r in self._map_roads.items():
            # Process road lane ids and sort lanes (from left to right) and sidewalks (from left to right)
            lane_ids = r["lane_ids"]
            drive_lane_ids = sort_lane_ids(
                [
                    lid
                    for lid in lane_ids
                    if self._map_lanes[lid]["type"] == mapv2.LANE_TYPE_DRIVING
                ]
            )
            walk_lane_ids = sort_lane_ids(
                [
                    lid
                    for lid in lane_ids
                    if self._map_lanes[lid]["type"] == mapv2.LANE_TYPE_WALKING
                ]
            )
            for index, lid in enumerate(drive_lane_ids):
                lane = self._map_lanes[lid]
                # The left-right relationship between lanes and sidewalks does not have this left-right relationship
                # left lane
                for i in range(0, index)[::-1]:
                    lane["left_lane_ids"].append(drive_lane_ids[i])
                # right lane
                for i in range(index + 1, len(lane_ids)):
                    lane["right_lane_ids"].append(drive_lane_ids[i])
            self._output_roads[rid] = {
                "id": rid,
                "lane_ids": drive_lane_ids + walk_lane_ids,
                "external": {
                    "highway": r["highway"],
                    "name": r["name"],
                },
            }
        # output lane
        logging.info("Make output lanes")
        # Convert LineString to the nodes format used by the simulator

        def line2nodes(line):
            return {"nodes": [{"x": c[0], "y": c[1]} for c in line.coords]}

        for lid, lane in self._map_lanes.items():
            line = lane["geo"]
            lane_width = lane["width"]
            self._output_lanes[lid] = {
                "id": lid,
                "type": lane["type"],
                "turn": lane["turn"],
                "max_speed": lane["max_speed"],
                "length": line.length,
                "width": lane_width,
                "center_line": line2nodes(line),
                "predecessors": lane["in"],
                "successors": lane["out"],
                "left_lane_ids": lane["left_lane_ids"],
                "right_lane_ids": lane["right_lane_ids"],
                "parent_id": lane["parent_id"],
                "overlaps": [],
                "aoi_ids": [],
            }
        # add aoi ids
        for _, a in self._map_aois.items():
            for pos in a["driving_positions"]:
                self._output_lanes[pos["lane_id"]]["aoi_ids"].append(a["id"])
            for pos in a["walking_positions"]:
                self._output_lanes[pos["lane_id"]["aoi_ids"]].append(a["id"])
        # output junction
        logging.info("Make output junctions")
        for junc_id, j in self._map_junctions.items():
            junc_center = j["center"]
            junc_lids = j["lane_ids"]
            self._output_junctions[junc_id] = {
                "id": junc_id,
                "external": {
                    "center": junc_center,
                },
                "lane_ids": junc_lids,
            }
        # overlap for junc lanes
        self._add_junc_lane_overlaps()
        # driving lane group for junc
        self._add_driving_lane_group()
        # traffic_light for junc
        self._add_traffic_light()
        # header
        xy = [
            [i["x"], i["y"]]
            for j in self._output_lanes.values()
            for k in ["center_line"]
            for i in j[k]["nodes"]
        ]
        x, y = [*zip(*xy)]
        header = {
            "name": "map_builder",
            "date": time.strftime("%a %b %d %H:%M:%S %Y"),
            "north": max(y),
            "south": min(y),
            "west": min(x),
            "east": max(x),
            "projection": self._projstr,
        }
        logging.info(
            f"{len(self._output_lanes)} Lanes  {len(self._output_roads)} Roads  {len(self._output_junctions)} Junctions  {len(self._output_aois)} AOIs  {len(self._output_pois)} POIs"
        )

        output_map = {
            "header": header,
            "lanes": list(self._output_lanes.values()),
            "roads": list(self._output_roads.values()),
            "aois": list(self._output_aois.values()),
            "pois": list(self._output_pois.values()),
            "junctions": list(self._output_junctions.values()),
        }
        return output_map

    def _add_aois_to_map(self):
        # Pre-process read AOI, POI and BUS STOP
        aois, pois = convert_sumo_aoi_poi(
            poly_path=self._poly_path,
            projstr=self._projstr,
            id2uid=self._id2uid,
            map_lanes=self._map_lanes,
        )
        stops = convert_sumo_stops(
            additional_path=self._additional_path,
            projstr=self._projstr,
            id2uid=self._id2uid,
            map_lanes=self._map_lanes,
        )
        aois, stops, pois = generate_sumo_aoi_poi(
            aois, pois, stops, self._workers, self._merge_aoi
        )
        d_matcher = []
        road_lane_matcher = []
        for r in self._map_roads.values():
            lane_ids = r["lane_ids"]
            # Lanes that are too short do not participate in the segmentation of aggregated AOIs.
            center_lane_id = lane_ids[len(lane_ids) // 2]
            center_lane_geo = self._map_lanes[center_lane_id]["geo"]
            if center_lane_geo.length > 100:
                road_lane_matcher.append(
                    {
                        "id": center_lane_id,
                        "geo": center_lane_geo,
                        "point": center_lane_geo.interpolate(
                            0.5, normalized=True
                        ).coords[:][0],
                        "length": center_lane_geo.length,
                    }
                )

            d_lane_ids = [
                lid
                for lid in lane_ids
                if self._map_lanes[lid]["type"] == mapv2.LANE_TYPE_DRIVING
            ]
            if len(d_lane_ids) < 1:
                continue
            else:
                lane_id = d_lane_ids[-1]
                geo = self._map_lanes[lane_id]["geo"]
                d_matcher.append(
                    {
                        "id": lane_id,
                        "geo": geo,
                        # mid-point
                        "point": geo.interpolate(0.5, normalized=True).coords[:][0],
                        "length": geo.length,
                    }
                )
        w_matcher = []
        for l in self._map_lanes.values():
            if l["type"] == mapv2.LANE_TYPE_WALKING:
                lane_id = l["id"]
                geo = l["geo"]
                w_matcher.append(
                    {
                        "id": lane_id,
                        "geo": geo,
                        "point": geo.interpolate(0.5, normalized=True).coords[:][0],
                        "length": geo.length,
                    }
                )

        (self._map_aois, self._map_pois) = add_sumo_aoi_to_map(
            matchers={
                "drive": d_matcher,
                "walk": w_matcher,
                "road_lane": road_lane_matcher,
            },
            projstr=self._projstr,
            input_aois=aois,
            input_pois=pois,
            input_stops=stops,
            workers=self._workers,
            merge_aoi=self._merge_aoi,
        )

    def convert_map(self):
        self._create_junctions()
        self._add_lane_conn()
        self._add_aois_to_map()
        self._converted_map = self._get_output_map()
        return self._converted_map

    def get_sumo_id_mappings(self):
        """
        return sumo original id -> uid mappings
        """
        return self._id2uid
