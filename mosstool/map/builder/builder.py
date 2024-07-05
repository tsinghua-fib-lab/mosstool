import json
import logging
import time
from collections import defaultdict
from copy import deepcopy
from math import atan2
from multiprocessing import cpu_count
from typing import Callable, Dict, List, Literal, Optional, Set, Tuple, Union, cast

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pycityproto.city.map.v2.map_pb2 as mapv2
import pyproj
import shapely.ops as ops
from geojson import FeatureCollection
from pymongo.collection import Collection
from scipy.spatial import ConvexHull
from shapely.geometry import LineString, MultiPoint, Point
from sklearn.cluster import KMeans

from ...type import Map
from .._map_util.add_aoi_pop import add_aoi_pop
from .._map_util.aoi_matcher import add_aoi_to_map
from .._map_util.aoiutils import generate_aoi_poi
from .._map_util.const import *
from .._map_util.convert_aoi import convert_aoi, convert_poi
from .._map_util.format_checker import geojson_format_check, output_format_check
from .._map_util.gen_traffic_light import generate_traffic_light
from .._map_util.map_aois_matchers import match_map_aois
from .._util.angle import abs_delta_angle, delta_angle
from .._util.line import (
    align_line,
    connect_line_string,
    get_line_angle,
    get_start_vector,
    line_extend,
    line_max_curvature,
    merge_line_start_end,
    offset_lane,
)

__all__ = ["Builder"]


class Builder:
    """
    build map from geojson format files
    """

    def __init__(
        self,
        net: Union[FeatureCollection, Map],
        proj_str: str,
        aois: Optional[FeatureCollection] = None,
        pois: Optional[FeatureCollection] = None,
        public_transport: Optional[Dict[str, List]] = None,
        pop_tif_path: Optional[str] = None,
        landuse_shp_path: Optional[str] = None,
        traffic_light_min_direction_group: int = 3,
        default_lane_width: float = 3.2,
        gen_sidewalk_speed_limit: float = 0,
        expand_roads: bool = False,
        road_expand_mode: Union[Literal["L"], Literal["M"], Literal["R"]] = "R",
        aoi_mode: Union[Literal["append"], Literal["overwrite"]] = "overwrite",
        green_time: float = 30.0,
        yellow_time: float = 5.0,
        strict_mode: bool = False,
        output_lane_length_check: bool = False,
        workers: int = cpu_count(),
    ):
        """
        Args:
        - net (FeatureCollection | Map): road network
        - proj_str (str): projection string
        - aois (FeatureCollection): area of interest
        - pois (FeatureCollection): point of interest
        - public_transport (Dict[str, List]): public transports in json format
        - pop_tif_path (str): path to population tif file
        - landuse_shp_path (str): path to landuse shape file
        - traffic_light_min_direction_group (int): minimum number of lane directions for traffic-light generation
        - default_lane_width (float): default lane width
        - gen_sidewalk_speed_limit (float): speed limit to generate sidewalk
        - expand_roads (bool): expand roads according to junction type
        - road_expand_mode (str): road expand mode
        - aoi_mode (str): aoi appending mode. `append` takes effect when the input `net` is Map, incrementally adding the input AOIs; `overwrite` only adds the input AOIs, ignoring existing ones.
        - green_time (float): green time
        - strict_mode (bool): when enabled, causes the program to exit whenever a warning occurs
        - output_lane_length_check (bool): when enabled, will do value checks lane lengths in output map.
        - yellow_time (float): yellow time
        - workers (int): number of workers
        """
        net_type = type(net)
        self.raw_aois = FeatureCollection(deepcopy(aois))
        self.raw_pois = FeatureCollection(deepcopy(pois))
        self.public_transport = public_transport
        self.default_lane_width = default_lane_width
        self.gen_sidewalk_speed_limit = gen_sidewalk_speed_limit
        self.expand_roads = expand_roads
        self.road_expand_mode = road_expand_mode
        self.aoi_mode = aoi_mode
        self.green_time = green_time
        self.yellow_time = yellow_time
        self.lane_uid = LANE_START_ID
        self.road_uid = ROAD_START_ID
        self.junc_uid = JUNC_START_ID
        self.public_transport_uid = PUBLIC_LINE_START_ID
        self.proj_str = proj_str
        self.projector = pyproj.Proj(proj_str)
        self.pop_tif_path = pop_tif_path
        self.landuse_shp_path = landuse_shp_path
        self.traffic_light_min_direction_group = traffic_light_min_direction_group
        self.strict_mode = strict_mode
        self.output_lane_length_check = output_lane_length_check
        self.workers = workers
        # id mapping relationship
        self.uid_mapping = {}
        """
        Intersection category: (in_cluster, out_cluster) -> []jid
        The junctions that have been processed are deleted from here.
        """
        self._junction_keys = []
        """To draw the junction shape, use to store the number of entry and exit roads of the junction"""

        # processed result
        self.map_roads = {}
        """id -> map road data{[]lane shapely(from left to right), highway, max_speed, name}"""
        self.map_junctions = {}
        """id -> map junction data{[]lane shapely}"""
        self.lane2data = {}
        """lane shapely -> map lane shapely(lane_uid, []in_lane uid, []out_lane uid)"""
        self.map_lanes = {}
        """id -> map lane data(lane shapely)"""
        self.map_aois = {}
        self.map_pois = {}
        self.no_left_walk = set()
        """There is no way id for the left sidewalk"""
        self.no_right_walk = set()
        """There is no way id for the right sidewalk"""
        self.public_transport_data = {"lines": {}, "stations": {}}
        self.from_pb = False
        # Perform coordinate conversion
        all_coords_lonlat = []
        if net_type == FeatureCollection:
            self.net = FeatureCollection(deepcopy(net))
            geojson_format_check(self.net)
            for feature in self.net["features"]:
                if feature["geometry"]["type"] not in ("MultiPoint", "LineString"):
                    raise ValueError("bad geometry type: " + feature)
                if "properties" not in feature:
                    raise ValueError("no properties in feature: " + feature)
                if "id" not in feature:
                    feature["id"] = feature["properties"]["id"]
                all_coords_lonlat.extend(
                    [c[:2] for c in feature["geometry"]["coordinates"]]
                )
                coords = np.array(feature["geometry"]["coordinates"], dtype=np.float64)
                z_coords = (
                    coords[:, 2]
                    if coords.shape[1] > 2
                    else np.zeros((coords.shape[0], 1), dtype=np.float64)
                )
                xy_coords = np.stack(self.projector(*coords.T[:2]), axis=1)  # (N, 2)
                xyz_coords = np.column_stack([xy_coords, z_coords])  # (N, 3)
                feature["geometry"]["coordinates_xy"] = xy_coords
                feature["geometry"]["coordinates_xyz"] = xyz_coords
                if feature["geometry"]["type"] == "LineString":
                    feature["shapely"] = LineString(
                        feature["geometry"]["coordinates_xyz"]
                    )
                    feature["uid"] = self.road_uid
                    self.road_uid += 1
                elif feature["geometry"]["type"] == "MultiPoint":
                    feature["shapely"] = MultiPoint(
                        feature["geometry"]["coordinates_xyz"]
                    )
                    feature["uid"] = self.junc_uid
                    self.junc_uid += 1
            all_coords_lonlat = np.array(all_coords_lonlat)
            self.min_lon, self.min_lat = np.min(all_coords_lonlat, axis=0)
            self.max_lon, self.max_lat = np.max(all_coords_lonlat, axis=0)
            # Classify and store
            self.junctions = {}
            """id -> junction data"""
            self.ways = {}
            """id -> road data"""
            for feature in self.net["features"]:
                if feature["geometry"]["type"] == "MultiPoint":
                    self.junctions[feature["id"]] = feature
                    self.uid_mapping[feature["uid"]] = feature["id"]
                elif feature["geometry"]["type"] == "LineString":
                    self.ways[feature["id"]] = feature
                    self.uid_mapping[feature["uid"]] = feature["id"]
            self.junction_types = defaultdict(list)
        elif net_type == Map:
            logging.info("Reading from pb files")
            self.net = net
            self.from_pb = True
            pb_lane_uids = set()
            # map_lanes & lane2data
            for l in net.lanes:
                line = LineString([[n.x, n.y, n.z] for n in l.center_line.nodes])
                l_id = l.id
                self.map_lanes[l_id] = line
                pb_lane_uids.add(l_id)
                self.lane2data[line] = {
                    "uid": l_id,
                    "in": [
                        {
                            "id": conn.id,
                            "type": conn.type,
                        }
                        for conn in l.predecessors
                    ],
                    "out": [
                        {
                            "id": conn.id,
                            "type": conn.type,
                        }
                        for conn in l.successors
                    ],
                    "max_speed": l.max_speed,
                    "left_lane_ids": [],
                    "right_lane_ids": [],
                    "parent_id": l.parent_id,
                    "type": l.type,
                    "turn": l.turn,
                    "width": l.width,
                }
            self.lane_uid = max(pb_lane_uids) + 1
            # map_roads
            pb_road_uids = set()
            for r in net.roads:
                road_lanes = [self.map_lanes[l_id] for l_id in r.lane_ids]
                drive_lanes = [
                    l
                    for l in road_lanes
                    if self.lane2data[l]["type"] == mapv2.LANE_TYPE_DRIVING
                ]
                if len(drive_lanes) >= 1:
                    road_angle = np.mean([get_line_angle(l) for l in drive_lanes])
                else:
                    road_angle = get_line_angle(road_lanes[-1])
                walk_lanes = [
                    l
                    for l in road_lanes
                    if self.lane2data[l]["type"] == mapv2.LANE_TYPE_WALKING
                ]
                r_id = r.id
                pb_road_uids.add(r_id)
                self.map_roads[r_id] = {
                    "lanes": drive_lanes,
                    "left_sidewalk": [
                        l
                        for l in walk_lanes
                        if abs_delta_angle(get_line_angle(l), road_angle) >= np.pi / 2
                    ],
                    "right_sidewalk": [
                        l
                        for l in walk_lanes
                        if abs_delta_angle(get_line_angle(l), road_angle) < np.pi / 2
                    ],
                    "highway": "",
                    "name": r.name,
                    "uid": r_id,
                }
            self.road_uid = max(pb_road_uids) + 1
            # map_junctions
            pb_junc_uids = set()
            for j in net.junctions:
                junc_lanes = [self.map_lanes[l_id] for l_id in j.lane_ids]
                all_junc_lane_coords = []
                for l in junc_lanes:
                    all_junc_lane_coords.extend(list(l.coords))
                all_junc_lane_coords = np.array(all_junc_lane_coords)
                if all_junc_lane_coords.shape[1] == 3:
                    x_center, y_center, z_center = np.mean(all_junc_lane_coords, axis=0)
                else:
                    x_center, y_center = np.mean(all_junc_lane_coords, axis=0)
                    z_center = 0
                j_id = j.id
                pb_junc_uids.add(j_id)
                self.map_junctions[j_id] = {
                    "lanes": junc_lanes,
                    "uid": j_id,
                    "center": {
                        "x": x_center,
                        "y": y_center,
                        "z": z_center,
                    },
                }
            self.junc_uid = max(pb_junc_uids) + 1
            # bbox
            self.max_lon, self.max_lat = self.projector(
                net.header.east, net.header.north, inverse=True
            )
            self.min_lon, self.min_lat = self.projector(
                net.header.west, net.header.south, inverse=True
            )
        else:
            raise ValueError(f"Unsupported data type {net_type}")

        # output data
        self.output_lanes = {}
        self.output_roads = {}
        self.output_junctions = {}
        self.output_aois = {}
        self.output_pois = {}

    def _connect_lane_group(
        self,
        in_lanes: List[LineString],
        out_lanes: List[LineString],
        lane_turn: mapv2.LaneTurn,
        lane_type: mapv2.LaneType,
        junc_id: int,
        in_walk_type: Union[Literal["in_way"], Literal["out_way"], Literal[""]] = "",
        out_walk_type: Union[Literal["in_way"], Literal["out_way"], Literal[""]] = "",
    ) -> List[LineString]:
        """
        Connect two lanes
        """
        # Method: After aligning the center, connect to the nearest lane (if the values are the same, connect both)
        # | | |    =>  | | |
        #             /\/\/\
        # | | | |  => | | | |
        results = []
        in_center = (len(in_lanes) - 1) / 2
        in_offsets = np.array(
            [[i - in_center for i in range(len(in_lanes))]],
            dtype=np.float64,
        )  # shape: (1, M)
        out_center = (len(out_lanes) - 1) / 2
        out_offsets = np.array(
            [[i - out_center for i in range(len(out_lanes))]],
            dtype=np.float64,
        )  # shape: (1, N)
        deltas = np.abs(in_offsets.T - out_offsets)  # shape: (M, N)
        min_each_in = np.min(deltas, axis=1, keepdims=True)  # shape: (M, 1)
        pair = np.where((deltas - min_each_in) < EPS)
        for i, j in zip(pair[0], pair[1]):
            in_lane = in_lanes[cast(int, i)]
            in_uid = self.lane2data[in_lane]["uid"]
            out_lane = out_lanes[cast(int, j)]
            out_uid = self.lane2data[out_lane]["uid"]
            # Lanes are connected with Bezier curves and sidewalks are connected with straight lines
            conn_lane = (
                connect_line_string(in_lane, out_lane)
                if lane_type != mapv2.LANE_TYPE_WALKING
                else LineString([in_lane.coords[-1], out_lane.coords[0]])
            )
            # limit the junc lane length
            if not in_walk_type and not out_walk_type:
                if conn_lane.length > 1e99:
                    continue
            else:
                if conn_lane.length > 1e99:
                    continue
            # Add new lane
            self.map_lanes[self.lane_uid] = conn_lane
            # Add the connection relationship of the new lane
            # When all are roadways
            if not in_walk_type and not out_walk_type:
                # Add the connection relationship of in_lane
                self.lane2data[in_lane]["out"].append(
                    {"id": self.lane_uid, "type": mapv2.LANE_CONNECTION_TYPE_HEAD}
                )
                # Add the connection relationship of out_lane
                self.lane2data[out_lane]["in"].append(
                    {"id": self.lane_uid, "type": mapv2.LANE_CONNECTION_TYPE_TAIL}
                )
                lane_conn_in = [
                    {
                        "id": in_uid,
                        "type": (mapv2.LANE_CONNECTION_TYPE_TAIL),
                    },
                ]
                lane_conn_out = [
                    {
                        "id": out_uid,
                        "type": (mapv2.LANE_CONNECTION_TYPE_HEAD),
                    },
                ]
                in_lane_width = self.lane2data[in_lane]["width"]
                out_lane_width = self.lane2data[out_lane]["width"]
                cur_lane_width = 0.5 * (in_lane_width + out_lane_width)
            else:
                # The connection relationship of the sidewalk will be added later.
                lane_conn_in = []
                lane_conn_out = []
                cur_lane_width = DEFAULT_JUNCTION_WALK_LANE_WIDTH
            cur_max_speed = 0.5 * (
                self.lane2data[in_lane]["max_speed"]
                + self.lane2data[out_lane]["max_speed"]
            )
            self.lane2data[conn_lane] = {
                "uid": self.lane_uid,
                "in": lane_conn_in,
                "out": lane_conn_out,
                "max_speed": cur_max_speed,
                "left_lane_ids": [],
                "right_lane_ids": [],
                "parent_id": junc_id,
                "type": lane_type,
                "turn": lane_turn,
                "width": cur_lane_width,
            }
            self.lane_uid += 1
            results.append(conn_lane)
        return results

    def _delete_lane(self, lane_id: int, delete_road: bool = False) -> None:
        """
        Delete the lane. If it belongs to road, the road will also be deleted. If it belongs to the junction, delete the lane from the junction.
        """
        line = self.map_lanes[lane_id]
        l_data = self.lane2data[line]
        parent_id = l_data["parent_id"]
        # Delete connection relationship
        for lane_conn in l_data["in"]:
            in_uid = lane_conn["id"]
            if in_uid in self.map_lanes:
                in_line = self.map_lanes[in_uid]
                in_l_data = self.lane2data[in_line]
                in_l_data["out"] = [
                    d for d in in_l_data["out"] if not d["id"] == lane_id
                ]
        for lane_conn in l_data["out"]:
            out_uid = lane_conn["id"]
            if out_uid in self.map_lanes:
                out_line = self.map_lanes[out_uid]
                out_l_data = self.lane2data[out_line]
                out_l_data["in"] = [
                    d for d in out_l_data["in"] if not d["id"] == lane_id
                ]
        if parent_id >= JUNC_START_ID:  # junc lane
            for _, junc in self.map_junctions.items():
                junc_uid = junc["uid"]
                if junc_uid == parent_id:
                    junc["lanes"] = [l for l in junc["lanes"] if not l == line]
                    break
        else:  # road lane
            for way_id, map_road in self.map_roads.items():
                way_uid = map_road["uid"]
                if way_uid == parent_id:
                    if delete_road:
                        del self.map_roads[way_id]
                    else:
                        road = self.map_roads[way_id]
                        road["lanes"] = [l for l in road["lanes"] if not l == line]
                        road["left_sidewalk"] = [
                            l for l in road["left_sidewalk"] if not l == line
                        ]
                        road["right_sidewalk"] = [
                            l for l in road["right_sidewalk"] if not l == line
                        ]
                    break
        del self.map_lanes[lane_id]
        del self.lane2data[line]

    def _reset_lane_uids(
        self, orig_lane_uids: List[int], new_lane_uids: List[int]
    ) -> None:
        """
        Reset lane uid
        """
        assert len(orig_lane_uids) == len(
            new_lane_uids
        ), "Different lane ids num between orig_lane_uids and new_lane_uids"
        new_map_lanes = {}
        orig_id2new_id = {
            orig_lane_uids[i]: new_lane_uids[i] for i in range(len(orig_lane_uids))
        }
        for lane_id, line in self.map_lanes.items():
            if lane_id in orig_id2new_id:
                new_lane_uid = orig_id2new_id[lane_id]
            else:
                new_lane_uid = lane_id
            l_data = self.lane2data[line]
            l_data["uid"] = new_lane_uid
            # Rewrite the connection relationship
            for lane_conn in l_data["in"]:
                in_uid = lane_conn["id"]
                if in_uid in orig_id2new_id:
                    new_in_uid = orig_id2new_id[in_uid]
                    lane_conn["id"] = new_in_uid
            for lane_conn in l_data["out"]:
                out_uid = lane_conn["id"]
                if out_uid in orig_id2new_id:
                    new_out_uid = orig_id2new_id[out_uid]
                    lane_conn["id"] = new_out_uid
            new_map_lanes[new_lane_uid] = line
        self.map_lanes = new_map_lanes

    def draw_junction(self, jid: int, save_path: str, trim_length: float = 50):
        """
        Draw the junction as a picture and save it in save_path
        Draw junction to image and save to save_path

        Args:
        - jid (int): junction id
        - save_path (str): path to save image
        - trim_length (float): length of the road to draw

        Returns:
        - None
        """
        j = self.junctions[jid]
        in_way_groups = j["properties"]["in_way_groups"]
        out_way_groups = j["properties"]["out_way_groups"]
        # set colormap
        colormap = plt.colormaps.get_cmap("tab20")
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        for i, (_, group) in enumerate(in_way_groups):
            for way_id in group:
                color = colormap(i * 2)
                road = self.map_roads[way_id]
                road_driving_lanes = [
                    l
                    for l in road["lanes"]
                    if self.lane2data[l]["type"] == mapv2.LANE_TYPE_DRIVING
                ]
                assert (
                    len(road_driving_lanes) > 0
                ), f"Driving lanes at {way_id} is equal to 0"
                for line in road_driving_lanes:
                    line = cast(LineString, line)
                    # 50 meters after the broken line
                    length = line.length
                    seg_length = min(trim_length, length)
                    show_line = ops.substring(line, length - seg_length, length)
                    ax.plot(
                        *show_line.coords.xy,
                        color=color,
                        label=f"in-{i}-{way_id}-*{len(road_driving_lanes)}",
                    )
        for i, (_, group) in enumerate(out_way_groups):
            for way_id in group:
                color = colormap(i * 2 + 1)
                road = self.map_roads[way_id]
                road_driving_lanes = [
                    l
                    for l in road["lanes"]
                    if self.lane2data[l]["type"] == mapv2.LANE_TYPE_DRIVING
                ]
                for line in road_driving_lanes:
                    line = cast(LineString, line)
                    # Take the first 50 meters of the polyline
                    length = line.length
                    seg_length = min(trim_length, length)
                    show_line = ops.substring(line, 0, seg_length)
                    ax.plot(
                        *show_line.coords.xy,
                        linestyle="dashed",
                        color=color,
                        label=f"out-{i}-{way_id}-*{len(road_driving_lanes)}",
                    )
        if jid in self.map_junctions:
            map_j = self.map_junctions[jid]
            junc_driving_lanes = [
                l
                for l in map_j["lanes"]
                if self.lane2data[l]["type"] == mapv2.LANE_TYPE_DRIVING
            ]
            for line in junc_driving_lanes:
                line = cast(LineString, line)
                ax.plot(
                    *line.coords.xy,
                    color="black",
                    linestyle="dotted",
                )
        ax.axis("equal")
        ax.grid(True)
        ax.set_title("junction {}".format(jid))
        ax.legend()
        fig.savefig(save_path)
        plt.close(fig)

    def draw_walk_junction(self, jid: int, save_path: str, trim_length: float = 50):
        """
        Draw the junction as a picture and save it in save_path
        Draw junction to image and save to save_path (for walking lane)

        Args:
        - jid (int): junction id
        - save_path (str): path to save image
        - trim_length (float): length of the road to draw

        Returns:
        - None
        """
        j = self.junctions[jid]
        in_way_groups = j["properties"]["in_way_groups"]
        out_way_groups = j["properties"]["out_way_groups"]
        coord_xys = j["geometry"]["coordinates_xy"]
        # set colormap
        colormap = plt.colormaps.get_cmap("tab20")
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        if jid in self.map_junctions:
            map_j = self.map_junctions[jid]
            wlane_half_length = [
                50 + l.length / 2
                for l in map_j["lanes"]
                if self.lane2data[l]["type"] == mapv2.LANE_TYPE_WALKING
            ]
        else:
            return
        trim_length = max([trim_length] + wlane_half_length)
        x0, y0 = np.mean(coord_xys, axis=0)
        ax.set_xlim(x0 - trim_length, x0 + trim_length)
        ax.set_ylim(y0 - trim_length, y0 + trim_length)
        ax.scatter(x0, y0, color="red")
        has_legend = False
        for i, (_, group) in enumerate(in_way_groups):
            for way_id in group:
                color = colormap(i * 2)
                in_uid = self.wid2ruid[way_id]
                in_r = self.output_roads[in_uid]
                in_ls = [self.map_lanes[lid] for lid in in_r["lane_ids"]]
                walk_lanes = [
                    l
                    for l in in_ls
                    if self.lane2data[l]["type"] == mapv2.LANE_TYPE_WALKING
                ]
                for line in walk_lanes:
                    has_legend = True
                    line = cast(LineString, line)
                    show_line = line
                    ax.plot(
                        *show_line.coords.xy,
                        linewidth=2,
                        color=color,
                        label=f"in-{i}-{way_id}-{self.wid2ruid[way_id]}",
                    )
                drive_lanes = [
                    l
                    for l in in_ls
                    if self.lane2data[l]["type"] == mapv2.LANE_TYPE_DRIVING
                ]
                for line in drive_lanes:
                    line = cast(LineString, line)
                    show_line = line
                    ax.plot(*show_line.coords.xy, color=color)
        for i, (_, group) in enumerate(out_way_groups):
            for way_id in group:
                color = colormap(i * 2 + 1)
                out_uid = self.wid2ruid[way_id]
                out_r = self.output_roads[out_uid]
                out_ls = [self.map_lanes[lid] for lid in out_r["lane_ids"]]
                walk_lanes = [
                    l
                    for l in out_ls
                    if self.lane2data[l]["type"] == mapv2.LANE_TYPE_WALKING
                ]
                for line in walk_lanes:
                    has_legend = True
                    line = cast(LineString, line)
                    show_line = line
                    ax.plot(
                        *show_line.coords.xy,
                        linewidth=2,
                        linestyle="dashed",
                        label=f"out-{i}-{way_id}-{self.wid2ruid[way_id]}",
                        color=color,
                    )
                road_driving_lanes = [
                    l
                    for l in out_ls
                    if self.lane2data[l]["type"] == mapv2.LANE_TYPE_DRIVING
                ]
                for line in road_driving_lanes:
                    line = cast(LineString, line)
                    show_line = line
                    ax.plot(*show_line.coords.xy, linestyle="dashed", color=color)
        if jid in self.map_junctions:
            map_j = self.map_junctions[jid]
            junc_walking_lanes = [
                l
                for l in map_j["lanes"]
                if self.lane2data[l]["type"] == mapv2.LANE_TYPE_WALKING
            ]
            for line in junc_walking_lanes:
                line = cast(LineString, line)
                ax.plot(
                    *line.coords.xy,
                    color="black",
                    linestyle="dotted",
                )
        ax.grid(True)
        ax.set_title("junction {}".format(jid))
        if has_legend:
            ax.legend()
        fig.savefig(save_path)
        plt.close(fig)

    def _classify(self):
        """
        Classify roads entering and leaving junctions
        1. First use KMeans for angle clustering
        """
        logging.info(f"Classifying junctions")

        def do_cluster(norm_vectors):
            assert (
                len(norm_vectors) >= 1
            ), f"norm_vectors should have at least one element, but {norm_vectors}"
            results = []  # (cluster, error, kmeans)
            for cluster in range(2, 5):
                if cluster > len(norm_vectors):
                    break
                kmeans = KMeans(n_clusters=cluster, n_init="auto", random_state=0)
                kmeans.fit(norm_vectors)
                # Calculate clustering error: cosine distance between cluster center and points of the same category
                error = cluster * CLUSTER_PENALTY
                for i in range(cluster):
                    center = kmeans.cluster_centers_[i]
                    center /= np.linalg.norm(center)
                    points = norm_vectors[kmeans.labels_ == i]
                    for point in points:
                        error += 1 - abs(np.dot(center, point))
                results.append((cluster, error, kmeans))
            min_result = min(results, key=lambda x: x[1])
            # Select the cluster with the largest error within the allowable range
            result = max(
                [r for r in results if np.abs(r[1] - min_result[1]) < UP_THRESHOLD],
                key=lambda x: x[0],
            )
            return result

        def groupby(
            in_way_ids: List[int],
            in_get_vector: Callable[[LineString], np.ndarray],
            out_way_ids: List[int],
            out_get_vector: Callable[[LineString], np.ndarray],
        ) -> Tuple[
            List[Tuple[np.ndarray, List[int]]], List[Tuple[np.ndarray, List[int]]]
        ]:
            norm_vectors = []
            for wid in in_way_ids:
                way = self.ways[wid]
                line = cast(LineString, way["shapely"])
                vector = in_get_vector(line)[:2]
                n_vec = vector / np.linalg.norm(vector)
                if any(np.isnan(x) for x in n_vec):
                    logging.warning(f"Invalid shapely at way_id: {wid}")
                norm_vectors.append(n_vec)
            for wid in out_way_ids:
                way = self.ways[wid]
                line = cast(LineString, way["shapely"])
                vector = out_get_vector(line)[:2]
                n_vec = vector / np.linalg.norm(vector)
                if any(np.isnan(x) for x in n_vec):
                    logging.warning(f"Invalid shapely at way_id: {wid}")
                norm_vectors.append(n_vec)
            norm_vectors = np.array(norm_vectors)
            cluster, _, kmeans = do_cluster(norm_vectors)  # (cluster, error, kmeans)
            in_way_groups = []  # (angle, []wid)
            out_way_groups = []  # (angle, []wid)
            for i in range(cluster):
                v = kmeans.cluster_centers_[i]
                angle = np.arctan2(v[1], v[0])
                in_way_groups.append((angle, []))
                out_way_groups.append((angle, []))
            for i, wid in enumerate(in_way_ids):
                in_way_groups[kmeans.labels_[i]][1].append(wid)
            for i, wid in enumerate(out_way_ids):
                out_way_groups[kmeans.labels_[i + len(in_way_ids)]][1].append(wid)
            # Clear clusters with empty []wid
            in_way_groups = [
                (angle, wids) for angle, wids in in_way_groups if len(wids) > 0
            ]
            out_way_groups = [
                (angle, wids) for angle, wids in out_way_groups if len(wids) > 0
            ]
            return in_way_groups, out_way_groups

        for jid, j in self.junctions.items():
            in_way_ids = j["properties"]["in_ways"]
            out_way_ids = j["properties"]["out_ways"]
            if len(in_way_ids) == 0 and len(out_way_ids) > 0:
                continue
            if len(in_way_ids) > 0 and len(out_way_ids) == 0:
                continue
            assert len(in_way_ids) >= 1, f"Junction {jid} has 0 in_way_id"
            assert len(out_way_ids) >= 1, f"Junction {jid} has 0 out_way_id"
            # =======================================================
            in_way_groups, out_way_groups = groupby(
                in_way_ids,
                lambda line: np.array(line.coords[-1])
                - np.array(line.coords[4 * len(line.coords) // 5 - 1]),
                out_way_ids,
                lambda line: np.array(line.coords[1 + len(line.coords) // 5])
                - np.array(line.coords[0]),
            )
            # =======================================================
            j["properties"]["in_way_groups"] = in_way_groups
            j["properties"]["out_way_groups"] = out_way_groups
            typ = (len(in_way_groups), len(out_way_groups))
            self.junction_types[typ].append(j["id"])

    def _classify_main_way_ids(self):
        """
        Return the main road and auxiliary road in wids according to in_way_group/out_way_group
        """

        def get_main_in_wids(way_angle, wids):
            vec = np.array(
                [
                    np.cos(way_angle + np.pi / 2),
                    np.sin(way_angle + np.pi / 2),
                ]
            )
            right_lanes = {
                wid: cast(LineString, self.ways[wid]["shapely"]) for wid in wids
            }
            right_vecs = [
                {
                    "wid": wid,
                    "vec": np.array(l.coords[-1][:2]),
                }
                for wid, l in right_lanes.items()
            ]
            # The smaller the inner product, the closer it is to the right
            sorted_wids = [
                vec["wid"]
                for vec in sorted(right_vecs, key=lambda x: -np.dot(x["vec"], vec))
            ]
            return (
                sorted_wids[0],
                sorted_wids[1:],
            )

        def get_main_out_wids(way_angle, wids):
            vec = np.array(
                [
                    np.cos(way_angle + np.pi / 2),
                    np.sin(way_angle + np.pi / 2),
                ]
            )
            left_lanes = {
                wid: cast(LineString, self.ways[wid]["shapely"]) for wid in wids
            }
            left_vecs = [
                {
                    "wid": wid,
                    "vec": np.array(l.coords[0][:2]),
                }
                for wid, l in left_lanes.items()
            ]
            sorted_wids = [
                vec["wid"]
                for vec in sorted(left_vecs, key=lambda x: -np.dot(x["vec"], vec))
            ]
            return (
                sorted_wids[0],
                sorted_wids[1:],
            )

        self.auxiliary_wids = set()
        for _, j in self.junctions.items():
            in_way_ids = j["properties"]["in_ways"]
            out_way_ids = j["properties"]["out_ways"]
            if len(in_way_ids) == 0 or len(out_way_ids) == 0:
                continue
            in_way_groups = j["properties"]["in_way_groups"]
            for in_angle, in_way_ids in in_way_groups:
                _, auxiliary_wids = get_main_in_wids(in_angle, in_way_ids)
                for wid in auxiliary_wids:
                    self.auxiliary_wids.add(wid)
            out_way_groups = j["properties"]["out_way_groups"]
            for out_angle, out_way_ids in out_way_groups:
                _, auxiliary_wids = get_main_out_wids(out_angle, out_way_ids)
                for wid in auxiliary_wids:
                    self.auxiliary_wids.add(wid)

    def _expand_roads(
        self,
        wids: List[int],
        junc_type,  # The number of identified entry and exit roads in the junction is a parameter used to calculate the shortened length of the road.
        junc_id: int,
        way_type: Union[
            Literal["main"],
            Literal["around"],
            Literal["right"],
            Literal["left"],
            Literal[""],
        ] = "",
    ):
        """
        Extension road
        1. Expand the lane of the specified wid
        2. The front and rear inward contraction is proportional to the width of the lane + the distance to the number of roads entering and exiting the junction.
        """
        way_num = sum(junc_type)
        for wid in wids:
            # Expanded operations will not be repeated
            if wid in self.map_roads:
                map_road = self.map_roads[wid]
                conn_junc_ids = map_road["conn_junc_ids"]
                conn_junc_ids.append(junc_id)
                continue
            way = self.ways[wid]
            line = cast(LineString, way["shapely"])
            orig_line = line
            # Calculate lane centerline
            lane_num = way["properties"].get("lanes", DEFAULT_LANE_NUM)
            # Lane width attribute
            lane_width = way["properties"].get(
                "width", way["properties"].get("lanewidth", self.default_lane_width)
            )
            # Expansion is only performed when initially processing topo.geojson. Subsequent iterations should not be expanded.
            if self.expand_roads:
                if way_type == "main":
                    if lane_num <= 2:
                        lane_num += 2
                if way_type == "around":
                    if lane_num <= 2:
                        lane_num += 2
                if way_type == "right":
                    pass
                if way_type == "left":
                    pass
            # width = lane_num * lane_width + way_num / 1.5 * lane_width
            width = DEFAULT_ROAD_SPLIT_LENGTH
            if wid in self.auxiliary_wids:
                width += 5
            if line.length > 3 * width:
                line = ops.substring(line, width, line.length - width)
            else:
                line = ops.substring(line, line.length * 0.4, line.length * 0.6)
            line = cast(LineString, line.simplify(0.1))
            if self.road_expand_mode == "L":
                # Expand to the left
                lanes = [
                    offset_lane(line, -(i - lane_num + 0.5) * lane_width)
                    for i in range(lane_num)
                ]
            elif self.road_expand_mode == "M":
                # Center expansion
                lanes = [
                    offset_lane(
                        line, -(i - lane_num / 2 + 0.495) * lane_width
                    )  # Make a slight offset to the center line to prevent two completely overlapping lanes in opposite directions from still completely overlapping after expansion.
                    for i in range(lane_num)
                ]
            else:
                # Expand to the right
                lanes = [
                    offset_lane(line, -(i + 0.5) * lane_width) for i in range(lane_num)
                ]

            # way["properties"]["lanes"] = lane_num # Update lane number
            for lane in lanes:
                # Add new lane
                self.map_lanes[self.lane_uid] = lane
                # Add the connection relationship of the new lane
                self.lane2data[lane] = {
                    "uid": self.lane_uid,
                    "in": [],
                    "out": [],
                    "max_speed": way["properties"]["max_speed"],
                    "type": mapv2.LANE_TYPE_DRIVING,
                    "turn": mapv2.LANE_TURN_STRAIGHT,
                    "width": lane_width,
                    "left_lane_ids": [],
                    "right_lane_ids": [],
                    "parent_id": way["uid"],
                }
                self.lane_uid += 1
            # Used for junction to generate sidewalks

            def get_in_angle(line):
                def in_get_vector(line):
                    return np.array(line.coords[-1]) - np.array(
                        line.coords[4 * len(line.coords) // 5 - 1]
                    )

                v = in_get_vector(line)
                return np.arctan2(v[1], v[0])

            # ATTENTION: This out angle is the angle from the middle of the line to the starting point of the line

            def get_out_angle(line):
                def out_get_vector(line):
                    return np.array(line.coords[0]) - np.array(
                        line.coords[1 + len(line.coords) // 5]
                    )

                v = out_get_vector(line)
                return np.arctan2(v[1], v[0])

            self.map_roads[way["id"]] = {
                "lanes": lanes,
                "left_sidewalk": [],
                "right_sidewalk": [],
                "start_end_left_side_walk": {},
                "start_end_right_side_walk": {},
                "highway": way["properties"]["highway"],
                "max_speed": way["properties"]["max_speed"],
                "name": way["properties"]["name"],
                "in_angle": get_in_angle(lanes[0]),
                "out_angle": get_out_angle(lanes[0]),
                "uid": way["uid"],
                "turn_config": way["properties"].get(
                    "turn", []
                ),  # Manually allocated turns
                "walk_lane_offset": way["properties"].get(
                    "walk_lane_offset", 0
                ),  # Additional extension of walking lanes
                "walk_lane_width": way["properties"].get("walk_lane_width", lane_width),
                "conn_junc_ids": [
                    junc_id,
                ],
            }

    def _expand_remain_roads(self):
        """
        Expand the remaining roads
        1. The front and rear shrink inward by a distance equal to the width of the lane (different from the road connected to the junction)
        """
        for way in self.ways.values():
            if way["id"] in self.map_roads:
                continue
            self._expand_roads(
                wids=[way["id"]], junc_type=(0, 0), junc_id=-1, way_type=""
            )

    def _add_sidewalk(
        self,
        wid,
        lane: LineString,
        other_lane: LineString,
        walk_type: Union[Literal["left"], Literal["right"]],
        walk_lane_end_type: Union[Literal["start"], Literal["end"]],
    ):
        """Add sidewalk"""
        if walk_type == "right":
            # Currently only one sidewalk is generated
            if walk_lane_end_type in self.map_roads[wid]["start_end_right_side_walk"]:
                return self.map_roads[wid]["start_end_right_side_walk"][
                    walk_lane_end_type
                ]
            walk_lane_width = self.map_roads[wid]["walk_lane_width"]
            lane_offset = (
                self.lane2data[lane]["width"] + walk_lane_width
            ) * 0.5 + self.map_roads[wid]["walk_lane_offset"]
            walk_lane = offset_lane(lane, -lane_offset)
            # Align with the lane of another road
            walk_lane = align_line(walk_lane, other_lane)
            # Extend before and after 1.8*default_lane_width
            walk_lane = line_extend(walk_lane, 1.8 * self.default_lane_width)
            # Add the connection relationship of the new lane
            self.lane2data[walk_lane] = {
                "uid": self.lane_uid,
                "in": [],
                "out": [],
                "max_speed": DEFAULT_MAX_SPEED["WALK"],
                "type": mapv2.LANE_TYPE_WALKING,
                "turn": mapv2.LANE_TURN_STRAIGHT,
                "width": walk_lane_width,
                "left_lane_ids": [],
                "right_lane_ids": [],
                "parent_id": self.map_roads[wid]["uid"],
            }
            self.map_roads[wid]["start_end_right_side_walk"][
                walk_lane_end_type
            ] = walk_lane
        elif walk_type == "left":
            if walk_lane_end_type in self.map_roads[wid]["start_end_left_side_walk"]:
                return self.map_roads[wid]["start_end_left_side_walk"][
                    walk_lane_end_type
                ]
            walk_lane_width = self.map_roads[wid]["walk_lane_width"]
            lane_offset = (
                self.lane2data[lane]["width"] + walk_lane_width
            ) * 0.5 + self.map_roads[wid]["walk_lane_offset"]
            # Move left and reverse direction
            walk_lane = LineString(
                [coord for coord in offset_lane(lane, lane_offset).coords[::-1]]
            )
            # If it is too short, the offset will be empty.
            if not walk_lane:
                return None
            # Align with the lane of another road
            walk_lane = align_line(walk_lane, other_lane)
            # Extend before and after 1.8*default_lane_width
            walk_lane = line_extend(walk_lane, 1.8 * self.default_lane_width)
            # Add the connection relationship of the new lane
            self.lane2data[walk_lane] = {
                "uid": self.lane_uid,
                "in": [],
                "out": [],
                "max_speed": DEFAULT_MAX_SPEED["WALK"],
                "type": mapv2.LANE_TYPE_WALKING,
                "turn": mapv2.LANE_TURN_STRAIGHT,
                "width": walk_lane_width,
                "left_lane_ids": [],
                "right_lane_ids": [],
                "parent_id": self.map_roads[wid]["uid"],
            }
            self.map_roads[wid]["start_end_left_side_walk"][
                walk_lane_end_type
            ] = walk_lane
        else:
            raise ValueError(f"walk_type {walk_type} not supported")
        return walk_lane

    def _create_junction_walk_pairs(
        self,
        in_way_groups: Tuple[List[Tuple[np.ndarray, List[int]]]],
        out_way_groups: Tuple[List[Tuple[np.ndarray, List[int]]]],
        has_main_group_wids: set,
        junc_center: Tuple[float, float],
    ):
        # Create walking lanes
        # rule:
        # 1. There is an in/out way in the same direction
        # Then in way and out way construct the sidewalk on the right
        # | |⬆ out way
        # | |⬆
        #
        # | |⬆
        # | |⬆ in way
        # 2. Separate in/out way
        # Then in/out way constructs the sidewalk on the left and right
        # ⬇| |⬆
        # ⬇| |⬆ in/out way
        walk_group = (
            []
        )  # Store the roads that generate sidewalks [[]angle,[]in_way_ids,[]out_way_ids,]

        def filter_way_ids(wids):
            """Filtering can generate the way id of the sidewalk"""
            return [
                wid
                for wid in wids
                if (
                    self.map_roads[wid]["highway"] in HAS_WALK_LANES_HIGHWAY
                    or self.map_roads[wid]["max_speed"] <= self.gen_sidewalk_speed_limit
                )
                and self.map_roads[wid]["lanes"][0].length >= MIN_HAS_WALK_LANE_LENGTH
            ]

        def get_lane(wid):
            """Returns the leftmost lane corresponding to wid"""
            return self.map_roads[wid]["lanes"][0]

        for _, in_way_ids in in_way_groups:
            has_walk_wids = filter_way_ids(in_way_ids)
            if not has_walk_wids:
                continue
            in_angle = np.mean(
                [self.map_roads[wid]["in_angle"] for wid in has_walk_wids]
            )
            for (
                angles,
                in_ids,
                _,
            ) in walk_group:  # Determine whether these roads are in the same direction
                if all(
                    np.abs(delta_angle(a, in_angle)) < SAME_DIREC_THRESHOLD
                    for a in angles
                ):
                    angles.append(in_angle)
                    in_ids += has_walk_wids
                    break
            else:
                walk_group.append(
                    [
                        [in_angle],
                        has_walk_wids,
                        [],
                    ]
                )
        for _, out_way_ids in out_way_groups:
            has_walk_wids = filter_way_ids(out_way_ids)
            if not has_walk_wids:
                continue
            out_angle = np.mean(
                [self.map_roads[wid]["out_angle"] for wid in has_walk_wids]
            )
            for angles, _, out_ids in walk_group:
                if all(
                    np.abs(delta_angle(a, out_angle)) < SAME_DIREC_THRESHOLD
                    for a in angles
                ):
                    angles.append(out_angle)
                    out_ids += has_walk_wids
                    break
            else:
                walk_group.append(
                    [
                        [out_angle],
                        [],
                        has_walk_wids,
                    ]
                )
        # Sort walk group counterclockwise
        # Because some ways cannot simply use the angle of the direction vector to determine the counterclockwise order.
        # Use the direction vector to sort way 0. The angle is smaller than way 1. In fact, way 1 is in the clockwise direction of way 0.
        #                     \ \  / /
        #                      \ \/ /
        #                       \/\/
        #                    ↙ /\/\  ↘
        #                way 0 / /\ \ way 1
        #
        #                          junc center
        # Changing the angle to the angle where the end of the road points to the center of the junction can solve the problem
        # Use convex hull here to achieve counterclockwise sorting

        if len(walk_group) >= 3:
            way_vertices = []
            for _, in_ids, out_ids in walk_group:
                if in_ids:
                    way_vertices.append(get_lane(in_ids[0]).coords[:][-1][:2])
                    continue
                if out_ids:
                    way_vertices.append(get_lane(out_ids[0]).coords[:][0][:2])
                    continue
            hull = ConvexHull(way_vertices)
            sorted_index = [
                hull.vertices[i] for i in range(hull.vertices.shape[0])
            ]  # Walk group subscript sorted counterclockwise
            if len(sorted_index) < len(walk_group):
                walk_group = sorted(
                    walk_group, key=lambda x: np.mean(x[0])
                )  # There are way vertices inside the convex hull, sorted by angle
            else:
                walk_group = [walk_group[i] for i in sorted_index]
        else:
            walk_group = sorted(
                walk_group, key=lambda x: np.mean(x[0])
            )  # There are too few vertices to form a convex hull. Sort by angle.
        junc_center_point = Point(junc_center)

        def get_side_wid(wids, way_angle, walk_type):
            """
            Find the left/right lane from the lane corresponding to wids and return to wid
            If the distance between the wid and the junction center is too different, select the farthest one.
            """
            center_lanes = {
                wid: self.map_roads[wid]["lanes"][
                    len(self.map_roads[wid]["lanes"]) // 2
                ]
                for wid in wids
            }
            center_lane_dis = [
                {"wid": wid, "dis": l.distance(junc_center_point)}
                for wid, l in center_lanes.items()
            ]
            max_dis_lane = max(center_lane_dis, key=lambda x: x["dis"])
            min_dis_lane = min(center_lane_dis, key=lambda x: x["dis"])
            if max_dis_lane["dis"] - min_dis_lane["dis"] > MAX_WAY_DIS_DIFFERENCE:
                return max_dis_lane["wid"]
            else:
                if walk_type == "right":
                    vec = np.array(
                        [
                            np.cos(way_angle + np.pi / 2),
                            np.sin(way_angle + np.pi / 2),
                        ]
                    )
                    right_lanes = {
                        wid: self.map_roads[wid]["lanes"][-1] for wid in wids
                    }
                    right_vecs = [
                        {
                            "wid": wid,
                            "vec": np.array(l.coords[-1][:2]),
                        }
                        for wid, l in right_lanes.items()
                    ]
                    right_wid = min(right_vecs, key=lambda x: np.dot(x["vec"], vec))[
                        "wid"
                    ]  # The one with the smallest inner product is closest to the right
                    return right_wid
                if walk_type == "left":
                    vec = np.array(
                        [
                            np.cos(way_angle + np.pi / 2),
                            np.sin(way_angle + np.pi / 2),
                        ]
                    )
                    left_lanes = {wid: self.map_roads[wid]["lanes"][0] for wid in wids}
                    left_vecs = [
                        {
                            "wid": wid,
                            "vec": np.array(l.coords[0][:2]),
                        }
                        for wid, l in left_lanes.items()
                    ]
                    left_wid = max(left_vecs, key=lambda x: np.dot(x["vec"], vec))[
                        "wid"
                    ]  # The one with the largest inner product is closest to the left
                    return left_wid

        # Store the entry and exit wids in each direction and the source of the entry and exit lane (whether it is the way into the junc or the way out)
        walk_pairs = []
        for angles, in_ids, out_ids in walk_group:
            angle = np.mean(angles)
            # There are four situations depending on whether in_ids and out_ids exist
            if not in_ids and not out_ids:
                continue
            if in_ids and out_ids:
                in_wid = get_side_wid(wids=in_ids, way_angle=angle, walk_type="right")
                out_wid = get_side_wid(wids=out_ids, way_angle=angle, walk_type="left")
                walk_pairs.append(
                    {
                        "in_walk": (
                            in_wid,
                            "in_way",
                        ),
                        "out_walk": (
                            out_wid,
                            "out_way",
                        ),
                    }
                )
                for wid in in_ids:
                    if wid == in_wid:
                        if wid in has_main_group_wids:
                            self.no_left_walk.add(wid)
                    else:
                        self.no_right_walk.add(wid)
                        self.no_left_walk.add(wid)
                for wid in out_ids:
                    if wid == out_wid:
                        if wid in has_main_group_wids:
                            self.no_left_walk.add(wid)
                    else:
                        self.no_right_walk.add(wid)
                        self.no_left_walk.add(wid)
            if in_ids and not out_ids:
                in_wid = get_side_wid(wids=in_ids, way_angle=angle, walk_type="right")
                walk_pairs.append(
                    {
                        "in_walk": (
                            in_wid,
                            "in_way",
                        ),
                        "out_walk": (
                            in_wid,
                            "in_way",
                        ),
                    }
                )
                for wid in in_ids:
                    if wid == in_wid:
                        if wid in has_main_group_wids:
                            self.no_left_walk.add(wid)
                    else:
                        self.no_right_walk.add(wid)
                        self.no_left_walk.add(wid)
            if not in_ids and out_ids:
                out_wid = get_side_wid(wids=out_ids, way_angle=angle, walk_type="left")
                walk_pairs.append(
                    {
                        "in_walk": (
                            out_wid,
                            "out_way",
                        ),
                        "out_walk": (
                            out_wid,
                            "out_way",
                        ),
                    }
                )
                for wid in out_ids:
                    if wid == out_wid:
                        if wid in has_main_group_wids:
                            self.no_left_walk.add(wid)
                    else:
                        self.no_right_walk.add(wid)
                        self.no_left_walk.add(wid)
        return walk_pairs

    def _create_junction_for_1_n(self):
        """
        For an junction with 1 in and n out, create an junction
        Basic logic:
        1. Identify the roads in the out direction that are in the same direction as the in direction and regard them as main roads, while the rest are left and right ramps.
        2. All lanes connecting incoming and outgoing main roads
        3. The left and right ramps are only connected to the outermost lane on the corresponding side of the main road.
        """
        logging.info(f"Creating junction for 1 in and N out")
        keys = []

        def classify_main_auxiliary_wid(wids, way_angle, group_type):
            """
            Return the main road and auxiliary road in wids according to in_way_group/out_way_group
            """
            if group_type == "in_ways":
                vec = np.array(
                    [
                        np.cos(way_angle + np.pi / 2),
                        np.sin(way_angle + np.pi / 2),
                    ]
                )
                right_lanes = {wid: self.map_roads[wid]["lanes"][-1] for wid in wids}
                right_vecs = [
                    {
                        "wid": wid,
                        "vec": np.array(l.coords[-1][:2]),
                    }
                    for wid, l in right_lanes.items()
                ]
                # The smaller the inner product, the closer it is to the right
                sorted_wids = [
                    vec["wid"]
                    for vec in sorted(right_vecs, key=lambda x: -np.dot(x["vec"], vec))
                ]
            elif group_type == "out_ways":
                vec = np.array(
                    [
                        np.cos(way_angle + np.pi / 2),
                        np.sin(way_angle + np.pi / 2),
                    ]
                )
                left_lanes = {wid: self.map_roads[wid]["lanes"][0] for wid in wids}
                left_vecs = [
                    {
                        "wid": wid,
                        "vec": np.array(l.coords[0][:2]),
                    }
                    for wid, l in left_lanes.items()
                ]
                sorted_wids = [
                    vec["wid"]
                    for vec in sorted(left_vecs, key=lambda x: -np.dot(x["vec"], vec))
                ]
            else:
                raise ValueError(f"Invalid group_type:{group_type}")
            return (
                sorted_wids[0],
                sorted_wids[1:],
            )

        for in_count, out_count in self.junction_types.keys():
            if in_count == 1:
                keys.append((in_count, out_count))
        for key in keys:
            jid_index = 0
            list_jids = list(self.junction_types[key])
            bad_turn_config_jids = set()
            while jid_index < len(list_jids):
                pre_lane_uid = self.lane_uid
                jid = list_jids[jid_index]
                j = self.junctions[jid]
                j_uid = j["uid"]
                in_way_groups = j["properties"]["in_way_groups"]
                out_way_groups = j["properties"]["out_way_groups"]
                coord_xyzs = j["geometry"]["coordinates_xyz"]
                has_main_group_wids = set()
                # junction center
                x_center, y_center, z_center = np.mean(coord_xyzs, axis=0)
                # ================================================= ======
                in_angle, in_way_ids = in_way_groups[0]
                # Expand into the road
                self._expand_roads(
                    wids=in_way_ids, junc_type=key, junc_id=jid, way_type="main"
                )
                # Identify the main road: the road most similar to out_angle and in_angle
                # Identify the right ramp: the road where out_angle is on the right (clockwise) of in_angle
                # Identify the left ramp: the road where out_angle is on the left (counterclockwise) of in_angle
                # The criterion is: the angle between out_angle and in_angle
                out_way_groups.sort(key=lambda x: np.abs(delta_angle(in_angle, x[0])))
                out_main_group = None
                out_left_groups = []
                out_right_groups = []
                out_around_groups = []
                in_way_id2available_conn = {}
                # Find the road with the smallest absolute value of delta and within the threshold (plus or minus 30 degrees), which is the main road
                if (
                    np.abs(delta_angle(in_angle, out_way_groups[0][0]))
                    <= MAIN_WAY_ANGLE_THRESHOLD
                ):
                    out_main_group = out_way_groups[0]
                    out_angle = out_main_group[0]
                    out_way_ids = out_main_group[1]
                    # Expand main road
                    self._expand_roads(
                        wids=out_way_ids, junc_type=key, junc_id=jid, way_type="main"
                    )
                    out_way_groups = out_way_groups[1:]
                    # Have main group
                    for in_id in in_way_ids:
                        has_main_group_wids.add(in_id)
                    for out_id in out_way_ids:
                        has_main_group_wids.add(out_id)

                # The rest are allocated to Around left and right ramps
                for out_angle, out_way_ids in out_way_groups:
                    way_delta_angle = delta_angle(in_angle, out_angle)
                    abs_way_delta_angle = abs_delta_angle(in_angle, out_angle)
                    # Assign to Around turn
                    if (
                        abs_delta_angle(abs_way_delta_angle, np.pi)
                        < AROUND_WAY_ANGLE_THRESHOLD
                    ):
                        out_around_groups.append((out_angle, out_way_ids))
                        self._expand_roads(
                            wids=out_way_ids,
                            junc_type=key,
                            junc_id=jid,
                            way_type="around",
                        )
                    else:
                        if way_delta_angle < 0:
                            out_right_groups.append((out_angle, out_way_ids))
                            # Expansion ramp
                            self._expand_roads(
                                wids=out_way_ids,
                                junc_type=key,
                                junc_id=jid,
                                way_type="right",
                            )
                        else:
                            out_left_groups.append((out_angle, out_way_ids))
                            self._expand_roads(
                                wids=out_way_ids,
                                junc_type=key,
                                junc_id=jid,
                                way_type="left",
                            )
                ## Record the groups that in_way_id can connect to
                available_groups = []
                if out_main_group:
                    available_groups.append(f"Straight: {out_main_group[1]}")
                if out_left_groups:
                    left_way_ids = [g[1] for g in out_left_groups]
                    available_groups.append(f"Left: {left_way_ids}")
                if out_right_groups:
                    right_way_ids = [g[1] for g in out_right_groups]
                    available_groups.append(f"Right: {right_way_ids}")
                if out_around_groups:
                    around_way_ids = [g[1] for g in out_around_groups]
                    available_groups.append(f"Around: {around_way_ids}")
                for in_way_id in in_way_ids:
                    in_way_id2available_conn[in_way_id] = available_groups
                ##
                junc_lanes = []
                # Due to the difference between main roads and auxiliary roads, there are 4 possibilities for connected wids depending on the presence or absence of auxiliary roads.
                # 1.Only main road
                # 2.in only has the main road and out has the main road and the auxiliary road
                # 3. There are main roads and auxiliary roads in and only main roads out.
                # 4. There are main roads and auxiliary roads in and there are main roads and auxiliary roads out.
                for in_angle, in_way_ids in in_way_groups:
                    # Expand into road (lane)
                    self._expand_roads(
                        wids=in_way_ids, junc_type=key, junc_id=jid, way_type="main"
                    )
                    in_main_wid, in_auxiliary_wids = classify_main_auxiliary_wid(
                        in_way_ids, in_angle, "in_ways"
                    )
                    # Connect lanes
                    in_main_lanes = self.map_roads[in_main_wid]["lanes"]
                    in_auxiliary_lanes = [
                        l
                        for wid in in_auxiliary_wids
                        for l in self.map_roads[wid]["lanes"]
                    ]
                    # Lane allocation into main road
                    in_main_turn_config = self.map_roads[in_main_wid]["turn_config"]
                    # Lane allocation for entering service roads
                    if len(in_auxiliary_wids) == 1:
                        in_auxiliary_turn_config = self.map_roads[in_auxiliary_wids[0]][
                            "turn_config"
                        ]
                    else:
                        in_auxiliary_turn_config = []
                    # if jid in bad_turn_config_jids:
                    # in_main_turn_config = []
                    # in_auxiliary_turn_config = []
                    # Default turn annotation
                    # Go straight
                    default_in_straight_main_lanes = in_main_lanes
                    # Around
                    default_main_around_lane_num = DEFAULT_TURN_NUM["MAIN_AROUND"]
                    # Turn left
                    default_main_left_lane_num = (
                        DEFAULT_TURN_NUM["MAIN_LARGE_LEFT"]
                        if len(in_main_lanes) > LARGE_LANE_NUM_THRESHOLD
                        else DEFAULT_TURN_NUM["MAIN_SMALL_LEFT"]
                    )
                    # Turn right
                    default_main_right_lane_num = (
                        DEFAULT_TURN_NUM["MAIN_LARGE_RIGHT"]
                        if len(in_main_lanes) > LARGE_LANE_NUM_THRESHOLD
                        else DEFAULT_TURN_NUM["MAIN_SMALL_RIGHT"]
                    )
                    # Mark turn according to turn config
                    if in_main_turn_config and len(in_main_turn_config) == len(
                        in_main_lanes
                    ):
                        # Go straight
                        in_straight_main_lanes = [
                            l
                            for i, l in enumerate(in_main_lanes)
                            if "S" in in_main_turn_config[i]
                            or "s" in in_main_turn_config[i]
                        ]
                        # Around
                        main_around_lane_num = len(
                            [t for t in in_main_turn_config if "A" in t or "a" in t]
                        )
                        # Turn left
                        main_left_lane_num = len(
                            [t for t in in_main_turn_config if "L" in t or "l" in t]
                        )
                        # Turn right
                        main_right_lane_num = len(
                            [t for t in in_main_turn_config if "R" in t or "r" in t]
                        )

                        #

                        # if (
                        #     (
                        #         len(out_around_groups) == 0 and ((main_around_lane_num > 0))
                        #     )
                        #     or (len(out_left_groups) == 0 and ((main_left_lane_num > 0)))
                        #     or (len(out_right_groups) == 0 and ((main_right_lane_num > 0)))
                        # ):
                        # main road
                        if jid in bad_turn_config_jids:
                            # straight
                            in_straight_main_lanes = (
                                default_in_straight_main_lanes
                                if len(in_straight_main_lanes) == 0
                                else in_straight_main_lanes
                            )
                            # around
                            main_around_lane_num = (
                                default_main_around_lane_num
                                if main_around_lane_num == 0
                                else main_around_lane_num
                            )
                            # left
                            main_left_lane_num = (
                                default_main_left_lane_num
                                if main_left_lane_num == 0
                                else main_left_lane_num
                            )
                            # right
                            main_right_lane_num = (
                                default_main_right_lane_num
                                if main_right_lane_num == 0
                                else main_right_lane_num
                            )

                    else:
                        in_straight_main_lanes = default_in_straight_main_lanes
                        main_around_lane_num = default_main_around_lane_num
                        main_left_lane_num = default_main_left_lane_num
                        main_right_lane_num = default_main_right_lane_num
                    # Default turn annotation
                    # Go straight
                    default_in_straight_auxiliary_lanes = in_auxiliary_lanes
                    # Around
                    default_auxiliary_around_lane_num = DEFAULT_TURN_NUM[
                        "AUXILIARY_AROUND"
                    ]
                    # Turn left
                    default_auxiliary_left_lane_num = DEFAULT_TURN_NUM[
                        "AUXILIARY_SMALL_LEFT"
                    ]
                    # Turn right
                    default_auxiliary_right_lane_num = DEFAULT_TURN_NUM[
                        "AUXILIARY_SMALL_RIGHT"
                    ]
                    # Mark turn according to turn config
                    if in_auxiliary_turn_config and len(
                        in_auxiliary_turn_config
                    ) == len(in_auxiliary_lanes):
                        # Go straight
                        in_straight_auxiliary_lanes = [
                            l
                            for i, l in enumerate(in_auxiliary_lanes)
                            if "S" in in_auxiliary_turn_config[i]
                            or "s" in in_auxiliary_turn_config[i]
                        ]
                        # Around
                        auxiliary_around_lane_num = len(
                            [
                                t
                                for t in in_auxiliary_turn_config
                                if "A" in t or "a" in t
                            ]
                        )
                        # Turn left
                        auxiliary_left_lane_num = len(
                            [
                                t
                                for t in in_auxiliary_turn_config
                                if "L" in t or "l" in t
                            ]
                        )
                        # Turn right
                        auxiliary_right_lane_num = len(
                            [
                                t
                                for t in in_auxiliary_turn_config
                                if "R" in t or "r" in t
                            ]
                        )
                        # Determine whether turn_config can be used to connect to the lane. If the default configuration cannot be used,
                        # if (
                        #      (
                        #         len(out_around_groups) == 0
                        #         and ((auxiliary_around_lane_num > 0))
                        #     )
                        #     or (
                        #         len(out_left_groups) == 0
                        #         and ((auxiliary_left_lane_num > 0))
                        #     )
                        #     or (
                        #         len(out_right_groups) == 0
                        #         and ((auxiliary_right_lane_num > 0))
                        #     )
                        # ):
                        # auxiliary road
                        if jid in bad_turn_config_jids:
                            # straight
                            in_straight_auxiliary_lanes = (
                                default_in_straight_auxiliary_lanes
                                if len(in_straight_auxiliary_lanes) == 0
                                else in_straight_auxiliary_lanes
                            )
                            # around
                            auxiliary_around_lane_num = (
                                default_auxiliary_around_lane_num
                                if auxiliary_around_lane_num == 0
                                else auxiliary_around_lane_num
                            )
                            # left
                            auxiliary_left_lane_num = (
                                default_auxiliary_left_lane_num
                                if auxiliary_left_lane_num == 0
                                else auxiliary_left_lane_num
                            )
                            # right
                            auxiliary_right_lane_num = (
                                default_auxiliary_right_lane_num
                                if auxiliary_right_lane_num == 0
                                else auxiliary_right_lane_num
                            )
                    else:
                        in_straight_auxiliary_lanes = (
                            default_in_straight_auxiliary_lanes
                        )
                        auxiliary_around_lane_num = default_auxiliary_around_lane_num
                        auxiliary_left_lane_num = default_auxiliary_left_lane_num
                        auxiliary_right_lane_num = default_auxiliary_right_lane_num
                    # All lanes connecting incoming and outgoing main roads
                    if out_main_group is not None:
                        out_angle, out_way_ids = out_main_group
                        out_main_wid, out_auxiliary_wids = classify_main_auxiliary_wid(
                            out_way_ids, out_angle, "out_ways"
                        )
                        out_main_lanes = self.map_roads[out_main_wid]["lanes"]
                        out_auxiliary_lanes = [
                            l
                            for wid in out_auxiliary_wids
                            for l in self.map_roads[wid]["lanes"]
                        ]
                        # main road to main road
                        if len(in_straight_main_lanes) > 0 and len(out_main_lanes) > 0:
                            junc_lanes += self._connect_lane_group(
                                in_lanes=in_straight_main_lanes,
                                out_lanes=out_main_lanes,
                                lane_turn=mapv2.LANE_TURN_STRAIGHT,
                                lane_type=mapv2.LANE_TYPE_DRIVING,
                                junc_id=j_uid,
                            )
                        # auxiliary road to auxiliary road
                        if (
                            len(in_straight_auxiliary_lanes) > 0
                            and len(out_auxiliary_lanes) > 0
                        ):
                            junc_lanes += self._connect_lane_group(
                                in_lanes=in_straight_auxiliary_lanes,
                                out_lanes=out_auxiliary_lanes,
                                lane_turn=mapv2.LANE_TURN_STRAIGHT,
                                lane_type=mapv2.LANE_TYPE_DRIVING,
                                junc_id=j_uid,
                            )
                        # When there is no in auxiliary road, connect the in main road to the out auxiliary road
                        if (
                            not len(in_straight_auxiliary_lanes) > 0
                            and len(out_auxiliary_lanes) > 0
                            and len(in_straight_main_lanes) > 0
                        ):
                            junc_lanes += self._connect_lane_group(
                                in_lanes=in_straight_main_lanes,
                                out_lanes=out_auxiliary_lanes,
                                lane_turn=mapv2.LANE_TURN_STRAIGHT,
                                lane_type=mapv2.LANE_TYPE_DRIVING,
                                junc_id=j_uid,
                            )
                        # When there is no out auxiliary road, connect the in auxiliary road to the main road
                        if (
                            len(in_straight_auxiliary_lanes) > 0
                            and not len(out_auxiliary_lanes) > 0
                            and len(out_main_lanes) > 0
                        ):
                            junc_lanes += self._connect_lane_group(
                                in_lanes=in_straight_auxiliary_lanes,
                                out_lanes=out_main_lanes,
                                lane_turn=mapv2.LANE_TURN_STRAIGHT,
                                lane_type=mapv2.LANE_TYPE_DRIVING,
                                junc_id=j_uid,
                            )

                    # connect around lanes
                    if len(out_around_groups) > 0:
                        for out_angle, out_way_ids in out_around_groups:
                            (
                                out_main_wid,
                                out_auxiliary_wids,
                            ) = classify_main_auxiliary_wid(
                                out_way_ids, out_angle, "out_ways"
                            )
                            out_main_lanes = self.map_roads[out_main_wid]["lanes"]
                            out_auxiliary_lanes = [
                                l
                                for wid in out_auxiliary_wids
                                for l in self.map_roads[wid]["lanes"]
                            ]
                            # ATTENTION: There are special penalties for around turns at junctions of type (1,1) in out all connections
                            if key == (1, 1):
                                main_around_lane_num = max(
                                    len(in_main_lanes),
                                    len(out_main_lanes),
                                    main_around_lane_num,
                                )
                                auxiliary_around_lane_num = max(
                                    len(in_auxiliary_lanes),
                                    len(out_auxiliary_lanes),
                                    auxiliary_around_lane_num,
                                )
                            # main road to main road
                            if (
                                len(in_main_lanes[:main_around_lane_num]) > 0
                                and len(out_main_lanes[:main_around_lane_num]) > 0
                            ):
                                junc_lanes += self._connect_lane_group(
                                    in_lanes=in_main_lanes[:main_around_lane_num],
                                    out_lanes=out_main_lanes[:main_around_lane_num],
                                    lane_turn=mapv2.LANE_TURN_AROUND,
                                    lane_type=mapv2.LANE_TYPE_DRIVING,
                                    junc_id=j_uid,
                                )
                            # auxiliary road to auxiliary road
                            if (
                                len(in_auxiliary_lanes[:auxiliary_around_lane_num]) > 0
                                and len(out_auxiliary_lanes[:auxiliary_around_lane_num])
                                > 0
                            ):
                                junc_lanes += self._connect_lane_group(
                                    in_lanes=in_auxiliary_lanes[
                                        :auxiliary_around_lane_num
                                    ],
                                    out_lanes=out_auxiliary_lanes[
                                        :auxiliary_around_lane_num
                                    ],
                                    lane_turn=mapv2.LANE_TURN_AROUND,
                                    lane_type=mapv2.LANE_TYPE_DRIVING,
                                    junc_id=j_uid,
                                )
                            # When there is no in auxiliary road, connect the in main road to the out auxiliary road
                            if (
                                not len(in_auxiliary_lanes[:main_around_lane_num]) > 0
                                and len(out_auxiliary_lanes[:main_around_lane_num]) > 0
                                and len(in_main_lanes[:main_around_lane_num]) > 0
                            ):
                                junc_lanes += self._connect_lane_group(
                                    in_lanes=in_main_lanes[:main_around_lane_num],
                                    out_lanes=out_auxiliary_lanes[
                                        :main_around_lane_num
                                    ],
                                    lane_turn=mapv2.LANE_TURN_AROUND,
                                    lane_type=mapv2.LANE_TYPE_DRIVING,
                                    junc_id=j_uid,
                                )
                            # When there is no out auxiliary road, connect the in auxiliary road to the main road
                            if (
                                len(in_auxiliary_lanes[:auxiliary_around_lane_num]) > 0
                                and not len(
                                    out_auxiliary_lanes[:auxiliary_around_lane_num]
                                )
                                > 0
                                and len(out_main_lanes[:auxiliary_around_lane_num]) > 0
                            ):
                                junc_lanes += self._connect_lane_group(
                                    in_lanes=in_auxiliary_lanes[
                                        :auxiliary_around_lane_num
                                    ],
                                    out_lanes=out_main_lanes[
                                        :auxiliary_around_lane_num
                                    ],
                                    lane_turn=mapv2.LANE_TURN_AROUND,
                                    lane_type=mapv2.LANE_TYPE_DRIVING,
                                    junc_id=j_uid,
                                )

                    # Connect left ramp
                    if len(out_left_groups) > 0:
                        for out_angle, out_way_ids in out_left_groups:
                            (
                                out_main_wid,
                                out_auxiliary_wids,
                            ) = classify_main_auxiliary_wid(
                                out_way_ids, out_angle, "out_ways"
                            )
                            out_main_lanes = self.map_roads[out_main_wid]["lanes"]
                            out_auxiliary_lanes = [
                                l
                                for wid in out_auxiliary_wids
                                for l in self.map_roads[wid]["lanes"]
                            ]
                            # main road to main road
                            if (
                                len(in_main_lanes[:main_left_lane_num]) > 0
                                and len(out_main_lanes) > 0
                            ):
                                junc_lanes += self._connect_lane_group(
                                    in_lanes=in_main_lanes[:main_left_lane_num],
                                    out_lanes=out_main_lanes,
                                    lane_turn=mapv2.LANE_TURN_LEFT,
                                    lane_type=mapv2.LANE_TYPE_DRIVING,
                                    junc_id=j_uid,
                                )
                            # auxiliary road to auxiliary road
                            if (
                                len(in_auxiliary_lanes[:auxiliary_left_lane_num]) > 0
                                and len(out_auxiliary_lanes) > 0
                            ):
                                junc_lanes += self._connect_lane_group(
                                    in_lanes=in_auxiliary_lanes[
                                        :auxiliary_left_lane_num
                                    ],
                                    out_lanes=out_auxiliary_lanes,
                                    lane_turn=mapv2.LANE_TURN_LEFT,
                                    lane_type=mapv2.LANE_TYPE_DRIVING,
                                    junc_id=j_uid,
                                )
                            # When there is no in auxiliary road, connect the in main road to the out auxiliary road
                            if (
                                not len(in_auxiliary_lanes[:main_left_lane_num]) > 0
                                and len(out_auxiliary_lanes) > 0
                                and len(in_main_lanes[:main_left_lane_num]) > 0
                            ):
                                junc_lanes += self._connect_lane_group(
                                    in_lanes=in_main_lanes[:main_left_lane_num],
                                    out_lanes=out_auxiliary_lanes,
                                    lane_turn=mapv2.LANE_TURN_LEFT,
                                    lane_type=mapv2.LANE_TYPE_DRIVING,
                                    junc_id=j_uid,
                                )
                            # When there is no out auxiliary road, connect the in auxiliary road to the main road
                            if (
                                len(in_auxiliary_lanes[:auxiliary_left_lane_num]) > 0
                                and not len(out_auxiliary_lanes) > 0
                                and len(out_main_lanes) > 0
                            ):
                                junc_lanes += self._connect_lane_group(
                                    in_lanes=in_auxiliary_lanes[
                                        :auxiliary_left_lane_num
                                    ],
                                    out_lanes=out_main_lanes,
                                    lane_turn=mapv2.LANE_TURN_LEFT,
                                    lane_type=mapv2.LANE_TYPE_DRIVING,
                                    junc_id=j_uid,
                                )
                    # Connect right ramp
                    if len(out_right_groups) > 0:
                        for out_angle, out_way_ids in out_right_groups:
                            (
                                out_main_wid,
                                out_auxiliary_wids,
                            ) = classify_main_auxiliary_wid(
                                out_way_ids, out_angle, "out_ways"
                            )
                            out_main_lanes = self.map_roads[out_main_wid]["lanes"]
                            out_auxiliary_lanes = [
                                l
                                for wid in out_auxiliary_wids
                                for l in self.map_roads[wid]["lanes"]
                            ]
                            # main road to main road
                            if (
                                len(in_main_lanes[-main_right_lane_num:]) > 0
                                and len(out_main_lanes) > 0
                            ):
                                junc_lanes += self._connect_lane_group(
                                    in_lanes=in_main_lanes[-main_right_lane_num:],
                                    out_lanes=out_main_lanes,
                                    lane_turn=mapv2.LANE_TURN_RIGHT,
                                    lane_type=mapv2.LANE_TYPE_DRIVING,
                                    junc_id=j_uid,
                                )
                            # auxiliary road to auxiliary road
                            if (
                                len(in_auxiliary_lanes[-auxiliary_right_lane_num:]) > 0
                                and len(out_auxiliary_lanes) > 0
                            ):
                                junc_lanes += self._connect_lane_group(
                                    in_lanes=in_auxiliary_lanes[
                                        -auxiliary_right_lane_num:
                                    ],
                                    out_lanes=out_auxiliary_lanes,
                                    lane_turn=mapv2.LANE_TURN_RIGHT,
                                    lane_type=mapv2.LANE_TYPE_DRIVING,
                                    junc_id=j_uid,
                                )
                            # When there is no in auxiliary road, connect the in main road to the out auxiliary road
                            if (
                                not len(in_auxiliary_lanes[-main_right_lane_num:]) > 0
                                and len(out_auxiliary_lanes) > 0
                                and len(in_main_lanes[-main_right_lane_num:]) > 0
                            ):
                                junc_lanes += self._connect_lane_group(
                                    in_lanes=in_main_lanes[-main_right_lane_num:],
                                    out_lanes=out_auxiliary_lanes,
                                    lane_turn=mapv2.LANE_TURN_RIGHT,
                                    lane_type=mapv2.LANE_TYPE_DRIVING,
                                    junc_id=j_uid,
                                )
                            # When there is no out auxiliary road, connect the in auxiliary road to the main road
                            if (
                                len(in_auxiliary_lanes[-auxiliary_right_lane_num:]) > 0
                                and not len(out_auxiliary_lanes) > 0
                                and len(out_main_lanes) > 0
                            ):
                                junc_lanes += self._connect_lane_group(
                                    in_lanes=in_auxiliary_lanes[
                                        -auxiliary_right_lane_num:
                                    ],
                                    out_lanes=out_main_lanes,
                                    lane_turn=mapv2.LANE_TURN_RIGHT,
                                    lane_type=mapv2.LANE_TYPE_DRIVING,
                                    junc_id=j_uid,
                                )
                # Check whether every road in the junction is connected
                valid_turn_config = True
                for _, in_way_ids in in_way_groups:
                    if valid_turn_config:
                        for in_way_id in in_way_ids:
                            in_map_road = self.map_roads[in_way_id]
                            if all(
                                len(self.lane2data[in_lane]["out"]) == 0
                                for in_lane in in_map_road["lanes"]
                            ):
                                logging.warning(
                                    f"No available out ways for {in_way_id} to connect due to invalid turn_config {in_map_road['turn_config']}, available turn groups: {in_way_id2available_conn[in_way_id]}"
                                )
                                valid_turn_config = False
                for _, out_way_ids in out_way_groups:
                    if valid_turn_config:
                        for out_way_id in out_way_ids:
                            out_map_road = self.map_roads[out_way_id]
                            if all(
                                len(self.lane2data[out_lane]["in"]) == 0
                                for out_lane in out_map_road["lanes"]
                            ):
                                logging.warning(
                                    f"No in ways connected to {out_way_id} due to invalid turn_config of in_ways, available turn groups for all in_ways: {in_way_id2available_conn}"
                                )
                                valid_turn_config = False
                if not valid_turn_config and jid not in bad_turn_config_jids:
                    logging.warning(f"Invalid turn_config at {jid}")
                    if self.strict_mode:
                        raise Exception("Encounter invalid road turn config.")
                    bad_turn_config_jids.add(jid)
                    # Delete all connection relationships between road lane and junc lane
                    # Delete all turn_config and take the default value
                    # Clear all roads expanded under this junction
                    for lane_id in range(pre_lane_uid, self.lane_uid):
                        self._delete_lane(lane_id, delete_road=True)
                    self.lane_uid = pre_lane_uid
                    continue

                jid_index += 1
                if out_main_group:
                    # restore out way group
                    out_way_groups = [out_main_group] + out_way_groups
                walk_pairs = self._create_junction_walk_pairs(
                    in_way_groups,
                    out_way_groups,
                    has_main_group_wids,
                    (x_center, y_center),
                )

                self.map_junctions[jid] = {
                    "lanes": junc_lanes,
                    "uid": j_uid,
                    "center": {
                        "x": x_center,
                        "y": y_center,
                        "z": z_center,
                    },
                    "walk_pairs": walk_pairs,
                }

    def _create_junction_for_n_n(self):
        """
        Dealing with junctions
        """
        logging.info(f"Creating junction for N in and N out")
        for in_count, out_count in self.junction_types.keys():
            self._junction_keys.append((in_count, out_count))
        keys_n_n = [
            (in_count, out_count)
            for in_count, out_count in sorted(
                self._junction_keys, key=lambda x: -sum(x)
            )
            if not in_count == 1
        ]

        def classify_main_auxiliary_wid(wids, way_angle, group_type):
            """
            Return the main road and auxiliary road in wids according to in_way_group/out_way_group
            """
            if group_type == "in_ways":
                vec = np.array(
                    [
                        np.cos(way_angle + np.pi / 2),
                        np.sin(way_angle + np.pi / 2),
                    ]
                )
                right_lanes = {wid: self.map_roads[wid]["lanes"][-1] for wid in wids}
                right_vecs = [
                    {
                        "wid": wid,
                        "vec": np.array(l.coords[-1][:2]),
                    }
                    for wid, l in right_lanes.items()
                ]
                # The smaller the inner product, the closer it is to the right
                sorted_wids = [
                    vec["wid"]
                    for vec in sorted(right_vecs, key=lambda x: -np.dot(x["vec"], vec))
                ]
            elif group_type == "out_ways":
                vec = np.array(
                    [
                        np.cos(way_angle + np.pi / 2),
                        np.sin(way_angle + np.pi / 2),
                    ]
                )
                left_lanes = {wid: self.map_roads[wid]["lanes"][0] for wid in wids}
                left_vecs = [
                    {
                        "wid": wid,
                        "vec": np.array(l.coords[0][:2]),
                    }
                    for wid, l in left_lanes.items()
                ]
                sorted_wids = [
                    vec["wid"]
                    for vec in sorted(left_vecs, key=lambda x: -np.dot(x["vec"], vec))
                ]
            else:
                raise ValueError(f"Invalid group_type:{group_type}")
            return (
                sorted_wids[0],
                sorted_wids[1:],
            )

        for key in keys_n_n:
            jid_index = 0
            list_jids = list(self.junction_types[key])
            bad_turn_config_jids = set()
            while jid_index < len(list_jids):
                pre_lane_uid = self.lane_uid
                jid = list_jids[jid_index]
                j = self.junctions[jid]
                j_uid = j["uid"]
                in_way_groups = j["properties"]["in_way_groups"]
                out_way_groups = j["properties"]["out_way_groups"]
                coord_xyzs = j["geometry"]["coordinates_xyz"]
                # junction center
                x_center, y_center, z_center = np.mean(coord_xyzs, axis=0)
                # =======================================================
                # For each in_way_group
                # 1. Identify the main road
                # 2. Identify left turn and right turn
                # 3. All lanes connecting incoming and outgoing main roads
                # 4. The left and right ramps are only connected to the outermost lane on the corresponding side of the main road.
                # =======================================================
                junc_lanes = []
                out_main_group = None
                has_main_group_wids = set()
                in_way_id2available_conn = {}
                for in_angle, in_way_ids in in_way_groups:
                    # Expand into the road
                    self._expand_roads(
                        wids=in_way_ids, junc_type=key, junc_id=jid, way_type="main"
                    )
                    in_main_wid, in_auxiliary_wids = classify_main_auxiliary_wid(
                        in_way_ids, in_angle, "in_ways"
                    )
                    # Connect lanes
                    in_main_lanes = self.map_roads[in_main_wid]["lanes"]
                    in_auxiliary_lanes = [
                        l
                        for wid in in_auxiliary_wids
                        for l in self.map_roads[wid]["lanes"]
                    ]
                    # Lane allocation into main road
                    in_main_turn_config = self.map_roads[in_main_wid]["turn_config"]
                    # Lane allocation for entering service roads
                    if len(in_auxiliary_wids) == 1:
                        in_auxiliary_turn_config = self.map_roads[in_auxiliary_wids[0]][
                            "turn_config"
                        ]
                    else:
                        in_auxiliary_turn_config = []
                    # if jid in bad_turn_config_jids:
                    # in_main_turn_config = []
                    # in_auxiliary_turn_config = []
                    # Identify the main road: the road most similar to out_angle and in_angle
                    # Identify the right ramp: the road where out_angle is on the right (clockwise) of in_angle
                    # Identify the left ramp: the road where out_angle is on the left (counterclockwise) of in_angle
                    # The criterion is: the angle between out_angle and in_angle
                    out_way_groups.sort(key=lambda x: abs_delta_angle(in_angle, x[0]))
                    abs_way_delta_angle = abs_delta_angle(
                        in_angle, out_way_groups[0][0]
                    )
                    if abs_way_delta_angle < MAIN_WAY_ANGLE_THRESHOLD:
                        out_main_group = out_way_groups[0]
                        out_angle = out_main_group[0]
                        out_way_ids = out_main_group[1]
                        # Expand main road
                        self._expand_roads(
                            wids=out_way_ids,
                            junc_type=key,
                            junc_id=jid,
                            way_type="main",
                        )
                        main_start_index = 1
                        # If there is a main group, there is no left sidewalk in this group.
                        for in_id in in_way_ids:
                            has_main_group_wids.add(in_id)
                        for out_id in out_way_ids:
                            has_main_group_wids.add(out_id)
                    else:
                        out_main_group = None
                        main_start_index = 0
                    out_left_groups = []
                    out_right_groups = []
                    out_around_groups = []
                    # The rest are allocated to Arounds and left and right ramps
                    for out_angle, out_way_ids in out_way_groups[main_start_index:]:
                        way_delta_angle = delta_angle(in_angle, out_angle)
                        abs_way_delta_angle = abs_delta_angle(in_angle, out_angle)
                        # Assign to Around
                        if (
                            abs_delta_angle(abs_way_delta_angle, np.pi)
                            < AROUND_WAY_ANGLE_THRESHOLD
                        ):
                            out_around_groups.append((out_angle, out_way_ids))
                            self._expand_roads(
                                wids=out_way_ids,
                                junc_type=key,
                                junc_id=jid,
                                way_type="around",
                            )
                        else:
                            if way_delta_angle < 0:
                                out_right_groups.append((out_angle, out_way_ids))
                                # Extended ramp
                                self._expand_roads(
                                    wids=out_way_ids,
                                    junc_type=key,
                                    junc_id=jid,
                                    way_type="right",
                                )
                            else:
                                out_left_groups.append((out_angle, out_way_ids))
                                self._expand_roads(
                                    wids=out_way_ids,
                                    junc_type=key,
                                    junc_id=jid,
                                    way_type="left",
                                )
                    ## Record the groups that in_way_id can connect to
                    available_groups = []
                    if out_main_group:
                        available_groups.append(f"Straight: {out_main_group[1]}")
                    if out_left_groups:
                        left_way_ids = [g[1] for g in out_left_groups]
                        available_groups.append(f"Left: {left_way_ids}")
                    if out_right_groups:
                        right_way_ids = [g[1] for g in out_right_groups]
                        available_groups.append(f"Right: {right_way_ids}")
                    if out_around_groups:
                        around_way_ids = [g[1] for g in out_around_groups]
                        available_groups.append(f"Around: {around_way_ids}")
                    for in_way_id in in_way_ids:
                        in_way_id2available_conn[in_way_id] = available_groups
                    ##
                    # No side roads
                    if not len(in_auxiliary_lanes) > 0:
                        # total number
                        # Default turn annotation
                        # Go straight
                        default_main_count = (
                            len(in_main_lanes) if out_main_group is not None else 0
                        )
                        # Around
                        default_around_count = 0
                        # Turn left
                        default_left_count = 0
                        # Turn right
                        default_right_count = 0
                        if len(out_around_groups) > 0:
                            default_around_count += DEFAULT_TURN_NUM["MAIN_AROUND"]
                        if len(out_right_groups) > 0:
                            default_right_count += DEFAULT_TURN_NUM["MAIN_SMALL_RIGHT"]
                            default_main_count -= default_right_count
                        if len(out_left_groups) > 0:
                            default_left_count += (
                                DEFAULT_TURN_NUM["MAIN_LARGE_LEFT"]
                                if default_main_count >= LARGE_LANE_NUM_THRESHOLD
                                else DEFAULT_TURN_NUM["MAIN_SMALL_LEFT"]
                            )
                            # There is a shared road here that turns left and goes straight
                            if default_main_count - default_left_count >= 1:
                                default_main_count -= default_left_count
                            else:
                                default_main_count -= default_left_count - 1
                            if (
                                len(in_main_lanes) <= SMALL_LANE_NUM_THRESHOLD
                                and out_main_group
                            ):
                                default_main_count = len(in_main_lanes)
                        default_main_out_start, default_main_out_end = (
                            max(default_left_count, 0),
                            max(default_left_count, 0) + default_main_count,
                        )
                        if len(in_main_lanes) <= SMALL_LANE_NUM_THRESHOLD:
                            default_main_out_start, default_main_out_end = 0, len(
                                in_main_lanes
                            )
                        # Get the road to be connected based on turn_config
                        if in_main_turn_config and len(in_main_turn_config) == len(
                            in_main_lanes
                        ):
                            # Go straight
                            main_indexes = [
                                i
                                for i, t in enumerate(in_main_turn_config)
                                if "S" in t or "s" in t
                            ]
                            main_count = len(main_indexes)
                            main_out_start = (
                                main_indexes[0] if len(main_indexes) > 0 else 0
                            )
                            main_out_end = (
                                main_indexes[-1] + 1
                                if len(main_indexes) > 0
                                else len(in_main_lanes)
                            )
                            # Around
                            around_count = len(
                                [t for t in in_main_turn_config if "A" in t or "a" in t]
                            )
                            # Turn left
                            left_count = len(
                                [t for t in in_main_turn_config if "L" in t or "l" in t]
                            )
                            # Turn right
                            right_count = len(
                                [t for t in in_main_turn_config if "R" in t or "r" in t]
                            )
                            # Determine whether turn_config can be used to connect to lane. If the default configuration cannot be used,
                            # if ((len(out_around_groups) == 0 and (around_count > 0))
                            #     or (len(out_left_groups) == 0 and (left_count > 0))
                            #     or (len(out_right_groups) == 0 and (right_count > 0))
                            # ):
                            if jid in bad_turn_config_jids:
                                around_count = (
                                    default_around_count
                                    if around_count == 0
                                    else around_count
                                )
                                right_count = (
                                    default_right_count
                                    if right_count == 0
                                    else right_count
                                )
                                left_count = (
                                    default_left_count
                                    if left_count == 0
                                    else left_count
                                )
                                main_count = (
                                    default_main_count
                                    if main_count == 0
                                    else main_count
                                )
                        else:
                            around_count = default_around_count
                            right_count = default_right_count
                            left_count = default_left_count
                            main_count = default_main_count
                            main_out_start = default_main_out_start
                            main_out_end = default_main_out_end
                        # connect lanes
                        if around_count > 0:
                            for out_angle, out_way_ids in out_around_groups:
                                for wid in out_way_ids:
                                    out_road = self.map_roads[wid]
                                    junc_lanes += self._connect_lane_group(
                                        in_lanes=in_main_lanes[:around_count],
                                        out_lanes=[out_road["lanes"][0]],
                                        lane_turn=mapv2.LANE_TURN_AROUND,
                                        lane_type=mapv2.LANE_TYPE_DRIVING,
                                        junc_id=j_uid,
                                    )
                        if right_count > 0:
                            for out_angle, out_way_ids in out_right_groups:
                                for wid in out_way_ids:
                                    out_road = self.map_roads[wid]
                                    junc_lanes += self._connect_lane_group(
                                        in_lanes=in_main_lanes[-right_count:],
                                        out_lanes=[out_road["lanes"][-1]],
                                        lane_turn=mapv2.LANE_TURN_RIGHT,
                                        lane_type=mapv2.LANE_TYPE_DRIVING,
                                        junc_id=j_uid,
                                    )
                        if left_count > 0:
                            for out_angle, out_way_ids in out_left_groups:
                                for wid in out_way_ids:
                                    out_road = self.map_roads[wid]
                                    junc_lanes += self._connect_lane_group(
                                        in_lanes=in_main_lanes[:left_count],
                                        out_lanes=out_road["lanes"][:left_count],
                                        lane_turn=mapv2.LANE_TURN_LEFT,
                                        lane_type=mapv2.LANE_TYPE_DRIVING,
                                        junc_id=j_uid,
                                    )
                        if main_count > 0 and out_main_group:
                            out_road = self.map_roads[out_main_group[1][0]]
                            junc_lanes += self._connect_lane_group(
                                in_lanes=in_main_lanes[main_out_start:main_out_end],
                                out_lanes=out_road["lanes"],
                                lane_turn=mapv2.LANE_TURN_STRAIGHT,
                                lane_type=mapv2.LANE_TYPE_DRIVING,
                                junc_id=j_uid,
                            )
                    else:  # There is a auxiliary road
                        # main road
                        # Default turn annotation
                        # Go straight
                        default_main_count = (
                            len(in_main_lanes) if out_main_group is not None else 0
                        )
                        # Around
                        default_around_count = 0
                        # Turn left
                        default_left_count = 0
                        # Turn right
                        default_right_count = 0
                        if len(out_around_groups) > 0:
                            default_around_count += DEFAULT_TURN_NUM["MAIN_AROUND"]
                        if len(out_right_groups) > 0:
                            default_right_count += DEFAULT_TURN_NUM["MAIN_SMALL_RIGHT"]
                            default_main_count -= default_right_count
                        if len(out_left_groups) > 0:
                            default_left_count += (
                                DEFAULT_TURN_NUM["MAIN_LARGE_LEFT"]
                                if default_main_count >= LARGE_LANE_NUM_THRESHOLD
                                else DEFAULT_TURN_NUM["MAIN_SMALL_LEFT"]
                            )
                            # There is a shared road that turns left and goes straight.
                            if default_main_count - default_left_count >= 1:
                                default_main_count -= default_left_count
                            else:
                                default_main_count -= default_left_count - 1
                            if (
                                len(in_main_lanes) <= SMALL_LANE_NUM_THRESHOLD
                                and out_main_group
                            ):
                                default_main_count = len(in_main_lanes)
                        default_main_out_start, default_main_out_end = (
                            max(default_left_count, 0),
                            max(default_left_count, 0) + default_main_count,
                        )
                        if len(in_main_lanes) <= SMALL_LANE_NUM_THRESHOLD:
                            default_main_out_start, default_main_out_end = 0, len(
                                in_main_lanes
                            )
                        # main road
                        if in_main_turn_config and len(in_main_turn_config) == len(
                            in_main_lanes
                        ):
                            # Go straight
                            main_indexes = [
                                i
                                for i, t in enumerate(in_main_turn_config)
                                if "S" in t or "s" in t
                            ]
                            main_count = len(main_indexes)
                            main_out_start = (
                                main_indexes[0] if len(main_indexes) > 0 else 0
                            )
                            main_out_end = (
                                main_indexes[-1] + 1
                                if len(main_indexes) > 0
                                else len(in_main_lanes)
                            )
                            # Around
                            around_count = len(
                                [t for t in in_main_turn_config if "A" in t or "a" in t]
                            )
                            # Turn left
                            left_count = len(
                                [t for t in in_main_turn_config if "L" in t or "l" in t]
                            )
                            # Turn right
                            right_count = len(
                                [t for t in in_main_turn_config if "R" in t or "r" in t]
                            )

                            # # Determine whether turn_config can be used to connect to lane. If the default configuration cannot be used,
                            # if (
                            #     (len(out_around_groups) == 0 and (around_count > 0))
                            #     or (len(out_left_groups) == 0 and left_count > 0)
                            #     or (len(out_right_groups) == 0 and right_count > 0)
                            # ):

                            if jid in bad_turn_config_jids:
                                around_count = (
                                    default_around_count
                                    if around_count == 0
                                    else around_count
                                )
                                right_count = (
                                    default_right_count
                                    if right_count == 0
                                    else right_count
                                )
                                left_count = (
                                    default_left_count
                                    if left_count == 0
                                    else left_count
                                )
                                main_count = (
                                    default_main_count
                                    if main_count == 0
                                    else main_count
                                )
                        else:
                            around_count = default_around_count
                            right_count = default_right_count
                            left_count = default_left_count
                            main_count = default_main_count
                            main_out_start = default_main_out_start
                            main_out_end = default_main_out_end

                        # secondary road
                        # Default turn annotation
                        # Go straight
                        default_auxiliary_main_count = (
                            len(in_auxiliary_lanes) if out_main_group is not None else 0
                        )
                        # Around
                        default_auxiliary_around_count = 0
                        # Turn left
                        default_auxiliary_left_count = 0
                        # Turn right
                        default_auxiliary_right_count = 0
                        if len(out_around_groups) > 0:
                            default_auxiliary_around_count += DEFAULT_TURN_NUM[
                                "AUXILIARY_AROUND"
                            ]
                        if len(out_right_groups) > 0:
                            default_auxiliary_right_count += DEFAULT_TURN_NUM[
                                "AUXILIARY_SMALL_RIGHT"
                            ]
                            default_auxiliary_main_count -= (
                                default_auxiliary_right_count
                            )
                        if len(out_left_groups) > 0:
                            default_auxiliary_left_count += (
                                DEFAULT_TURN_NUM["AUXILIARY_LARGE_LEFT"]
                                if default_auxiliary_main_count
                                >= LARGE_LANE_NUM_THRESHOLD
                                else DEFAULT_TURN_NUM["AUXILIARY_SMALL_LEFT"]
                            )
                            # There is a shared road that turns left and goes straight.
                            if (
                                default_auxiliary_main_count
                                - default_auxiliary_left_count
                                >= 1
                            ):
                                default_auxiliary_main_count -= (
                                    default_auxiliary_left_count
                                )
                            else:
                                default_auxiliary_main_count -= (
                                    default_auxiliary_left_count - 1
                                )
                            if (
                                len(in_auxiliary_lanes) <= SMALL_LANE_NUM_THRESHOLD
                                and out_main_group
                            ):
                                default_auxiliary_main_count = len(in_auxiliary_lanes)
                        (
                            default_auxiliary_main_out_start,
                            default_auxiliary_main_out_end,
                        ) = (
                            max(left_count, 0),
                            max(left_count, 0) + default_auxiliary_main_count,
                        )
                        if len(in_auxiliary_lanes) <= SMALL_LANE_NUM_THRESHOLD:
                            (
                                default_auxiliary_main_out_end,
                                default_auxiliary_main_out_end,
                            ) = 0, len(in_auxiliary_lanes)
                        if in_auxiliary_turn_config and len(
                            in_auxiliary_turn_config
                        ) == len(in_auxiliary_lanes):
                            # Go straight
                            auxiliary_main_indexes = [
                                i
                                for i, t in enumerate(in_auxiliary_turn_config)
                                if "S" in t or "s" in t
                            ]
                            auxiliary_main_count = len(auxiliary_main_indexes)
                            auxiliary_main_out_start = (
                                auxiliary_main_indexes[0]
                                if len(auxiliary_main_indexes) > 0
                                else 0
                            )
                            auxiliary_main_out_end = (
                                auxiliary_main_indexes[-1] + 1
                                if len(auxiliary_main_indexes) > 0
                                else len(in_auxiliary_lanes)
                            )
                            # Around
                            auxiliary_around_count = len(
                                [
                                    t
                                    for t in in_auxiliary_turn_config
                                    if "A" in t or "a" in t
                                ]
                            )
                            # Turn left
                            auxiliary_left_count = len(
                                [
                                    t
                                    for t in in_auxiliary_turn_config
                                    if "L" in t or "l" in t
                                ]
                            )
                            # Turn right
                            auxiliary_right_count = len(
                                [
                                    t
                                    for t in in_auxiliary_turn_config
                                    if "R" in t or "r" in t
                                ]
                            )
                            # Determine whether turn_config can be used to connect to lane. If the default configuration cannot be used,
                            # if (
                            #     (
                            #         len(out_around_groups) == 0
                            #         and ((auxiliary_around_count > 0))
                            #     )
                            #     or (
                            #         len(out_left_groups) == 0
                            #         and ((auxiliary_left_count > 0))
                            #     )
                            #     or (
                            #         len(out_right_groups) == 0
                            #         and ((auxiliary_right_count > 0))
                            #     )
                            # ):

                            if jid in bad_turn_config_jids:
                                auxiliary_around_count = (
                                    default_auxiliary_around_count
                                    if auxiliary_around_count == 0
                                    else auxiliary_around_count
                                )
                                auxiliary_left_count = (
                                    default_auxiliary_left_count
                                    if auxiliary_left_count == 0
                                    else auxiliary_left_count
                                )
                                auxiliary_right_count = (
                                    default_auxiliary_right_count
                                    if auxiliary_right_count == 0
                                    else auxiliary_right_count
                                )
                                auxiliary_main_count = (
                                    default_auxiliary_main_count
                                    if auxiliary_main_count == 0
                                    else auxiliary_main_count
                                )
                        else:
                            auxiliary_around_count = default_auxiliary_around_count
                            auxiliary_left_count = default_auxiliary_left_count
                            auxiliary_right_count = default_auxiliary_right_count
                            auxiliary_main_count = default_auxiliary_main_count
                            auxiliary_main_out_start = default_auxiliary_main_out_start
                            auxiliary_main_out_end = default_auxiliary_main_out_end
                        # connect lanes
                        if around_count > 0:
                            if key == (1, 1):
                                for out_angle, out_way_ids in out_around_groups:
                                    (
                                        out_main_wid,
                                        out_auxiliary_wids,
                                    ) = classify_main_auxiliary_wid(
                                        out_way_ids, out_angle, "out_ways"
                                    )
                                    out_main_lanes = self.map_roads[out_main_wid][
                                        "lanes"
                                    ]
                                    out_auxiliary_lanes = [
                                        l
                                        for wid in out_auxiliary_wids
                                        for l in self.map_roads[wid]["lanes"]
                                    ]
                                    # main road to main road
                                    junc_lanes += self._connect_lane_group(
                                        in_lanes=in_main_lanes,
                                        out_lanes=out_main_lanes,
                                        lane_turn=mapv2.LANE_TURN_AROUND,
                                        lane_type=mapv2.LANE_TYPE_DRIVING,
                                        junc_id=j_uid,
                                    )
                                    # auxiliary road to auxiliary road
                                    if (
                                        len(in_auxiliary_lanes) > 0
                                        and len(out_auxiliary_lanes) > 0
                                    ):
                                        junc_lanes += self._connect_lane_group(
                                            in_lanes=in_auxiliary_lanes,
                                            out_lanes=out_auxiliary_lanes,
                                            lane_turn=mapv2.LANE_TURN_AROUND,
                                            lane_type=mapv2.LANE_TYPE_DRIVING,
                                            junc_id=j_uid,
                                        )
                                    # When there is no out auxiliary road, connect the in auxiliary road to the main road
                                    if (
                                        len(in_auxiliary_lanes) > 0
                                        and not len(out_auxiliary_lanes) > 0
                                    ):
                                        junc_lanes += self._connect_lane_group(
                                            in_lanes=in_auxiliary_lanes,
                                            out_lanes=out_main_lanes,
                                            lane_turn=mapv2.LANE_TURN_AROUND,
                                            lane_type=mapv2.LANE_TYPE_DRIVING,
                                            junc_id=j_uid,
                                        )
                            else:
                                for out_angle, out_way_ids in out_around_groups:
                                    (
                                        out_main_wid,
                                        out_auxiliary_wids,
                                    ) = classify_main_auxiliary_wid(
                                        out_way_ids, out_angle, "out_ways"
                                    )
                                    out_main_lanes = self.map_roads[out_main_wid][
                                        "lanes"
                                    ]
                                    out_auxiliary_lanes = [
                                        l
                                        for wid in out_auxiliary_wids
                                        for l in self.map_roads[wid]["lanes"]
                                    ]
                                    # main road to main road
                                    if (
                                        len(in_main_lanes[:around_count]) > 0
                                        and len(out_main_lanes[:around_count]) > 0
                                    ):
                                        junc_lanes += self._connect_lane_group(
                                            in_lanes=in_main_lanes[:around_count],
                                            out_lanes=out_main_lanes[:around_count],
                                            lane_turn=mapv2.LANE_TURN_AROUND,
                                            lane_type=mapv2.LANE_TYPE_DRIVING,
                                            junc_id=j_uid,
                                        )
                                    # auxiliary road to auxiliary road
                                    if (
                                        len(in_auxiliary_lanes[:auxiliary_around_count])
                                        > 0
                                        and len(
                                            out_auxiliary_lanes[:auxiliary_around_count]
                                        )
                                        > 0
                                    ):
                                        junc_lanes += self._connect_lane_group(
                                            in_lanes=in_auxiliary_lanes[
                                                :auxiliary_around_count
                                            ],
                                            out_lanes=out_auxiliary_lanes[
                                                :auxiliary_around_count
                                            ],
                                            lane_turn=mapv2.LANE_TURN_AROUND,
                                            lane_type=mapv2.LANE_TYPE_DRIVING,
                                            junc_id=j_uid,
                                        )
                                    # When there is no out auxiliary road, connect the in auxiliary road to the main road
                                    if (
                                        len(in_auxiliary_lanes[:auxiliary_around_count])
                                        > 0
                                        and not len(out_auxiliary_lanes) > 0
                                        and len(out_main_lanes[:around_count]) > 0
                                    ):
                                        junc_lanes += self._connect_lane_group(
                                            in_lanes=in_auxiliary_lanes[
                                                :auxiliary_around_count
                                            ],
                                            out_lanes=out_main_lanes[:around_count],
                                            lane_turn=mapv2.LANE_TURN_AROUND,
                                            lane_type=mapv2.LANE_TYPE_DRIVING,
                                            junc_id=j_uid,
                                        )
                        if right_count > 0:
                            for out_angle, out_way_ids in out_right_groups:
                                (
                                    out_main_wid,
                                    out_auxiliary_wids,
                                ) = classify_main_auxiliary_wid(
                                    out_way_ids, out_angle, "out_ways"
                                )
                                out_main_lanes = self.map_roads[out_main_wid]["lanes"]
                                out_auxiliary_lanes = [
                                    l
                                    for wid in out_auxiliary_wids
                                    for l in self.map_roads[wid]["lanes"]
                                ]
                                # main road to main road
                                if (
                                    len(in_main_lanes[-right_count:]) > 0
                                    and len(out_main_lanes[-right_count:]) > 0
                                ):
                                    junc_lanes += self._connect_lane_group(
                                        in_lanes=in_main_lanes[-right_count:],
                                        out_lanes=out_main_lanes[-right_count:],
                                        lane_turn=mapv2.LANE_TURN_RIGHT,
                                        lane_type=mapv2.LANE_TYPE_DRIVING,
                                        junc_id=j_uid,
                                    )
                                # auxiliary road to auxiliary road
                                if (
                                    len(in_auxiliary_lanes[-auxiliary_right_count:]) > 0
                                    and len(
                                        out_auxiliary_lanes[-auxiliary_right_count:]
                                    )
                                    > 0
                                ):
                                    junc_lanes += self._connect_lane_group(
                                        in_lanes=in_auxiliary_lanes[
                                            -auxiliary_right_count:
                                        ],
                                        out_lanes=out_auxiliary_lanes[
                                            -auxiliary_right_count:
                                        ],
                                        lane_turn=mapv2.LANE_TURN_RIGHT,
                                        lane_type=mapv2.LANE_TYPE_DRIVING,
                                        junc_id=j_uid,
                                    )
                                # When there is no out auxiliary road, connect the in auxiliary road to the main road
                                if (
                                    len(in_auxiliary_lanes[-auxiliary_right_count:]) > 0
                                    and not len(out_auxiliary_lanes) > 0
                                    and len(out_main_lanes[-right_count:]) > 0
                                ):
                                    junc_lanes += self._connect_lane_group(
                                        in_lanes=in_auxiliary_lanes[
                                            -auxiliary_right_count:
                                        ],
                                        out_lanes=out_main_lanes[-right_count:],
                                        lane_turn=mapv2.LANE_TURN_RIGHT,
                                        lane_type=mapv2.LANE_TYPE_DRIVING,
                                        junc_id=j_uid,
                                    )
                        if left_count > 0:
                            for out_angle, out_way_ids in out_left_groups:
                                (
                                    out_main_wid,
                                    out_auxiliary_wids,
                                ) = classify_main_auxiliary_wid(
                                    out_way_ids, out_angle, "out_ways"
                                )
                                out_main_lanes = self.map_roads[out_main_wid]["lanes"]
                                out_auxiliary_lanes = [
                                    l
                                    for wid in out_auxiliary_wids
                                    for l in self.map_roads[wid]["lanes"]
                                ]
                                # main road to main road
                                if (
                                    len(in_main_lanes[:left_count]) > 0
                                    and len(out_main_lanes[:left_count]) > 0
                                ):
                                    junc_lanes += self._connect_lane_group(
                                        in_lanes=in_main_lanes[:left_count],
                                        out_lanes=out_main_lanes[:left_count],
                                        lane_turn=mapv2.LANE_TURN_LEFT,
                                        lane_type=mapv2.LANE_TYPE_DRIVING,
                                        junc_id=j_uid,
                                    )
                                # auxiliary road to auxiliary road
                                if (
                                    len(in_auxiliary_lanes[:auxiliary_left_count]) > 0
                                    and len(out_auxiliary_lanes[:auxiliary_left_count])
                                    > 0
                                ):
                                    junc_lanes += self._connect_lane_group(
                                        in_lanes=in_auxiliary_lanes[
                                            :auxiliary_left_count
                                        ],
                                        out_lanes=out_auxiliary_lanes[
                                            :auxiliary_left_count
                                        ],
                                        lane_turn=mapv2.LANE_TURN_LEFT,
                                        lane_type=mapv2.LANE_TYPE_DRIVING,
                                        junc_id=j_uid,
                                    )
                                # When there is no out auxiliary road, connect the in auxiliary road to the main road
                                if (
                                    len(in_auxiliary_lanes[:auxiliary_left_count]) > 0
                                    and not len(out_auxiliary_lanes) > 0
                                    and len(out_main_lanes[:left_count])
                                ):
                                    junc_lanes += self._connect_lane_group(
                                        in_lanes=in_auxiliary_lanes[
                                            :auxiliary_left_count
                                        ],
                                        out_lanes=out_main_lanes[:left_count],
                                        lane_turn=mapv2.LANE_TURN_LEFT,
                                        lane_type=mapv2.LANE_TYPE_DRIVING,
                                        junc_id=j_uid,
                                    )
                        if main_count > 0 and out_main_group is not None:
                            assert (
                                out_main_group is not None
                            ), f"Junction {jid} arranged straight out_ways but no out_ways to connect!"
                            out_angle, out_way_ids = out_main_group
                            (
                                out_main_wid,
                                out_auxiliary_wids,
                            ) = classify_main_auxiliary_wid(
                                out_way_ids, out_angle, "out_ways"
                            )
                            out_main_lanes = self.map_roads[out_main_wid]["lanes"]
                            out_auxiliary_lanes = [
                                l
                                for wid in out_auxiliary_wids
                                for l in self.map_roads[wid]["lanes"]
                            ]
                            # main road to main road
                            if (
                                len(in_main_lanes[main_out_start:main_out_end])
                            ) > 0 and len(out_main_lanes) > 0:
                                junc_lanes += self._connect_lane_group(
                                    in_lanes=in_main_lanes[main_out_start:main_out_end],
                                    out_lanes=out_main_lanes,
                                    lane_turn=mapv2.LANE_TURN_STRAIGHT,
                                    lane_type=mapv2.LANE_TYPE_DRIVING,
                                    junc_id=j_uid,
                                )
                            # auxiliary road to auxiliary road
                            if (
                                len(
                                    in_auxiliary_lanes[
                                        auxiliary_main_out_start:auxiliary_main_out_end
                                    ]
                                )
                                > 0
                                and len(out_auxiliary_lanes) > 0
                            ):
                                junc_lanes += self._connect_lane_group(
                                    in_lanes=in_auxiliary_lanes[
                                        auxiliary_main_out_start:auxiliary_main_out_end
                                    ],
                                    out_lanes=out_auxiliary_lanes,
                                    lane_turn=mapv2.LANE_TURN_STRAIGHT,
                                    lane_type=mapv2.LANE_TYPE_DRIVING,
                                    junc_id=j_uid,
                                )
                            # When there is no in auxiliary road, connect the in main road to the out auxiliary road
                            if (
                                not len(in_auxiliary_lanes) > 0
                                and len(out_auxiliary_lanes) > 0
                                and len(in_main_lanes[main_out_start:main_out_end]) > 0
                            ):
                                junc_lanes += self._connect_lane_group(
                                    in_lanes=in_main_lanes[main_out_start:main_out_end],
                                    out_lanes=out_auxiliary_lanes,
                                    lane_turn=mapv2.LANE_TURN_STRAIGHT,
                                    lane_type=mapv2.LANE_TYPE_DRIVING,
                                    junc_id=j_uid,
                                )
                            # When there is no out auxiliary road, connect the in auxiliary road to the main road
                            if (
                                len(
                                    in_auxiliary_lanes[
                                        auxiliary_main_out_start:auxiliary_main_out_end
                                    ]
                                )
                                > 0
                                and not len(out_auxiliary_lanes) > 0
                                and len(out_main_lanes) > 0
                            ):
                                junc_lanes += self._connect_lane_group(
                                    in_lanes=in_auxiliary_lanes[
                                        auxiliary_main_out_start:auxiliary_main_out_end
                                    ],
                                    out_lanes=out_main_lanes,
                                    lane_turn=mapv2.LANE_TURN_STRAIGHT,
                                    lane_type=mapv2.LANE_TYPE_DRIVING,
                                    junc_id=j_uid,
                                )
                # Check whether every road in the junction is connected
                valid_turn_config = True
                for _, in_way_ids in in_way_groups:
                    if valid_turn_config:
                        for in_way_id in in_way_ids:
                            in_map_road = self.map_roads[in_way_id]
                            if all(
                                len(self.lane2data[in_lane]["out"]) == 0
                                for in_lane in in_map_road["lanes"]
                            ):
                                logging.warning(
                                    f"No available out ways for {in_way_id} to connect due to invalid turn_config {in_map_road['turn_config']}, available turn groups: {in_way_id2available_conn[in_way_id]}"
                                )
                                valid_turn_config = False
                for _, out_way_ids in out_way_groups:
                    if valid_turn_config:
                        for out_way_id in out_way_ids:
                            out_map_road = self.map_roads[out_way_id]
                            if all(
                                len(self.lane2data[out_lane]["in"]) == 0
                                for out_lane in out_map_road["lanes"]
                            ):
                                logging.warning(
                                    f"No in ways connected to {out_way_id} due to invalid turn_config of in_ways, available turn groups for all in_ways: {in_way_id2available_conn}"
                                )
                                valid_turn_config = False
                if not valid_turn_config and jid not in bad_turn_config_jids:
                    logging.warning(f"Invalid turn_config at {jid}")
                    if self.strict_mode:
                        raise Exception("Encounter invalid road turn config.")
                    bad_turn_config_jids.add(jid)
                    # Delete all connection relationships between road lane and junc lane
                    # Delete all turn_config and take the default value
                    # Clear all roads expanded under this junction
                    for lane_id in range(pre_lane_uid, self.lane_uid):
                        self._delete_lane(lane_id, delete_road=True)
                    self.lane_uid = pre_lane_uid
                    continue

                jid_index += 1
                walk_pairs = self._create_junction_walk_pairs(
                    in_way_groups,
                    out_way_groups,
                    has_main_group_wids,
                    (x_center, y_center),
                )

                self.map_junctions[jid] = {
                    "lanes": junc_lanes,
                    "uid": j_uid,
                    "center": {
                        "x": x_center,
                        "y": y_center,
                        "z": z_center,
                    },
                    "walk_pairs": walk_pairs,
                }

    def _create_walking_lanes(self):
        logging.info(f"Creating walking lanes")
        pre_lane_uid = self.lane_uid

        def add_lane(wid, other_wid, walk_type, walk_lane_end_type):
            """
            Add a sidewalk to the left/right side of the lane. If a sidewalk already exists, return None. If the corresponding sidewalk should not be generated according to the rules, return None.
            """
            way = self.map_roads[wid]
            other_way = self.map_roads[other_wid]
            other_raw_way = self.ways[other_wid]
            if walk_type == "right":
                if wid in self.no_right_walk:
                    return None
                wlane = self._add_sidewalk(
                    wid=wid,
                    lane=way["lanes"][-1],
                    other_lane=other_way["lanes"][0],
                    # other_lane=other_raw_way["shapely"],
                    walk_type="right",
                    walk_lane_end_type=walk_lane_end_type,
                )
                return wlane
            if walk_type == "left":
                if wid in self.no_left_walk:
                    return None
                wlane = self._add_sidewalk(
                    wid=wid,
                    lane=way["lanes"][0],
                    other_lane=other_way["lanes"][-1],
                    # other_lane=other_raw_way["shapely"],
                    walk_type="left",
                    walk_lane_end_type=walk_lane_end_type,
                )
                return wlane

        def merge_walk_pairs(walk_pairs):
            """Merge walk pair"""
            # Because the way is affected by other junctions, some of the previously created walk pairs may be None, resulting in adjacent walk pairs that may be merged.
            if not walk_pairs:
                return []
            if len(walk_pairs) < 2:
                in_cur, out_cur = walk_pairs[0]["in_walk"], walk_pairs[0]["out_walk"]
                (in_lane_cur, in_type_cur) = in_cur
                (out_lane_cur, out_type_cur) = out_cur
                return [
                    {
                        "in_walk": (
                            [in_lane_cur] if in_lane_cur else [],
                            in_type_cur,
                        ),
                        "out_walk": (
                            [out_lane_cur] if out_lane_cur else [],
                            out_type_cur,
                        ),
                    }
                ]
            # Merge step 0
            # Only pairs of in walk and out walk are adjacent
            #  ⬇|  |   |  |⬆
            #  ⬇|  | + |  |⬆
            #  ⬇|  |   |  |⬆
            merge_0 = [walk_pairs[0]]
            for p in walk_pairs[1:]:
                in_cur, out_cur = p["in_walk"], p["out_walk"]
                in_pre, out_pre = merge_0[-1]["in_walk"], merge_0[-1]["out_walk"]
                (in_lane_pre, _) = in_pre
                (out_lane_pre, out_type_pre) = out_pre
                (in_lane_cur, in_type_cur) = in_cur
                (out_lane_cur, _) = out_cur
                if not in_lane_pre and not out_lane_cur:
                    merge_0[-1] = {
                        "in_walk": (
                            in_lane_cur,
                            in_type_cur,
                        ),
                        "out_walk": (
                            out_lane_pre,
                            out_type_pre,
                        ),
                    }
                else:
                    merge_0.append(p)
            if len(merge_0) < 2:
                in_cur, out_cur = merge_0[0]["in_walk"], merge_0[0]["out_walk"]
                (in_lane_cur, in_type_cur) = in_cur
                (out_lane_cur, out_type_cur) = out_cur
                return [
                    {
                        "in_walk": (
                            [in_lane_cur] if in_lane_cur else [],
                            in_type_cur,
                        ),
                        "out_walk": (
                            [out_lane_cur] if out_lane_cur else [],
                            out_type_cur,
                        ),
                    }
                ]
            # The end and beginning should also be merged
            in_cur, out_cur = merge_0[0]["in_walk"], merge_0[0]["out_walk"]
            in_pre, out_pre = merge_0[-1]["in_walk"], merge_0[-1]["out_walk"]
            (in_lane_pre, _) = in_pre
            (out_lane_pre, out_type_pre) = out_pre
            (in_lane_cur, in_type_cur) = in_cur
            (out_lane_cur, _) = out_cur
            if not in_lane_pre and not out_lane_cur:
                merge_0[-1] = {
                    "in_walk": (
                        in_lane_cur,
                        in_type_cur,
                    ),
                    "out_walk": (
                        out_lane_pre,
                        out_type_pre,
                    ),
                }
                merge_0 = merge_0[1:]

            # Merge step 1
            # Only pairs of in walk or out walk are adjacent
            #  |  |⬆   |  |⬆
            #  |  |⬆ + |  |⬆
            #  |  |⬆   |  |⬆

            merge_1 = []
            i = 0
            while i < len(merge_0):
                in_cur, out_cur = merge_0[i]["in_walk"], merge_0[i]["out_walk"]
                (in_lane_cur, in_type_cur) = in_cur
                (out_lane_cur, out_type_cur) = out_cur

                if not in_lane_cur and out_lane_cur:
                    merge_1.append(
                        {
                            "in_walk": (
                                [],
                                in_type_cur,
                            ),
                            "out_walk": (
                                [out_lane_cur],
                                out_type_cur,
                            ),
                        }
                    )
                    i += 1
                    while i < len(merge_0):
                        in_next, out_next = (
                            merge_0[i]["in_walk"],
                            merge_0[i]["out_walk"],
                        )
                        (in_lane_next, in_type_next) = in_next
                        (out_lane_next, out_type_next) = out_next
                        if (
                            not in_lane_next
                            and out_lane_next
                            and out_type_next == out_type_cur
                        ):
                            merge_1[-1]["out_walk"][0].append(out_lane_next)
                            i += 1
                        else:
                            break
                if in_lane_cur and not out_lane_cur:
                    merge_1.append(
                        {
                            "in_walk": (
                                [in_lane_cur],
                                in_type_cur,
                            ),
                            "out_walk": (
                                [],
                                out_type_cur,
                            ),
                        }
                    )
                    i += 1
                    while i < len(merge_0):
                        in_next, out_next = (
                            merge_0[i]["in_walk"],
                            merge_0[i]["out_walk"],
                        )
                        (in_lane_next, in_type_next) = in_next
                        (out_lane_next, out_type_next) = out_next
                        if (
                            in_lane_next
                            and not out_lane_next
                            and in_type_next == in_type_cur
                        ):
                            merge_1[-1]["in_walk"][0].append(in_lane_next)
                            i += 1
                        else:
                            break

                if in_lane_cur and out_lane_cur:
                    merge_1.append(
                        {
                            "in_walk": (
                                [in_lane_cur],
                                in_type_cur,
                            ),
                            "out_walk": (
                                [out_lane_cur],
                                out_type_cur,
                            ),
                        }
                    )
                    i += 1
                if not in_lane_cur and not out_lane_cur:
                    i += 1
            if len(merge_1) < 2:
                return merge_1
            # The end and beginning should also be merged
            in_next, out_next = merge_1[0]["in_walk"], merge_1[0]["out_walk"]
            in_cur, out_cur = merge_1[-1]["in_walk"], merge_1[-1]["out_walk"]
            (in_lanes_cur, in_type_cur) = in_cur
            (out_lanes_cur, out_type_cur) = out_cur
            (in_lanes_next, in_type_next) = in_next
            (out_lanes_next, out_type_next) = out_next
            if not in_lanes_cur and not in_lanes_next and out_type_cur == out_type_next:
                merge_1[-1] = {
                    "in_walk": (
                        [],
                        in_type_cur,
                    ),
                    "out_walk": (
                        out_lanes_cur + out_lanes_next,
                        out_type_cur,
                    ),
                }
                merge_1 = merge_1[1:]
            if not out_lanes_cur and not out_lanes_next and in_type_cur == in_type_next:
                merge_1[-1] = {
                    "in_walk": (
                        in_lanes_cur + in_lanes_next,
                        in_type_cur,
                    ),
                    "out_walk": (
                        [],
                        out_type_cur,
                    ),
                }
                merge_1 = merge_1[1:]

            return merge_1

        for jid, j in self.map_junctions.items():
            j_uid = j["uid"]
            walk_pairs = j["walk_pairs"]
            # Shallow copy
            junc_lanes = j["lanes"]
            # Based on the sidewalk identification of all junctions, store the entry and exit walk lanes in each direction and the source of the entry and exit lane (incoming way/outgoing way)
            filter_walk_pairs = []
            for p in walk_pairs:
                (in_wid, in_type) = p["in_walk"]
                (out_wid, out_type) = p["out_walk"]
                in_wlane = None
                out_wlane = None
                if in_type == "in_way" and out_type == "out_way":
                    in_wlane = add_lane(in_wid, out_wid, "right", "end")
                    in_wlane_l = add_lane(in_wid, out_wid, "left", "start")
                    out_wlane = add_lane(out_wid, in_wid, "right", "start")
                    out_wlane_l = add_lane(out_wid, in_wid, "left", "end")
                    if in_wlane_l or out_wlane_l:
                        if out_wlane_l or out_wlane:
                            filter_walk_pairs.append(
                                {
                                    "in_walk": (
                                        out_wlane_l,
                                        "out_way",
                                    ),
                                    "out_walk": (
                                        out_wlane,
                                        "out_way",
                                    ),
                                }
                            )
                        if in_wlane or in_wlane_l:
                            filter_walk_pairs.append(
                                {
                                    "in_walk": (
                                        in_wlane,
                                        "in_way",
                                    ),
                                    "out_walk": (
                                        in_wlane_l,
                                        "in_way",
                                    ),
                                }
                            )
                        continue
                if in_type == "in_way" and out_type == "in_way":
                    in_wlane = add_lane(in_wid, out_wid, "right", "end")
                    out_wlane = add_lane(out_wid, in_wid, "left", "start")

                if in_type == "out_way" and out_type == "out_way":
                    in_wlane = add_lane(in_wid, out_wid, "left", "end")
                    out_wlane = add_lane(out_wid, in_wid, "right", "start")
                if not in_wlane and not out_wlane:
                    continue
                filter_walk_pairs.append(
                    {
                        "in_walk": (
                            in_wlane,
                            in_type,
                        ),
                        "out_walk": (
                            out_wlane,
                            out_type,
                        ),
                    }
                )

            filter_walk_pairs = merge_walk_pairs(filter_walk_pairs)

            in_walk_data = [p["in_walk"] for p in filter_walk_pairs]
            out_walk_data = [p["out_walk"] for p in filter_walk_pairs]
            # Sidewalk connection method
            # After the previous processing, the counterclockwise arranged filter_walk_pairs are obtained. Each pair contains out walk lanes on the right and in walk lanes on the left (at least one of the two is not empty)
            # It can only be in the following two forms
            # 1. There are both in way and out way
            #        junc center
            #
            #   ⬇|  |         |  |⬆
            #   ⬇|  | out way |  |⬆ in way
            #
            # 2.Only in way/only out way
            #       junc center
            #
            #         ⬇|  |⬆
            #         ⬇|  |⬆ in/out way
            # The internal in out connection of the pair results in crosswalk
            #    ⬅  ⬅  ⬅  ⬅  ⬅                  ⬅
            #   ⬇|  |         |  |⬆         or    ⬇|  |⬆
            #   ⬇|  | out way |  |⬆ in way        ⬇|  |⬆ in/out way
            # Connect pair and pair in out that is shifted one bit to the left to get sidewalk.
            #               out way
            #                —— ——
            #                —— ——
            #                ➡ ➡
            #             ↗
            #          ↗
            #    |  |⬆
            #    |  |⬆ in way
            # connect sidewalks
            if len(out_walk_data) >= 2:
                for in_data, out_data in zip(
                    in_walk_data, out_walk_data[1:] + out_walk_data[:1]
                ):
                    (in_lanes, in_walk_type) = in_data
                    (out_lanes, out_walk_type) = out_data
                    if not in_lanes or not out_lanes:
                        continue
                    junc_lanes += self._connect_lane_group(
                        in_lanes=in_lanes,
                        out_lanes=out_lanes,
                        lane_turn=mapv2.LANE_TURN_STRAIGHT,
                        lane_type=mapv2.LANE_TYPE_WALKING,
                        junc_id=j_uid,
                        in_walk_type=in_walk_type,
                        out_walk_type=out_walk_type,
                    )
            # connect crosswalks

            for in_data, out_data in zip(in_walk_data, out_walk_data):
                (in_lanes, in_walk_type) = in_data
                (out_lanes, out_walk_type) = out_data
                if not in_lanes or not out_lanes:
                    continue
                junc_lanes += self._connect_lane_group(
                    in_lanes=in_lanes,
                    out_lanes=out_lanes,
                    lane_turn=mapv2.LANE_TURN_STRAIGHT,
                    lane_type=mapv2.LANE_TYPE_WALKING,
                    junc_id=j_uid,
                    in_walk_type=in_walk_type,
                    out_walk_type=out_walk_type,
                )
        # Merge the two parts divided into the beginning and end of the left and right sidewalks of each road
        del_lanes = set()
        for _, way in self.map_roads.items():
            right_walks = way["start_end_right_side_walk"]
            if len(right_walks) > 0:
                if len(right_walks) == 1:
                    right_walk_lane = list(right_walks.values())[0]
                else:
                    right_walk_end = right_walks["end"]
                    right_walk_start = right_walks["start"]
                    right_walk_lane = merge_line_start_end(
                        right_walk_start, right_walk_end
                    )
                    self.lane2data[right_walk_lane] = deepcopy(
                        self.lane2data[right_walk_start]
                    )
                    if not right_walk_start == right_walk_lane:
                        del_lanes.add(right_walk_start)
                    if not right_walk_end == right_walk_lane:
                        del_lanes.add(right_walk_end)
                self.map_lanes[self.lane_uid] = right_walk_lane
                self.lane2data[right_walk_lane]["uid"] = self.lane_uid
                way["right_sidewalk"].append(right_walk_lane)
                self.lane_uid += 1
            left_walks = way["start_end_left_side_walk"]
            if len(left_walks) > 0:
                if len(left_walks) == 1:
                    left_walk_lane = list(left_walks.values())[0]
                else:
                    left_walk_end = left_walks["end"]
                    left_walk_start = left_walks["start"]
                    left_walk_lane = merge_line_start_end(
                        left_walk_start, left_walk_end
                    )
                    self.lane2data[left_walk_lane] = deepcopy(
                        self.lane2data[left_walk_start]
                    )
                    if not left_walk_start == left_walk_lane:
                        del_lanes.add(left_walk_start)
                    if not left_walk_end == left_walk_lane:
                        del_lanes.add(left_walk_end)
                self.map_lanes[self.lane_uid] = left_walk_lane
                self.lane2data[left_walk_lane]["uid"] = self.lane_uid
                way["left_sidewalk"].append(left_walk_lane)
                self.lane_uid += 1
        for del_lane in del_lanes:
            del self.lane2data[del_lane]
        # Add walking lanes connections
        point2walk_lane = defaultdict(list)
        for line, l_data in self.lane2data.items():
            lane_id = l_data["uid"]
            if not l_data["type"] == mapv2.LANE_TYPE_WALKING:
                continue
            start_point = tuple(line.coords[0])
            point2walk_lane[start_point].append(
                (
                    lane_id,
                    mapv2.LANE_CONNECTION_TYPE_HEAD,
                )
            )
            end_point = tuple(line.coords[-1])
            point2walk_lane[end_point].append(
                (
                    lane_id,
                    mapv2.LANE_CONNECTION_TYPE_TAIL,
                )
            )
        for _, d in point2walk_lane.items():
            if len(d) <= 1:
                continue
            else:
                for i in range(len(d)):
                    lane_id, lane_conn_type = d[i]
                    line = self.map_lanes[lane_id]
                    l_data = self.lane2data[line]
                    for j, (other_lane_id, other_lane_conn_type) in enumerate(d):
                        if j == i:
                            continue
                        elif lane_conn_type == mapv2.LANE_CONNECTION_TYPE_HEAD:
                            l_data["in"].append(
                                {
                                    "id": other_lane_id,
                                    "type": (other_lane_conn_type),
                                },
                            )

                        elif lane_conn_type == mapv2.LANE_CONNECTION_TYPE_TAIL:
                            l_data["out"].append(
                                {
                                    "id": other_lane_id,
                                    "type": (other_lane_conn_type),
                                },
                            )

        # Delete sidewalks with too small connected components
        walk_lane_edges = []  # (lane_uid,lane_uid)
        walk_lane_uids = set()
        walk_graph = nx.DiGraph()
        bad_walk_lane_uids = set()
        for _, l_data in self.lane2data.items():
            lane_id = l_data["uid"]
            if not l_data["type"] == mapv2.LANE_TYPE_WALKING:
                continue
            walk_lane_uids.add(lane_id)
            lane_conn_out = l_data["out"]
            for d in lane_conn_out:
                walk_lane_edges.append((lane_id, d["id"]))
            # Isolated sidewalks are also removed
            if len(l_data["out"]) == 0 and len(l_data["in"]) == 0:
                bad_walk_lane_uids.add(lane_id)
        walk_graph.add_edges_from(walk_lane_edges)
        components = list(nx.weakly_connected_components(walk_graph))
        bad_walk_lane_uids = bad_walk_lane_uids.union(
            set(
                lane_id
                for component in components
                if len(component) < MIN_WALK_CONNECTED_COMPONENT
                for lane_id in component
            )
        )
        for lane_id in bad_walk_lane_uids:
            self._delete_lane(lane_id)
        walk_lane_uids = list(walk_lane_uids - bad_walk_lane_uids)
        new_lane_uids = [
            i for i in range(pre_lane_uid, pre_lane_uid + len(walk_lane_uids))
        ]
        self._reset_lane_uids(
            orig_lane_uids=walk_lane_uids, new_lane_uids=new_lane_uids
        )
        self.lane_uid = pre_lane_uid + len(walk_lane_uids)

    def _add_junc_lane_overlaps(self):
        """
        junc lanesadd overlap
        """
        for _, junc in self.output_junctions.items():
            junc_poly_lanes = [
                (l, LineString([[n["x"], n["y"]] for n in l["center_line"]["nodes"]]))
                for l in (self.output_lanes[lid] for lid in junc["lane_ids"])
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
                                    < np.pi / 4  # Close to 90 degrees
                                ):
                                    if (
                                        np.cross(start_vec_i, start_vec_j) < 0
                                    ):  # i is about 90 degrees to the right of j
                                        # ⬆
                                        # ⬆ j
                                        # ⬆
                                        # ➡➡➡➡i
                                        # ⬆
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
        lanes = {lid: d for lid, d in self.output_lanes.items()}
        roads = {rid: d for rid, d in self.output_roads.items()}
        juncs = {jid: d for jid, d in self.output_junctions.items()}

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
        for jid, j in juncs.items():
            # (in road id, out road id) -> [lane id]
            lane_groups = defaultdict(list)
            for lid in j["lane_ids"]:
                l = lanes[lid]
                if l["type"] != mapv2.LANE_TYPE_DRIVING:  # driving
                    # Only process traffic lanes
                    continue
                pres = l["predecessors"]
                assert (
                    len(pres) == 1
                ), f"Lane {lid} at junction {jid} has multiple predecessors!"
                pre_lane_id = pres[0]["id"]
                pre_lane = lanes[pre_lane_id]
                sucs = l["successors"]
                assert (
                    len(sucs) == 1
                ), f"Lane {lid} at junction {jid} has multiple successors!"
                suc_lane_id = sucs[0]["id"]
                suc_lane = lanes[suc_lane_id]
                pre_parent_id, suc_parent_id = (
                    pre_lane["parent_id"],
                    suc_lane["parent_id"],
                )
                if pre_parent_id >= JUNC_START_ID:
                    logging.warning(
                        f"Junction lane {lid} has predecessor lane {pre_lane_id} in junction {pre_parent_id}, all predecessors and successors of a junction lane should be road lanes."
                    )
                    continue
                if suc_parent_id >= JUNC_START_ID:
                    logging.warning(
                        f"Junction lane {lid} has successor lane {suc_lane_id} in junction {suc_parent_id}, all predecessors and successors of a junction lane should be road lanes."
                    )
                    continue
                key = (pre_parent_id, suc_parent_id)
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
                # Turn the Around into a left turn
                turns = [
                    mapv2.LANE_TURN_LEFT if t == mapv2.LANE_TURN_AROUND else t
                    for t in turns
                ]
                # Check if all turns are the same
                assert all(
                    t == turns[0] for t in turns
                ), f"Not all lane turns are the same at junction {jid} driving group"
                group["turn"] = turns[0]

    def _add_traffic_light(self):
        lanes = {lid: d for lid, d in self.output_lanes.items()}
        roads = {rid: d for rid, d in self.output_roads.items()}
        juncs = {jid: d for jid, d in self.output_junctions.items()}
        green_time = self.green_time
        yellow_time = self.yellow_time
        min_direction_group = self.traffic_light_min_direction_group
        generate_traffic_light(
            lanes, roads, juncs, green_time, yellow_time, min_direction_group
        )

    def _add_public_transport(self) -> Set[int]:
        logging.info("Adding public transport to map")
        public_road_uids = set()

        def create_subway_connections(geos: list, stas: list) -> List[List[int]]:
            # Create new subway lane road and junction
            station_connection_road_ids = []
            next_way_id = max(self.map_roads.keys()) + 1
            next_junc_id = max(self.map_junctions.keys()) + 1
            new_lanes = []
            for i, geo in enumerate(geos):
                pre_sta_name = stas[i]["name"]
                pre_sta_lane_matcher = stas[i]["subline_lane_matcher"]
                cur_sta_name = stas[i + 1]["name"]
                cur_sta_lane_matcher = stas[i + 1]["subline_lane_matcher"]
                coords = np.array(geo, dtype=np.float64)
                coords_z = (
                    coords[:, 2]
                    if coords.shape[1] > 2
                    else np.zeros((coords.shape[0], 1), dtype=np.float64)
                )
                coords_xy = np.stack(projector(*coords.T[:2]), axis=1)  # (N, 2)
                coords_xyz = np.column_stack([coords_xy, coords_z])  # (N, 3)
                lane_width = self.default_lane_width
                line = LineString(coords_xyz)
                if line.length > 3 * lane_width:
                    lane = ops.substring(line, lane_width, line.length - lane_width)
                else:
                    lane = ops.substring(
                        line, line.length / 3, line.length - line.length / 3
                    )
                lane = cast(LineString, lane)
                lane = offset_lane(lane, -0.495 * lane_width)
                # Public section routes will not be added to lanes repeatedly.
                if lane in self.lane2data:
                    station_connection_road_ids.append(
                        {
                            "road_ids": [self.lane2data[lane]["parent_id"]],
                        }
                    )
                    continue
                cur_matcher = {
                    "id": self.lane_uid,
                    "geo": lane,
                    # midpoint of polyline
                    "point": lane.interpolate(0.5, normalized=True).coords[:][0],
                    "length": lane.length,
                }
                pre_sta_lane_matcher.append(cur_matcher)
                cur_sta_lane_matcher.append(cur_matcher)
                new_lanes.append(lane)
                # Add new lane
                self.map_lanes[self.lane_uid] = lane
                # Add the connection relationship of the new lane
                self.lane2data[lane] = {
                    "uid": self.lane_uid,
                    "in": [],
                    "out": [],
                    "max_speed": DEFAULT_MAX_SPEED["SUBWAY"],
                    "type": mapv2.LANE_TYPE_DRIVING,
                    "turn": mapv2.LANE_TURN_STRAIGHT,
                    "width": lane_width,
                    "left_lane_ids": [],
                    "right_lane_ids": [],
                    "parent_id": self.road_uid,
                }
                self.map_roads[next_way_id] = {
                    "lanes": [lane],
                    "left_sidewalk": [],
                    "right_sidewalk": [],
                    "highway": "SUBWAY",
                    "max_speed": DEFAULT_MAX_SPEED["SUBWAY"],
                    "name": f"{pre_sta_name}->{cur_sta_name}",
                    "uid": self.road_uid,
                }
                station_connection_road_ids.append(
                    {
                        "road_ids": [self.road_uid],
                    }
                )  # There is only one road connecting adjacent subway stations.
                self.lane_uid += 1
                self.road_uid += 1
                next_way_id += 1
            for i in range(len(new_lanes) - 1):
                cur_sta_geo = stas[i + 1]["geo"]
                z_center = (
                    np.mean([c[2] for c in cur_sta_geo])
                    if all(len(c) > 2 for c in cur_sta_geo)
                    else 0
                )
                self.map_junctions[next_junc_id] = {
                    "lanes": self._connect_lane_group(
                        in_lanes=new_lanes[i : i + 1],
                        out_lanes=new_lanes[i + 1 : i + 2],
                        lane_turn=mapv2.LANE_TURN_STRAIGHT,
                        lane_type=mapv2.LANE_TYPE_DRIVING,
                        junc_id=self.junc_uid,
                    ),
                    "uid": self.junc_uid,
                    "center": {
                        "x": np.mean([c[0] for c in cur_sta_geo]),
                        "y": np.mean([c[1] for c in cur_sta_geo]),
                        "z": z_center,
                    },
                }
                self.junc_uid += 1
                next_junc_id += 1
            return station_connection_road_ids

        projector = self.projector
        public_transport = self.public_transport
        raw_stations = (
            public_transport["stations"] if public_transport is not None else []
        )
        raw_lines = public_transport["lines"] if public_transport is not None else []
        stations = {stype: {} for stype in STATION_CAPACITY.keys()}
        lines = {stype: {} for stype in STATION_CAPACITY.keys()}
        for sta in raw_stations:
            stype = sta.get("type", "")
            sname = sta.get("name", "")
            geo = sta["geo"]
            sid = sta["id"]
            if stype in stations:
                scapacity = sta.get("capacity", STATION_CAPACITY[stype])
                stations[stype][sid] = {
                    "name": sname,
                    "capacity": scapacity,
                    "geo": [projector(*c) for c in geo],
                    "subline_lane_matcher": [],  # for subway
                    "subline_geos": defaultdict(list),  # for bus
                }
        line_id = 0
        for l in raw_lines:
            ltype = l.get("type", "")
            lname = str(l.get("name", ""))
            sublines = l["sublines"]
            if ltype in lines:
                lines[ltype][line_id] = {}
                for sl in sublines:
                    slid = sl["id"]
                    slname = sl["name"]
                    stas = sl["stations"]
                    geos = sl["geo"]
                    station_connection_road_ids = (
                        []
                    )  # All road uids corresponding to each geo segment
                    schedules = sl["schedules"]
                    # Subway lines are independent of the map road network
                    if ltype == "SUBWAY":
                        station_connection_road_ids = create_subway_connections(
                            geos, [stations[ltype][sta_id] for sta_id in stas]
                        )
                    elif ltype == "BUS":
                        for i, sta_id in enumerate(stas):
                            sta = stations[ltype][sta_id]
                            if i - 1 >= 0:
                                sta["subline_geos"][slid].append(
                                    LineString([projector(*c) for c in geos[i - 1]])
                                )
                            if i < len(geos):
                                sta["subline_geos"][slid].append(
                                    LineString([projector(*c) for c in geos[i]])
                                )
                    lines[ltype][line_id][slid] = {
                        "name": slname,
                        "raw_station_ids": stas,
                        "schedules": schedules,
                        "station_connection_road_ids": station_connection_road_ids,
                        "parent_name": lname,
                    }
                    public_road_uids.update(
                        {rid for part in station_connection_road_ids for rid in part}
                    )
                line_id += 1

        self.public_transport_data = {
            "lines": lines,
            "stations": stations,
        }
        return public_road_uids

    def _add_reuse_aoi(self):
        """Match aoi to road network"""
        reuse_aois, reuse_pois = {}, {}
        d_right_lanes = [w["lanes"][-1] for w in self.map_roads.values() if w["lanes"]]
        d_matcher = [
            {
                "id": self.lane2data[geo]["uid"],
                "geo": geo,
                # middle point
                "point": geo.interpolate(0.5, normalized=True).coords[:][0],
                "length": geo.length,
            }
            for geo in d_right_lanes
        ]
        w_lanes = [
            l for l, d in self.lane2data.items() if d["type"] == mapv2.LANE_TYPE_WALKING
        ]
        w_matcher = [
            {
                "id": self.lane2data[geo]["uid"],
                "geo": geo,
                "point": geo.interpolate(0.5, normalized=True).coords[:][0],
                "length": geo.length,
            }
            for geo in w_lanes
        ]
        logging.info("Reusing AOIs")
        matchers = {
            "drive": d_matcher,
            "walk": w_matcher,
        }
        if type(self.net) == Map:
            (reuse_aois, reuse_pois) = match_map_aois(
                net=self.net, matchers=matchers, workers=self.workers
            )
        return (reuse_aois, reuse_pois)

    def _add_input_aoi(self):
        """Match aoi to road network"""
        added_aois, added_pois = {}, {}
        aois, stops = convert_aoi(
            self.raw_aois, self.public_transport_data, self.proj_str
        )
        pois = convert_poi(self.raw_pois, self.proj_str)
        aois, stops, pois = generate_aoi_poi(aois, pois, stops, self.workers)
        d_right_lanes = [
            w["lanes"][-1]
            for w in self.map_roads.values()
            if w["lanes"] and w["uid"] not in self.public_road_uids
        ]
        d_matcher = [
            {
                "id": self.lane2data[geo]["uid"],
                "geo": geo,
                # middle point
                "point": geo.interpolate(0.5, normalized=True).coords[:][0],
                "length": geo.length,
            }
            for geo in d_right_lanes
        ]
        w_lanes = [
            l
            for l, d in self.lane2data.items()
            if d["type"] == mapv2.LANE_TYPE_WALKING
            and d["parent_id"] not in self.public_road_uids
        ]
        w_matcher = [
            {
                "id": self.lane2data[geo]["uid"],
                "geo": geo,
                "point": geo.interpolate(0.5, normalized=True).coords[:][0],
                "length": geo.length,
            }
            for geo in w_lanes
        ]
        road_center_lanes = [
            w["lanes"][len(w["lanes"]) // 2]
            for w in self.map_roads.values()
            if w["lanes"] and w["uid"] not in self.public_road_uids
        ]  # Single point AOI used to split aggregation
        road_lane_matcher = [
            {
                "id": self.lane2data[geo]["uid"],
                "geo": geo,
                "point": geo.interpolate(0.5, normalized=True).coords[:][0],
                "length": geo.length,
            }
            # Roads that are too short will not participate in the segmentation of aggregated AOI
            for geo in road_center_lanes
            if geo.length > 100
        ]
        logging.info("Adding AOIs")
        matchers = {
            "drive": d_matcher,
            "walk": w_matcher,
            "road_lane": road_lane_matcher,
        }
        (added_aois, added_pois) = add_aoi_to_map(
            matchers=matchers,
            input_aois=aois,
            input_pois=pois,
            input_stops=stops,
            bbox=(self.min_lat, self.min_lon, self.max_lat, self.max_lon),
            projstr=self.proj_str,
            shp_path=self.landuse_shp_path,
        )
        # added_aois = add_aoi_pop(
        #     aois=added_aois,
        #     max_latitude=self.max_lat,
        #     max_longitude=self.max_lon,
        #     min_latitude=self.min_lat,
        #     min_longitude=self.min_lon,
        #     proj_str=self.proj_str,
        #     upsample_factor=UPSAMPLE_FACTOR,
        #     workers=self.workers,
        #     tif_path=self.pop_tif_path,
        # )
        return (added_aois, added_pois)

    def _add_all_aoi(self):
        # add PT
        self.public_road_uids = self._add_public_transport()
        if self.aoi_mode == "append":
            (reuse_aois, reuse_pois) = self._add_reuse_aoi()
        elif self.aoi_mode == "overwrite":
            (reuse_aois, reuse_pois) = ({}, {})
        else:
            raise ValueError(f"bad aoi_mode: {self.aoi_mode}")
        (added_aois, added_pois) = self._add_input_aoi()
        _item_uid_dict = defaultdict(dict)
        _aoi_uid = AOI_START_ID
        _poi_uid = POI_START_ID
        _all_aois, _all_pois = {}, {}
        # rearragne aoi and poi ids
        _all_items = ((reuse_aois, reuse_pois), (added_aois, added_pois))
        for _idx, (_aois, _pois) in enumerate(_all_items):
            for aid, _ in _aois.items():
                _item_uid_dict[_idx][aid] = _aoi_uid
                _aoi_uid += 1
            for pid, _ in _pois.items():
                _item_uid_dict[_idx][pid] = _poi_uid
                _poi_uid += 1
        for _idx, (_aois, _pois) in enumerate(_all_items):
            for aid, a in _aois.items():
                new_aid = _item_uid_dict[_idx][aid]
                # update poi_ids
                a["poi_ids"] = [_item_uid_dict[_idx][ii] for ii in a["poi_ids"]]
                # update uid
                a["id"] = new_aid
                _all_aois[new_aid] = a
            for pid, p in _pois.items():
                new_pid = _item_uid_dict[_idx][pid]
                # update aoi_id
                p["aoi_id"] = _item_uid_dict[_idx][p["aoi_id"]]
                # update uid
                p["id"] = new_pid
                _all_pois[new_pid] = p
        self.map_aois, self.map_pois = _all_aois, _all_pois

    def write2json(self, topo_path: str, output_path: str):
        # Write the widened road into the geojson file
        logging.info(f"Writing expanded topo file to {output_path}")
        with open(topo_path, "r") as f:
            expand_topo = json.load(f)
        for feature in expand_topo["features"]:
            feature_id = (
                feature["id"] if "id" in feature else feature["properties"]["id"]
            )
            if feature_id in self.ways:
                way = self.ways[feature_id]
                feature["properties"]["lanes"] = way["properties"]["lanes"]
                feature["properties"]["uid"] = way["uid"]
            if feature_id in self.junctions:
                junc = self.junctions[feature_id]
                feature["properties"]["uid"] = junc["uid"]
        with open(output_path, "w") as f:
            json.dump(expand_topo, f)

    def _post_process(self):
        """
        Map data post-processing
        """
        # Speed limit for non-straight lanes based on curvature
        turn_dict = {
            mapv2.LANE_TURN_AROUND: "AROUND",
            mapv2.LANE_TURN_LEFT: "LEFT",
            mapv2.LANE_TURN_RIGHT: "RIGHT",
        }

        def max_turn_speed(r):
            MAX_TYPICAL_LATERAL_ACC = 1.98  # 0.2g
            INF_LANE_MAX_SPEED = 10 / 3.6  # 10km/h
            return max(np.sqrt(MAX_TYPICAL_LATERAL_ACC * r), INF_LANE_MAX_SPEED)

        for line, l_data in self.lane2data.items():
            if l_data["turn"] == mapv2.LANE_TURN_STRAIGHT:
                continue
            else:
                lane_turn = turn_dict[l_data["turn"]]
                curvature = line_max_curvature(line)
                if curvature > CURVATURE_THRESHOLDS[lane_turn]:
                    max_speed_threshold = max_turn_speed(1 / curvature)
                    l_data["max_speed"] = min(max_speed_threshold, l_data["max_speed"])

    def get_output_map(self, name: str):
        """Post-processing converts map data into output format"""
        public_lines = self.public_transport_data["lines"]
        public_stations = self.public_transport_data["stations"]
        # output aoi
        logging.info("Making output aois")
        self.output_aois = {}
        for uid, d in self.map_aois.items():
            # d["name"] = "" # The specific name of the AOI needs to be obtained by running patcher. The default is an empty string.
            external = d.get("external", {})
            if "stop_id" in external:
                station_type = external["station_type"]
                stop_id = external["stop_id"]
                sta = public_stations[station_type][stop_id]
                sta["aoi_uid"] = uid
                d["name"] = sta["name"]
                if not "duration" in external:
                    external["duration"] = DEFAULT_STATION_DURATION[station_type]
            self.output_aois[uid] = d
        # output public transport
        self.output_public_transport = {}
        for line_type, lines in public_lines.items():
            for _, sublines in lines.items():
                matched_lines = []
                for _, subline in sublines.items():
                    aoi_ids = []
                    slname = subline["name"]
                    sl_parent_name = subline["parent_name"]
                    schedules = subline["schedules"]
                    if not schedules:
                        schedules = DEFAULT_SCHEDULES[line_type]
                    station_connection_road_ids = subline["station_connection_road_ids"]
                    for sta_id in subline["raw_station_ids"]:
                        sta = public_stations[line_type][sta_id]
                        if not "aoi_uid" in sta:
                            continue
                        else:
                            aoi_id = sta["aoi_uid"]
                            aoi_ids.append(aoi_id)
                    if len(aoi_ids) > 1:
                        matched_lines.append(
                            {
                                "name": slname,
                                "aoi_ids": aoi_ids,
                                "station_connection_road_ids": station_connection_road_ids,
                                "parent_name": sl_parent_name,
                                "schedules": schedules,
                            }
                        )
                if len(matched_lines) >= 1:
                    self.output_public_transport[self.public_transport_uid] = {
                        "sublines": matched_lines,
                        "type": line_type,
                    }
                    self.public_transport_uid += 1
        # If public transportation simulation is required, the map needs to be post-processed
        logging.info("Making output public transport")
        # output poi
        logging.info("Making output pois")
        self.output_pois = {uid: d for uid, d in self.map_pois.items()}
        # output road
        logging.info("Making output roads")
        self.wid2ruid = {}  # for drawing
        for wid, r in self.map_roads.items():
            rid = r["uid"]
            self.wid2ruid[wid] = rid
            # Process road lane
            # The lane order of road in the original osm2map is Lane (from left to right) Sidewalk (from left to right) The same process is done here
            road_lanes = r["lanes"]
            lane_ids = [self.lane2data[l]["uid"] for l in road_lanes]
            for index, l in enumerate(road_lanes):
                l_data = self.lane2data[l]
                cur_lane_type = l_data["type"]
                # Lane left and right relationship
                # left lane
                for i in range(0, index)[::-1]:
                    left_l = road_lanes[i]
                    left_data = self.lane2data[left_l]
                    if left_data["type"] == cur_lane_type:
                        l_data["left_lane_ids"].append(lane_ids[i])
                # right lane
                for i in range(index + 1, len(lane_ids)):
                    right_l = road_lanes[i]
                    right_data = self.lane2data[right_l]
                    if right_data["type"] == cur_lane_type:
                        l_data["right_lane_ids"].append(lane_ids[i])
            # Add sidewalk
            road_lanes += r["left_sidewalk"]
            road_lanes += r["right_sidewalk"]
            # Update lane_ids after adding sidewalks
            lane_ids = [self.lane2data[l]["uid"] for l in road_lanes]
            self.output_roads[rid] = {
                "id": rid,
                "lane_ids": lane_ids,
                "name": r["name"],
                "external": {
                    "highway": r["highway"],
                    "name": r["name"],
                },
            }
        # output lane
        logging.info("Making output lanes")
        # Transform LineString to MOSS format

        def line2nodes(line):
            res = {"nodes": [{"x": c[0], "y": c[1], "z": c[2]} for c in line.coords]}
            return res

        for lid, l in self.map_lanes.items():
            l_data = self.lane2data[l]
            self.output_lanes[lid] = {
                "id": lid,
                "type": l_data["type"],
                "turn": l_data["turn"],
                "max_speed": l_data["max_speed"],
                "length": l.length,
                "width": l_data["width"],
                "center_line": line2nodes(l),
                "predecessors": l_data["in"],
                "successors": l_data["out"],
                "left_lane_ids": l_data["left_lane_ids"],
                "right_lane_ids": l_data["right_lane_ids"],
                "parent_id": l_data["parent_id"],
                "overlaps": [],
                "aoi_ids": [],
            }
        # aoi ids
        for _, a in self.map_aois.items():
            for pos in a["driving_positions"]:
                self.output_lanes[pos["lane_id"]]["aoi_ids"].append(a["id"])
            for pos in a["walking_positions"]:
                self.output_lanes[pos["lane_id"]]["aoi_ids"].append(a["id"])
        # output junction
        logging.info("Making output junctions")
        for _, j in self.map_junctions.items():
            junc_id = j["uid"]
            junc_center = j["center"]
            junc_lids = [self.lane2data[l]["uid"] for l in j["lanes"]]
            self.output_junctions[junc_id] = {
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
            for j in self.output_lanes.values()
            for k in ["center_line"]
            for i in j[k]["nodes"]
        ]
        x, y = [*zip(*xy)]
        header = {
            "name": name,
            "date": time.strftime("%a %b %d %H:%M:%S %Y"),
            "north": max(y),
            "south": min(y),
            "west": min(x),
            "east": max(x),
            "projection": self.proj_str,
        }
        logging.info(
            f"{len(self.output_lanes)} Lanes  {len(self.output_roads)} Roads  {len(self.output_junctions)} Juncions"
        )

        logging.info(f"{len(self.output_aois)} AOIs")
        logging.info(f"{len(self.output_pois)} POIs")
        output_map = {
            "header": header,
            "lanes": list(self.output_lanes.values()),
            "roads": list(self.output_roads.values()),
            "junctions": list(self.output_junctions.values()),
            "aois": list(self.output_aois.values()),
            "pois": list(self.output_pois.values()),
            "public_transport": list(self.output_public_transport.values()),
        }
        return output_map

    def write2db(self, coll: Collection, name: str):
        output_map = self.get_output_map(name)
        logging.info("Writing to MongoDB")
        BATCH = 500  # The number of writes to mongodb each time to prevent BSONSize errors from being written too much at one time
        coll.drop()
        coll.insert_one({"class": "header", "data": output_map["header"]})
        for class_name, data_type in [
            ("lane", "lanes"),
            ("road", "roads"),
            ("junction", "junctions"),
            ("aoi", "aois"),
            ("poi", "pois"),
        ]:
            map_data = [{"class": class_name, "data": d} for d in output_map[data_type]]
            for i in range(0, len(map_data), BATCH):
                coll.insert_many(map_data[i : i + BATCH], ordered=False)

    def build(self, name: str):
        if not self.from_pb:
            self._classify()
            self._classify_main_way_ids()  # Identify main and auxiliary roads before connecting to the road network for road shortening
            self._create_junction_for_1_n()
            self._create_junction_for_n_n()
            self._create_walking_lanes()
            self._expand_remain_roads()
            self._post_process()
        self._add_all_aoi()
        output_map = self.get_output_map(name)
        output_format_check(output_map, self.output_lane_length_check)
        return output_map


# def main():
#     # Configure log and store it in the file mapbuilder2.log
#     logging.basicConfig(
#         filename="mapbuilder2-topo2map.log",
#         filemode="w",
#         level=logging.INFO,
#         format="%(asctime)s %(levelname)s %(message)s",
#     )
#     handler = logging.StreamHandler()
#     handler.setLevel(logging.INFO)
#     handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
#     logging.getLogger().addHandler(handler)
#     logging.info(f"Loading parameters and topo file {args.topo}")
#     # Load parameters and topology files
#     t2m = Topo2Map(args.topo, PROJSTR)
#     # step 1 Create road network topology
#     t2m.classify()
#     t2m.create_junction_for_1_n()
#     t2m.create_junction_for_n_n()
#     t2m.create_walking_lanes()
#     t2m.expand_remain_roads()
#     if args.aoi:
#         # step 1.5 Create AOI
#         t2m.create_aoi()
#         # step 2
#         # Join aoi
#         t2m.add_aoi()
#     # Output the expanded geojson file
#     if EXPAND_ROADS:
#         t2m.write2json(args.topo, args.output)
#     # Warehouse
#     t2m.write2db()
#     # Draw junction
#     # Junctions in t2m.map_junctions
#     for key in self._junction_keys:
#         jids = list(t2m.junction_types[key])
#         if len(jids) < 1:
#             continue
#         folder = f"cache/img/{key[0]}_{key[1]}"
#         shutil.rmtree(folder, ignore_errors=True)
#         os.makedirs(folder, exist_ok=True)
#         wfolder = f"cache/img/walk_{key[0]}_{key[1]}"
#         shutil.rmtree(wfolder, ignore_errors=True)
#         os.makedirs(wfolder, exist_ok=True)
#         random.seed(2023)
#         random.shuffle(jids)
#         for jid in jids[:20]:
#             t2m.draw_junction(jid, f"{folder}/{jid}.png", 20)
#             t2m.draw_walk_junction(jid, f"{wfolder}/{jid}.png", 80)


# if __name__ == "__main__":
#     main()
