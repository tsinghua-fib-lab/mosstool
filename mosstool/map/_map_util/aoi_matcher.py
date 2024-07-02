import logging
import sys
from collections import defaultdict
from math import ceil
from multiprocessing import Pool
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import pyproj
import shapely.ops as ops
from scipy.spatial import KDTree
from shapely.affinity import scale
from shapely.geometry import (LineString, MultiLineString, MultiPoint,
                              MultiPolygon, Point, Polygon)
from shapely.strtree import STRtree

from ...type import AoiType
from .._util.angle import abs_delta_angle, delta_angle
from .._util.line import (connect_line_string, get_line_angle,
                          get_start_vector, line_extend, offset_lane)
from .aoiutils import geo_coords

# ATTENTION: In order to achieve longer distance POI merging, the maximum recursion depth needs to be modified.
sys.setrecursionlimit(50000)

from .const import *


def _split_merged_poi_unit(points):
    """Polygon formed by dividing merged_poi using road network"""
    global road_lane_matcher
    CONVEX_DIS_GATE = 50
    res_aoi = []
    res_poi = []
    if (
        len(points) == 1
    ):  # Indicates that this poi cannot be merged if there are no other poi nearby.
        res_poi += points
    else:
        coord = [[p["geo"].x, p["geo"].y] for p in points]
        if len(coord) == 2:
            # Add a point to form an isosceles triangle
            coord.append([coord[0][0], coord[-1][1]])
        convex = MultiPoint(coord).convex_hull
        if not isinstance(convex, Polygon):
            res_poi += points
        else:
            # Post-processing uses road network segmentation to form polygons
            lines = []  # dividing lines
            (x, y) = geo_coords(convex.centroid)[0][:2]
            length = convex.length
            # {'id', 'geo', 'point', 'length'}
            for lane in road_lane_matcher:
                mid_x, mid_y = lane["point"][:2]
                dis_upper_bound = (
                    np.sqrt(2) * (abs(x - mid_x) + abs(y - mid_y))
                    - length
                    - lane["length"]
                )
                if dis_upper_bound < 2 * CONVEX_DIS_GATE:
                    extend_lane = line_extend(lane["geo"], lane["length"])
                    intersect_geo = extend_lane.intersection(convex)
                    # Only keep valid LineStrings
                    if intersect_geo and isinstance(intersect_geo, LineString):
                        lines.append(
                            line_extend(intersect_geo, intersect_geo.length / 2)
                        )
            if len(lines) > 0:
                # There are other dividing lines besides borders
                lines.append(convex.boundary)
                lines = ops.unary_union(lines)
                if isinstance(
                    lines, MultiLineString
                ):  # The union has parts outside the boundary. Otherwise, there is only one circle around the boundary and cannot be linemerged.
                    lines = ops.linemerge(lines)
                polygons = ops.polygonize(lines)
            else:
                # only borders
                polygons = [convex]
            # Classify which p_s are included in each segmentation result
            contained_points = set()
            polygons = sorted(
                polygons, key=lambda x: -x.area
            )  # Larger areas are preferred
            for poly in polygons:
                extend_poly = poly.buffer(15.0)  # Prevent boundary judgment errors
                p_s = []
                for p in points:
                    if p["id"] in contained_points:
                        continue
                    else:
                        if extend_poly.contains(p["geo"]):
                            p_s.append(p)
                            contained_points.add(p["id"])
                if (
                    not p_s
                ):  # It is possible to segment polygons without internal poi. Ignore
                    continue
                aoi = {}
                aoi["id"] = p_s[0]["id"]
                aoi["geo"] = scale(
                    poly, xfact=0.95, yfact=0.95
                )  # Scale to avoid boundary overlap
                land_types = p_s[0]["external"].get("land_types", defaultdict(float))
                names = p_s[0]["external"].get("names", defaultdict(float))
                inner_pois = []
                for p in p_s:
                    inner_pois.extend(p["external"].get("poi_id", []))
                aoi_pop = int(sum(p["external"].get("population", 0) for p in p_s))
                for p in p_s[1:]:
                    child_land_types = p["external"].get(
                        "land_types", defaultdict(float)
                    )
                    child_names = p["external"].get("names", defaultdict(float))
                    for land_type, area in child_land_types.items():
                        land_types[land_type] += area
                    for child_name, area in child_names.items():
                        names[child_name] += area

                aoi["external"] = {
                    "inner_poi": inner_pois,
                    "population": aoi_pop,
                    "land_types": land_types,
                    "names": names,
                }
                res_aoi.append(aoi)
            assert len(contained_points) == len(
                points
            )  # If it can be solved without adjusting the buffer size
    return (res_aoi, res_poi)


def _find_aoi_parent_unit(i_aoi):
    """
    Find out when aoi is contained by other aoi
    """
    global aois_to_merge
    SQRT2 = 2**0.5
    COVER_GATE = 0.8  # aoi whose area is covered beyond this por
    i, aoi = i_aoi
    aoi["has_parent"] = False
    aoi["parent"] = -1
    x, y = aoi["point"][:2]
    geo = aoi["geo"]
    area = aoi["area"]
    if not aoi["valid"]:
        return aoi
    for j, aoi2 in enumerate(aois_to_merge):
        if j != i and aoi["grid_idx"] == aoi2["grid_idx"]:
            if (
                aoi2["area"] > area and aoi2["valid"]
            ):  # Avoid two aoi whose overlap ratio exceeds their respective area thresholds from including each other.
                x2, y2 = aoi2["point"][:2]  # Search only between adjacent aoi
                # If the large aoi contains a small aoi, the distance between two points in the two aoi cannot exceed half of the circumference of the large aoi
                # Use 1 norm for the distance and divide it by sqrt(2) to estimate the lower bound and reduce the amount of calculation.
                if SQRT2 * (abs(x - x2) + abs(y - y2)) < aoi2["length"]:
                    if (
                        geo.intersection(aoi2["geo"]).area > COVER_GATE * area
                    ):  # Overlap ratio
                        aoi["has_parent"] = True
                        aoi["parent"] = j
                        break
    return aoi


def _find_aoi_overlap_unit(i_aoi):
    """
    Find out where aoi overlap
    """
    global aois_with_overlap
    SQRT2 = 2**0.5
    i, aoi = i_aoi
    aoi["overlaps"] = []
    x, y = aoi["point"][:2]
    geo = aoi["geo"]
    length = aoi["length"]
    area = aoi["area"]
    if not aoi["valid"]:
        return aoi
    for j, aoi2 in enumerate(aois_with_overlap):
        if j != i and aoi["grid_idx"] == aoi2["grid_idx"]:
            if (
                aoi2["area"] > area and aoi2["valid"]
            ):  # Avoid two aoi whose overlap ratio exceeds their respective area thresholds from including each other.
                x2, y2 = aoi2["point"][:2]
                if SQRT2 * (abs(x - x2) + abs(y - y2)) < 2 * (
                    length + aoi2["length"]
                ):  # Take nearby aoi to reduce the amount of calculation
                    if geo.intersects(aoi2["geo"]):
                        aoi["overlaps"].append(j)
    return aoi


def _merge_point_aoi(aois_pois, workers):
    """Merge point aoi into poly aoi"""
    logging.info("Merging Point AOI")
    # There are many objects that are very close to each other in poi. Use union search and KD_Tree to merge them.
    MERGE_GATE = 50
    NUM_BLOCK = 4  # Number of blocks processed by the partition
    res_poly = []
    res_point = []
    if len(aois_pois) < 1:
        return (res_poly, res_point)
    aois_point_xys = [[p["geo"].x, p["geo"].y] for p in aois_pois]
    pois_x, pois_y = [*zip(*aois_point_xys)]
    x_min, y_min = np.min(pois_x), np.min(pois_y)
    x_max, y_max = np.max(pois_x), np.max(pois_y)
    x_step = (x_max - x_min) / NUM_BLOCK
    y_step = (y_max - y_min) / NUM_BLOCK
    for index_x in range(NUM_BLOCK):
        for index_y in range(NUM_BLOCK):
            block_poi_xys = []  # All aoi point xy coordinates of this block
            block_poi = []
            for p in aois_pois:
                x, y = p["geo"].x, p["geo"].y
                if (
                    x_min + index_x * x_step < x < x_min + (index_x + 1) * x_step
                    and y_min + index_y * y_step < y < y_min + (index_y + 1) * y_step
                ):
                    block_poi_xys.append([x, y])
                    block_poi.append(p)
            if len(block_poi) < 1:
                continue
            father_id = {
                i: i for i, _ in enumerate(block_poi)
            }  # Initialize the parent node id of each poi
            tree = KDTree(block_poi_xys)
            visited_pid = set()

            def find_neighbor(pid):
                visited_pid.add(pid)
                p_xy = block_poi_xys[pid]
                # KDTree Find nearby poi
                for near_pid in tree.query_ball_point(p_xy, MERGE_GATE):  # type: ignore
                    if near_pid in visited_pid:
                        continue
                    father_id[near_pid] = pid
                    find_neighbor(near_pid)

            for pid, _ in enumerate(block_poi):
                if pid in visited_pid:
                    continue
                find_neighbor(pid)

            for pid in range(len(block_poi)):
                while father_id[pid] != father_id[father_id[pid]]:
                    father_id[pid] = father_id[father_id[pid]]
            # poi is merged into the parent node
            merged_poi = defaultdict(list)
            for pid, p in enumerate(block_poi):
                merged_poi[father_id[pid]].append(p)
            with Pool(processes=workers) as pool:
                post_res = pool.map(
                    _split_merged_poi_unit,
                    list(merged_poi.values()),
                    chunksize=max(
                        min(ceil(len(list(merged_poi.values())) / workers), 500), 1
                    ),
                )
            for res_aoi, res_poi in post_res:
                res_poly += res_aoi
                res_point += res_poi

    return (res_poly, res_point)


def _process_matched_result(aoi, d_matched, w_matched):
    """
    Input matched: [(lane_id, lane_s, (x_gate, y_gate), (x_lane_proj, y_lane_proj), distance)]
    Generate output_aoi according to output format
    """
    d_positions, d_gates, d_externals, w_positions, w_gates, w_externals = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for matched, positions, gates, externals in (
        (d_matched, d_positions, d_gates, d_externals),
        (w_matched, w_positions, w_gates, w_externals),
    ):
        for lane_id, s, (x_gate, y_gate), (x, y), dis in matched:
            positions.append({"lane_id": lane_id, "s": s})
            gates.append({"x": x_gate, "y": y_gate})
            externals.append((dis, {"x": x, "y": y}))
    aoi["driving_positions"] = (
        d_positions  # Even if there is none, create an empty list
    )
    aoi["driving_gates"] = d_gates
    if d_positions:
        aoi["external"]["driving_distances"] = [x[0] for x in d_externals]
        aoi["external"]["driving_lane_project_point"] = [x[1] for x in d_externals]
    aoi["walking_positions"] = (
        w_positions  # Even if there is none, create an empty list
    )
    aoi["walking_gates"] = w_gates
    if w_positions:
        aoi["external"]["walking_distances"] = [x[0] for x in w_externals]
        aoi["external"]["walking_lane_project_point"] = [x[1] for x in w_externals]
    aoi["subline_ids"] = []
    return aoi


# def _matcher_unit(
#     geo,
#     matcher,
#     dis_gate,
#     huge_gate,
#     direction_geos: Optional[Dict[int, List[LineString]]] = None,
# ):
#     global LENGTH_PER_DOOR, MAX_DOOR_NUM, AOI_GATE_OFFSET
#     double_dis_gate = 2 * dis_gate
#     double_huge_gate = 2 * huge_gate
#     stop_lane_angles = (
#         [[get_line_angle(l) for l in lanes] for lanes in direction_geos.values()]
#         if direction_geos is not None
#         else None
#     )

#     matched = []
#     huge_candidate = []
#     (x, y) = geo.centroid.coords[:][0][:2]
#     length = geo.length
#     bound_poss = [c[:2] for c in geo_coords(geo)[:-1]]
#     for lane in matcher:  # {'id', 'geo', 'point', 'length'}
#         mid_x, mid_y = lane["point"][:2]
#         dis_upper_bound = (
#             np.sqrt(2) * (abs(x - mid_x) + abs(y - mid_y)) - length - lane["length"]
#         )
#         if dis_upper_bound < double_huge_gate:
#             huge_candidate.append(lane)
#             if dis_upper_bound < double_dis_gate:
#                 # p_aoi: The point on the polygon closest to the lane
#                 # p_lane: The point closest to the polygon on the lane
#                 p_aoi, p_lane = ops.nearest_points(
#                     geo, lane["geo"]
#                 )  # Returns the calculated nearest points in the input geometries
#                 distance = p_aoi.distance(p_lane)
#                 lane_angle = get_line_angle(lane["geo"])
#                 if distance < dis_gate and (
#                     stop_lane_angles is None
#                     or any(
#                         np.mean(
#                             [abs_delta_angle(lane_angle, angle) for angle in angles]
#                         )
#                         < np.pi / 4
#                         for angles in stop_lane_angles
#                     )
#                 ):
#                     # Project the point on the lane closest to the poly to the lane and return s
#                     s = lane["geo"].project(p_lane)
#                     if (
#                         0 < s < lane["length"]
#                     ):  # Remove results matching the starting point or end point
#                         # If the door is on the poly vertex, move it appropriately so that it is on the edge
#                         p_aoi_pos = (p_aoi.x, p_aoi.y)
#                         for i, p in enumerate(bound_poss):
#                             if p == p_aoi_pos:
#                                 l_suc = LineString(
#                                     [p, bound_poss[(i + 1) % len(bound_poss)]]
#                                 )
#                                 l_pre = LineString(
#                                     [p, bound_poss[(i - 1) % len(bound_poss)]]
#                                 )
#                                 p_aoi_suc = l_suc.interpolate(
#                                     min(AOI_GATE_OFFSET, l_suc.length / 2)
#                                 )  # Move a certain distance clockwise/counterclockwise respectively, but not more than half the side length
#                                 p_aoi_pre = l_pre.interpolate(
#                                     min(AOI_GATE_OFFSET, l_pre.length / 2)
#                                 )
#                                 dis_suc, dis_pre = p_aoi_suc.distance(
#                                     lane["geo"]
#                                 ), p_aoi_pre.distance(lane["geo"])
#                                 p_aoi, distance = (
#                                     (p_aoi_suc, dis_suc)
#                                     if dis_suc < dis_pre
#                                     else (p_aoi_pre, dis_pre)
#                                 )  # Select the point closest to the lane after movement
#                                 s = lane["geo"].project(p_aoi)
#                                 p_lane = Point(lane["geo"].interpolate(s))
#                                 break
#                         # Avoid matching to the start or end point.
#                         if s < 1:
#                             s = min(1, lane["length"] / 2)
#                         if s > lane["length"] - 1:
#                             s = max(lane["length"] - 1, lane["length"] / 2)

#                         matched.append(
#                             (
#                                 lane["id"],
#                                 s,
#                                 (p_aoi.x, p_aoi.y),
#                                 (p_lane.x, p_lane.y),
#                                 distance,
#                             )
#                         )
#     if not matched:
#         """
#         No lane can match aoi, indicating that aoi may be located in the center of the road grid
#         At this time, select the nearest lanes, relax the distance threshold, and match
#         The number of lanes (number of gates in an aoi) depends on the size of the aoi: proportional to the perimeter, not proportional to the area
#         (Imagine a square requires 4 gates. When the side length is doubled, it should require 8 gates instead of 16 gates)
#         (Since the perimeter of a very flat building is deceptive, it should be converted into the perimeter of a square with the same area)
#         """
#         huge_match = []
#         for lane in huge_candidate:
#             p_aoi, p_lane = ops.nearest_points(geo, lane["geo"])
#             dis = p_aoi.distance(p_lane)
#             if dis < huge_gate:
#                 huge_match.append((dis, p_aoi, p_lane, lane))
#         huge_match.sort(key=lambda x: x[0])  # Sort by dis from small to large
#         door_num = min(1 + np.sqrt(geo.area) // LENGTH_PER_DOOR, MAX_DOOR_NUM)
#         for dis, p_aoi, p_lane, lane in huge_match:
#             s = lane["geo"].project(p_lane)
#             if 0 < s < lane["length"]:
#                 # If the door is on the poly vertex, move it appropriately so that it is on the edge
#                 p_aoi_pos = (p_aoi.x, p_aoi.y)
#                 for i, p in enumerate(bound_poss):
#                     if p == p_aoi_pos:
#                         l_suc = LineString([p, bound_poss[(i + 1) % len(bound_poss)]])
#                         l_pre = LineString([p, bound_poss[(i - 1) % len(bound_poss)]])
#                         p_aoi_suc = l_suc.interpolate(
#                             min(AOI_GATE_OFFSET, l_suc.length / 2)
#                         )  # Move a certain distance clockwise/counterclockwise respectively, but not more than half the side length
#                         p_aoi_pre = l_pre.interpolate(
#                             min(AOI_GATE_OFFSET, l_pre.length / 2)
#                         )
#                         dis_suc, dis_pre = p_aoi_suc.distance(
#                             lane["geo"]
#                         ), p_aoi_pre.distance(lane["geo"])
#                         p_aoi, dis = (
#                             (p_aoi_suc, dis_suc)
#                             if dis_suc < dis_pre
#                             else (p_aoi_pre, dis_pre)
#                         )  # Select the point closest to the lane after movement
#                         s = lane["geo"].project(p_aoi)
#                         p_lane = Point(lane["geo"].interpolate(s))
#                         break

#                 if s < 1:
#                     s = min(1, lane["length"] / 2)
#                 if s > lane["length"] - 1:
#                     s = max(lane["length"] - 1, lane["length"] / 2)

#                 matched.append(
#                     (
#                         lane["id"],
#                         s,
#                         (p_aoi.x, p_aoi.y),
#                         (p_lane.x, p_lane.y),
#                         dis,
#                     )
#                 )
#                 if len(matched) == door_num:
#                     break
#     return matched


def _str_tree_matcher_unit(
    geo,
    matcher,
    matcher_lane_tree,
    dis_gate,
    huge_gate,
    direction_geos: Optional[Dict[int, List[LineString]]] = None,
):
    global LENGTH_PER_DOOR, MAX_DOOR_NUM, AOI_GATE_OFFSET
    matched = []
    bound_poss = [c[:2] for c in geo_coords(geo)[:-1]]
    small_tree_ids = matcher_lane_tree.query(geo.buffer(dis_gate))
    stop_lane_angles = (
        [[get_line_angle(l) for l in lanes] for lanes in direction_geos.values()]
        if direction_geos is not None
        else None
    )
    for tid in small_tree_ids:
        lane = matcher[tid]  # {'id', 'geo', 'point', 'length'}
        p_aoi, p_lane = ops.nearest_points(
            geo, lane["geo"]
        )  # Returns the calculated nearest points in the input geometries
        distance = p_aoi.distance(p_lane)
        lane_angle = get_line_angle(lane["geo"])
        if stop_lane_angles is None or any(
            np.mean([abs_delta_angle(lane_angle, angle) for angle in angles])
            < np.pi / 4
            for angles in stop_lane_angles
        ):
            # Project the point on the lane closest to the poly to the lane and return s
            s = lane["geo"].project(p_lane)
            if (
                0 < s < lane["length"]
            ):  # Remove results matching the starting point or end point
                # If the door is on the poly vertex, move it appropriately so that it is on the edge
                p_aoi_pos = (p_aoi.x, p_aoi.y)
                for i, p in enumerate(bound_poss):
                    if p == p_aoi_pos:
                        l_suc = LineString([p, bound_poss[(i + 1) % len(bound_poss)]])
                        l_pre = LineString([p, bound_poss[(i - 1) % len(bound_poss)]])
                        p_aoi_suc = l_suc.interpolate(
                            min(AOI_GATE_OFFSET, l_suc.length / 2)
                        )  # Move a certain distance clockwise/counterclockwise respectively, but not more than half the side length
                        p_aoi_pre = l_pre.interpolate(
                            min(AOI_GATE_OFFSET, l_pre.length / 2)
                        )
                        dis_suc, dis_pre = p_aoi_suc.distance(
                            lane["geo"]
                        ), p_aoi_pre.distance(lane["geo"])
                        p_aoi, distance = (
                            (p_aoi_suc, dis_suc)
                            if dis_suc < dis_pre
                            else (p_aoi_pre, dis_pre)
                        )  # Select the point closest to the lane after movement
                        s = lane["geo"].project(p_aoi)
                        p_lane = Point(lane["geo"].interpolate(s))
                        break
                # Avoid matching to the start or end point.
                if s < 1:
                    s = min(1, lane["length"] / 2)
                if s > lane["length"] - 1:
                    s = max(lane["length"] - 1, lane["length"] / 2)

                matched.append(
                    (
                        lane["id"],
                        s,
                        (p_aoi.x, p_aoi.y),
                        (p_lane.x, p_lane.y),
                        distance,
                    )
                )
    if not matched:
        """
        No lane can match aoi, indicating that aoi may be located in the center of the road grid
        At this time, select the nearest lanes, relax the distance threshold, and match
        The number of lanes (number of gates in an aoi) depends on the size of the aoi: proportional to the perimeter, not proportional to the area
        (Imagine a square requires 4 gates. When the side length is doubled, it should require 8 gates instead of 16 gates)
        (Since the perimeter of a very flat building is deceptive, it should be converted into the perimeter of a square with the same area)
        """
        huge_tree_ids = matcher_lane_tree.query(geo.buffer(huge_gate))
        huge_match = []
        for tid in huge_tree_ids:
            lane = matcher[tid]
            p_aoi, p_lane = ops.nearest_points(geo, lane["geo"])
            dis = p_aoi.distance(p_lane)
            # if dis < huge_gate:
            if True:
                huge_match.append((dis, p_aoi, p_lane, lane))
        huge_match.sort(key=lambda x: x[0])  # Sort by dis from small to large
        door_num = min(1 + np.sqrt(geo.area) // LENGTH_PER_DOOR, MAX_DOOR_NUM)
        for dis, p_aoi, p_lane, lane in huge_match:
            s = lane["geo"].project(p_lane)
            if 0 < s < lane["length"]:
                # If the door is on the poly vertex, move it appropriately so that it is on the edge
                p_aoi_pos = (p_aoi.x, p_aoi.y)
                for i, p in enumerate(bound_poss):
                    if p == p_aoi_pos:
                        l_suc = LineString([p, bound_poss[(i + 1) % len(bound_poss)]])
                        l_pre = LineString([p, bound_poss[(i - 1) % len(bound_poss)]])
                        p_aoi_suc = l_suc.interpolate(
                            min(AOI_GATE_OFFSET, l_suc.length / 2)
                        )  # Move a certain distance clockwise/counterclockwise respectively, but not more than half the side length
                        p_aoi_pre = l_pre.interpolate(
                            min(AOI_GATE_OFFSET, l_pre.length / 2)
                        )
                        dis_suc, dis_pre = p_aoi_suc.distance(
                            lane["geo"]
                        ), p_aoi_pre.distance(lane["geo"])
                        p_aoi, dis = (
                            (p_aoi_suc, dis_suc)
                            if dis_suc < dis_pre
                            else (p_aoi_pre, dis_pre)
                        )  # Select the point closest to the lane after movement
                        s = lane["geo"].project(p_aoi)
                        p_lane = Point(lane["geo"].interpolate(s))
                        break

                if s < 1:
                    s = min(1, lane["length"] / 2)
                if s > lane["length"] - 1:
                    s = max(lane["length"] - 1, lane["length"] / 2)

                matched.append(
                    (
                        lane["id"],
                        s,
                        (p_aoi.x, p_aoi.y),
                        (p_lane.x, p_lane.y),
                        dis,
                    )
                )
                if len(matched) == door_num:
                    break
    return matched


def _add_point_aoi_unit(arg):
    """
    Matching of single point aoi and lane
    First, preliminary filtering is based on the distance between the point and the lane center, the perimeter of the two and the matching distance threshold.
    Then do the projection lane.project(point), which is the matching result
    """
    global d_matcher, w_matcher
    global D_DIS_GATE, D_HUGE_GATE
    global W_DIS_GATE, W_HUGE_GATE
    aoi, aoi_type = arg
    geo = aoi["geo"]
    x, y = geo_coords(geo)[0][:2]
    d_matched, w_matched = [], []
    for (
        matcher,
        matched,
        dis_gate,
        huge_gate,
    ) in (
        (
            d_matcher,
            d_matched,
            D_DIS_GATE,
            D_HUGE_GATE,
        ),
        (
            w_matcher,
            w_matched,
            W_DIS_GATE,
            W_HUGE_GATE,
        ),
    ):
        double_dis_gate = 2 * dis_gate
        double_huge_gate = 2 * huge_gate
        huge_candidate = []
        for lane in matcher:  # {'id', 'geo', 'point', 'length'}
            (mid_x, mid_y) = lane["point"][:2]  # lane midpoint
            dis_upper_bound = (
                np.sqrt(2) * (abs(x - mid_x) + abs(y - mid_y)) - lane["length"]
            )
            if dis_upper_bound < double_huge_gate:
                huge_candidate.append(lane)
                if dis_upper_bound < double_dis_gate:
                    s = lane["geo"].project(geo)
                    if (
                        0 < s < lane["length"]
                    ):  # Remove results matching the starting point or end point
                        p_lane = Point(lane["geo"].interpolate(s))
                        distance = geo.distance(p_lane)
                        if distance < dis_gate:
                            if s < 1:
                                s = min(1, lane["length"] / 2)
                            if s > lane["length"] - 1:
                                s = max(lane["length"] - 1, lane["length"] / 2)
                            matched.append(
                                (lane["id"], s, (x, y), (p_lane.x, p_lane.y), distance)
                            )
        if not matched:
            """
            No lane can match aoi, indicating that aoi may be located in the center of the road grid
            At this time, the distance threshold is relaxed and the nearest lane is matched.
            """
            huge_res = []
            for lane in huge_candidate:
                s = lane["geo"].project(geo)
                if 0 < s < lane["length"]:
                    p_lane = Point(lane["geo"].interpolate(s))
                    distance = geo.distance(p_lane)
                    if distance < huge_gate:
                        if s < 1:
                            s = min(1, lane["length"] / 2)
                        if s > lane["length"] - 1:
                            s = max(lane["length"] - 1, lane["length"] / 2)
                        huge_res.append(
                            (lane["id"], s, (x, y), (p_lane.x, p_lane.y), distance)
                        )
            if huge_res:
                matched.append(min(huge_res, key=lambda x: x[-1]))

    if d_matched or w_matched:
        # if aoi_type == AoiType.AOI_TYPE_BUS_STATION:
        if not (
            d_matched and w_matched
        ):  # Make sure the bus stop has both walking_position and driving_position
            return
        # busstop_id in fudan.bus_stops
        external = {
            "busstop_id": aoi["id"],
            "land_types": {
                "busstop": 0.0,
            },
            "names": {
                "busstop": 0.0,
            },
        }
        d_matched = [
            min(d_matched, key=lambda x: x[-1])
        ]  # Ensure that the bus stop is only matched to the nearest lane
        w_matched = [min(w_matched, key=lambda x: x[-1])]

        external["population"] = (
            aoi["external"].get("population", 0) if "external" in aoi else 0
        )
        base_aoi = {
            "id": 0,  # It is difficult to deal with the problem of uid += 1 during parallelization, so assign the id after parallelization is completed.
            "type": aoi_type,
            "positions": [{"x": x, "y": y}],
            "external": external,
        }
        return _process_matched_result(
            aoi=base_aoi, d_matched=d_matched, w_matched=w_matched
        )


def _add_poly_aoi_unit(arg):
    """
    Matching of polygon aoi and lane
    First, preliminary filtering is based on the distance between the center of the polygon centroid and the lane, the perimeter of the two and the matching distance threshold.
    Then use shapely.ops.nearest_points to find the two points closest to each other, which is the matching result.
    """
    global d_matcher, w_matcher
    global d_tree, w_tree
    global D_DIS_GATE, D_HUGE_GATE
    global W_DIS_GATE, W_HUGE_GATE
    aoi, aoi_type = arg
    geo = aoi["geo"]
    d_matched = _str_tree_matcher_unit(geo, d_matcher, d_tree, D_DIS_GATE, D_HUGE_GATE)
    w_matched = _str_tree_matcher_unit(geo, w_matcher, w_tree, W_DIS_GATE, W_HUGE_GATE)
    if d_matched or w_matched:
        base_aoi = {
            "id": 0,  # It is difficult to deal with the problem of uid += 1 during parallelization, so assign the id after parallelization is completed.
            "type": aoi_type,
            "positions": [{"x": c[0], "y": c[1]} for c in geo_coords(geo)],
            "area": geo.area,
            "external": {
                "osm_tencent_ids": [
                    aoi["id"]
                ],  # id in aoi_osm_tencent_fudan.aoi_beijing5ring
                "ex_poi_ids": aoi["external"].get(
                    "inner_poi", []
                ),  # Tencent id of poi contained in aoi
                # aoi population
                "population": aoi["external"].get("population", 0),
                "land_types": aoi["external"].get("land_types", defaultdict(float)),
                "names": aoi["external"].get("names", defaultdict(float)),
            },
        }
        return _process_matched_result(
            aoi=base_aoi, d_matched=d_matched, w_matched=w_matched
        )


def _add_aoi_stop_unit(arg):
    """
    Matching of station aoi and lane
    """
    global d_matcher, w_matcher
    global d_tree, w_tree
    global D_DIS_GATE, D_HUGE_GATE
    global STOP_DIS_GATE, STOP_HUGE_GATE
    global W_DIS_GATE, W_HUGE_GATE
    global LENGTH_PER_DOOR, MAX_DOOR_NUM, AOI_GATE_OFFSET
    aoi, aoi_type = arg
    geo = aoi["geo"]
    bound_poss = [c[:2] for c in geo_coords(geo)[:-1]]
    station_type = aoi["external"]["station_type"]
    d_matched, w_matched = [], []
    if station_type == "SUBWAY":
        matcher = aoi["external"]["subline_lane_matcher"]
        d_matched = []
        for lane in matcher:  # {'id', 'geo', 'point', 'length'}
            # p_aoi: The point on the polygon closest to the lane
            # p_lane: The point closest to the polygon on the lane
            p_aoi, p_lane = ops.nearest_points(
                geo, lane["geo"]
            )  # Returns the calculated nearest points in the input geometries
            distance = p_aoi.distance(p_lane)
            # Project the point on the lane closest to the poly to the lane and return s
            s = lane["geo"].project(p_lane)
            if (
                0 < s < lane["length"]
            ):  # Remove results matching the starting point or end point
                # If the door is on the poly vertex, move it appropriately so that it is on the edge
                p_aoi_pos = (p_aoi.x, p_aoi.y)
                for i, p in enumerate(bound_poss):
                    if p == p_aoi_pos:
                        l_suc = LineString([p, bound_poss[(i + 1) % len(bound_poss)]])
                        l_pre = LineString([p, bound_poss[(i - 1) % len(bound_poss)]])
                        p_aoi_suc = l_suc.interpolate(
                            min(AOI_GATE_OFFSET, l_suc.length / 2)
                        )  # Move a certain distance clockwise/counterclockwise respectively, but not more than half the side length
                        p_aoi_pre = l_pre.interpolate(
                            min(AOI_GATE_OFFSET, l_pre.length / 2)
                        )
                        dis_suc, dis_pre = p_aoi_suc.distance(
                            lane["geo"]
                        ), p_aoi_pre.distance(lane["geo"])
                        p_aoi, distance = (
                            (p_aoi_suc, dis_suc)
                            if dis_suc < dis_pre
                            else (p_aoi_pre, dis_pre)
                        )  # Select the point closest to the lane after movement
                        s = lane["geo"].project(p_aoi)
                        p_lane = Point(lane["geo"].interpolate(s))
                        break
            # Avoid matching to the start or end point.
            if s < 1:
                s = min(1, lane["length"] / 2)
            if s > lane["length"] - 1:
                s = max(lane["length"] - 1, lane["length"] / 2)
            d_matched.append(
                (
                    lane["id"],
                    s,
                    (p_aoi.x, p_aoi.y),
                    (p_lane.x, p_lane.y),
                    distance,
                )
            )
        w_matched = _str_tree_matcher_unit(
            geo, w_matcher, w_tree, W_DIS_GATE, W_HUGE_GATE
        )
    elif station_type == "BUS":
        d_matched = _str_tree_matcher_unit(
            geo,
            d_matcher,
            d_tree,
            STOP_DIS_GATE,
            STOP_HUGE_GATE,
            aoi["external"]["subline_geos"],
        )
        w_matched = _str_tree_matcher_unit(
            geo, w_matcher, w_tree, W_DIS_GATE, W_HUGE_GATE
        )
    if d_matched and w_matched:  # The station must connect both roadways and sidewalks
        base_aoi = {
            "id": 0,  # It is difficult to deal with the problem of uid += 1 during parallelization, so assign the id after parallelization is completed.
            "type": aoi_type,
            "positions": [{"x": c[0], "y": c[1]} for c in geo_coords(geo)],
            "area": geo.area,
            "external": {
                "osm_tencent_ids": [
                    aoi["id"]
                ],  # id in aoi_osm_tencent_fudan.aoi_beijing5ring
                "stop_id": aoi["external"]["stop_id"],
                "station_type": station_type,
                "ex_poi_ids": aoi["external"].get(
                    "inner_poi", []
                ),  # Tencent id of poi contained in aoi
                # aoi population
                "population": aoi["external"].get("population", 0),
                "land_types": aoi["external"].get("land_types", defaultdict(float)),
                "names": aoi["external"].get("names", defaultdict(float)),
            },
        }
        return _process_matched_result(
            aoi=base_aoi, d_matched=d_matched, w_matched=w_matched
        )


def _add_aoi_land_use(aois, shp_path: Optional[str], bbox, projstr):
    """
    AOI添加用地类型
    直接加到最后地图的AOI里面去
    """
    logging.info("Adding land_use to aois")

    def map_land_use(euluc: int) -> int:
        # EULUC类型
        # First-level category Second-level category Description
        # 01 Residential land   0101 Residential land Land mainly used for housing bases and ancillary facilities for people's living
        # 02 Commercial Land    0201 Commercial Office Buildings where people work, including office buildings, commerce, economy, IT, e-commerce, media, etc.
        #   0202 Commercial services Land for commercial retail, catering, accommodation and entertainment.
        # 03 Industrial Land    0301 Industrial Land Land and construction land used for production, warehousing, mining, etc.
        # 04 Transportation Land    0401 Road Paved roads include highways, urban roads, etc.
        #   0402 Transportation terminal Transportation facilities include logistics, buses, train stations and ancillary facilities, etc.
        #   0403 Airport land Airport land used for civil, military or mixed use
        # 05 Public management and public service land 0501 Land used by government agencies and organizations Land used by party and government agencies, the military, public service agencies and organizations, etc.
        #   0502 Land for education and scientific research Land for education and scientific research, including universities, primary and secondary schools, research institutes and ancillary facilities, etc.
        #   0503 Medical land Hospital, disease control and emergency services land
        #   0504 Sports culture Mass sports and training, cultural service land, including sports centers, libraries, museums and exhibition centers, etc.
        #   0505 Parks and green spaces Parks and green spaces used for recreation or environmental protection

        # target type
        # # Land use type, refer to the national standard GB/T 21010-2007
        # # http://www.gscloud.cn/static/cases/%E3%80%8A%E5%9C%9F%E5%9C%B0%E5%88%A9%E7%94%A8%E7%8E%B0%E7%8A%B6%E5%88%86%E7%B1%BB%E3%80%8B%E5%9B%BD%E5%AE%B6%E6%A0%87%E5%87%86gb_t21010-2007(1).pdf
        # enum LandUseType {
        #     // unspecified
        #     LAND_USE_TYPE_UNSPECIFIED = 0;
        #     // commercial land
        #     LAND_USE_TYPE_COMMERCIAL = 5;
        #     // Industrial and storage land
        #     LAND_USE_TYPE_INDUSTRIAL = 6;
        #     // residential land
        #     LAND_USE_TYPE_RESIDENTIAL = 7;
        #     // Public management and public service land
        #     LAND_USE_TYPE_PUBLIC = 8;
        #     // transportation land
        #     LAND_USE_TYPE_TRANSPORTATION = 10;
        #     // other land
        #     LAND_USE_TYPE_OTHER = 12;
        # }

        mapper = {
            1: 7,
            2: 5,
            3: 6,
            4: 10,
            5: 8,
            6: 12,
        }
        return mapper.get(euluc, 12)

    if not shp_path:
        for _, aoi in aois.items():
            aoi["land_use"] = map_land_use(-1)
        return aois
    df = gpd.read_file(
        shp_path,
        bbox=bbox,
    )
    # Convert the coordinates of the geometry in df
    df["geometry"] = df["geometry"].to_crs(projstr)
    # Fix possible invalid polygons
    for i, polygon in enumerate(df.geometry):
        if not polygon.is_valid:
            logging.warning(f"Shapefile indice {i} is invalid polygon")
            geo = polygon.buffer(0)
            if isinstance(geo, Polygon) and geo.is_valid:
                polygon = geo
            elif isinstance(geo, MultiPolygon):
                candidate_poly = None
                for poly in geo.geoms:
                    if poly.is_valid:
                        candidate_poly = poly
                        break
                if candidate_poly is not None:
                    polygon = candidate_poly
                else:
                    polygon = MultiPoint(
                        [pt for g in geo.geoms for pt in geo_coords(g)]
                    ).convex_hull
            else:
                polygon = MultiPoint([pt for pt in geo_coords(polygon)]).convex_hull
            df.geometry[i] = polygon

    # Processing AOIs
    aois_shapely = {}
    for aoi_id, aoi in aois.items():
        if "area" not in aoi:
            # Non-polygonal AOI (POI) exclusion
            continue
        # "positions": [{'x': 447310.67707190284, 'y': 4417286.262865324},
        # {'x': 447273.4744241001, 'y': 4417283.789243901},
        # {'x': 447251.27340908017, 'y': 4417276.250666907},
        # {'x': 447232.451260092, 'y': 4417266.568797507},
        # {'x': 447208.3071892054, 'y': 4417250.019894581},
        polygon = Polygon([(pos["x"], pos["y"]) for pos in aoi["positions"]])
        if not polygon.is_valid:  #  invalid polygon
            logging.warning(f"Aoi {aoi_id} is invalid polygon")
            # fix invalid polygon
            geo = polygon.buffer(0)
            if isinstance(geo, Polygon) and geo.is_valid:
                polygon = geo
            elif isinstance(geo, MultiPolygon):
                candidate_poly = None
                for poly in geo.geoms:
                    if poly.is_valid:
                        candidate_poly = poly
                        break
                if candidate_poly is not None:
                    polygon = candidate_poly
                else:
                    polygon = MultiPoint(
                        [pt for g in geo.geoms for pt in geo_coords(g)]
                    ).convex_hull
            else:
                aoi["land_use"] = map_land_use(-1)  # 默认
                continue
        assert isinstance(polygon, Polygon)
        aois_shapely[aoi_id] = polygon

    # Match AOI data and land type data
    for aoi_id, geo in aois_shapely.items():
        indices = df.sindex.query(geo, predicate="intersects")
        if len(indices) == 0:
            # Unable to match land type
            aois[aoi_id]["land_use"] = map_land_use(-1)
            continue
        sub_df = df.loc[indices]
        areas = sub_df.intersection(geo).area
        # join area + Level1
        # Sum the area according to different Level1
        # Select Level1 with the largest area
        area_level1 = areas.groupby(sub_df["Level1"]).sum()
        max_area_level1 = area_level1.idxmax()
        land_use = map_land_use(max_area_level1)
        aois[aoi_id]["land_use"] = land_use
    return aois


def _add_aoi_urban_land_use(aois):
    logging.info("Adding urban_land_use to aois")

    def get_urban_land_use(aoi: dict) -> str:
        """Obtain land type based on original OSM information"""
        # Get the original OSM land type with the largest current AOI area
        land_types = (
            aoi["external"]["land_types"]
            if "external" in aoi and "land_types" in aoi["external"]
            else {}
        )
        main_land_type = (
            max(
                [(land_type, area) for land_type, area in land_types.items()],
                key=lambda x: x[1],
            )[0]
            if land_types
            else None
        )
        # Target type
        # Reference standard GB 50137-2011 Classification and Code of Urban Construction Land (https://www.planning.org.cn/law/uploads/2013/1383993139.pdf)
        if main_land_type is not None:
            # https://wiki.openstreetmap.org/wiki/Key:leisure
            if main_land_type.startswith("leisure"):
                leisure_type = main_land_type.split("|")[-1]
                # Leisure
                if leisure_type in [
                    "adult_gaming_centre",
                    "amusement_arcade",
                    "bandstand",
                    "bathing_place",
                    "dance",
                    "escape_game",
                    "fishing",
                    "garden",
                    "hackerspace",
                    "horse_riding",
                    "marina",
                    "sauna",
                    "track",
                    "trampoline_park",
                ]:
                    # 娱乐用地
                    return "B31"
                elif leisure_type in [
                    "beach_resort",
                    "bird_hide",
                    "bleachers",
                    "bowling_alley",
                    "disc_golf_course",
                    "pitch",
                    "playground",
                    "resort",
                    "water_park",
                    "golf_course",
                    "miniature_golf",
                    "tanning_salon",
                    "fitness_centre",
                    "fitness_station",
                ]:
                    # 康体用地
                    return "B32"
                elif leisure_type in [
                    "sports_hall",
                    "stadium",
                    "swimming_area",
                    "swimming_pool",
                    "sports_centre",
                    "slipway",
                    "ice_rink",
                ]:
                    # 体育用地
                    return "A4"
                elif leisure_type in [
                    "firepit",
                    "park",
                    "dog_park",
                    "picnic_table",
                    "summer_camp",
                    "outdoor_seating",
                ]:
                    # 公园绿地
                    return "G1"
                elif leisure_type in [
                    "common",
                ]:
                    # 公共管理与公共服务用地
                    return "A"
                elif leisure_type in ["nature_reserve", "wildlife_hide"]:
                    # 其他非建设用地
                    return "E3"
                # Other leisure
                else:
                    return "B3"
            # 详见https://wiki.openstreetmap.org/wiki/Key:amenity
            elif main_land_type.startswith("amenity"):
                amenity_type = main_land_type.split("|")[-1]
                # Sustenance
                if amenity_type in [
                    "bar",
                    "biergarten",
                    "cafe",
                    "fast_food",
                    "food_court",
                    "ice_cream",
                    "pub",
                    "restaurant",
                ]:
                    # 饭店、餐厅、酒吧等用地
                    return "B13"
                # Education
                elif amenity_type in [
                    "college",
                    "dancing_school",
                    "driving_school",
                    "first_aid_school",
                    "kindergarten",
                    "language_school",
                    "library",
                    "surf_school",
                    "toy_library",
                    "research_institute",
                    "training",
                    "music_school",
                    "school",
                    "traffic_park",
                    "university",
                ]:
                    # 教育科研用地
                    return "A3"
                # Transportation
                elif amenity_type in [
                    "bicycle_parking",
                    "bicycle_repair_station",
                    "bicycle_rental",
                    "bicycle_wash",
                    "boat_rental",
                    "boat_sharing",
                    "bus_station",
                    "car_rental",
                    "car_sharing",
                    "car_wash",
                    "compressed_air",
                    "vehicle_inspection",
                    "charging_station",
                    "driver_training",
                    "ferry_terminal",
                    "fuel",
                    "grit_bin",
                    "motorcycle_parking",
                    "parking",
                    "parking_entrance",
                    "parking_space",
                    "taxi",
                    "weighbridge",
                ]:
                    # 交通场站用地
                    return "S4"
                # Financial
                elif amenity_type in [
                    "atm",
                    "payment_terminal",
                    "bank",
                    "bureau_de_change",
                    "money_transfer",
                    "payment_centre",
                ]:
                    # 商业设施用地
                    return "B1"
                # Healthcare
                elif amenity_type in [
                    "baby_hatch",
                    "clinic",
                    "dentist",
                    "doctors",
                    "hospital",
                    "nursing_home",
                    "pharmacy",
                    "social_facility",
                    "veterinary",
                ]:
                    # 医疗卫生用地
                    return "A5"
                # Entertainment, Arts & Culture
                elif amenity_type in [
                    "arts_centre",
                    "brothel",
                    "casino",
                    "cinema",
                    "community_centre",
                    "conference_centre",
                    "events_venue",
                    "exhibition_centre",
                    "fountain",
                    "gambling",
                    "love_hotel",
                    "music_venue",
                    "nightclub",
                    "planetarium",
                    "public_bookcase",
                    "social_centre",
                    "stripclub",
                    "studio",
                    "swingerclub",
                    "theatre",
                ]:
                    # 文化设施用地
                    return "A2"
                # Public Service
                elif amenity_type in [
                    "courthouse",
                    "fire_station",
                    "police",
                    "post_box",
                    "post_depot",
                    "post_office",
                    "prison",
                    "ranger_station",
                    "townhall",
                ]:
                    if amenity_type in [
                        "fire_station",
                        "police",
                        "ranger_station",
                    ]:
                        # 安全设施用地
                        return "U4"
                    elif amenity_type in [
                        "courthouse",
                        "townhall",
                    ]:
                        # 行政办公用地
                        return "A1"
                    elif amenity_type in [
                        "prison",
                    ]:
                        # 特殊用地
                        return "H4"
                    elif amenity_type in [
                        "post_box",
                        "post_depot",
                        "post_office",
                    ]:
                        # 其他商务设施用地
                        return "B29"
                    else:
                        # 公用设施用地
                        return "U"
                # Facilities
                elif amenity_type in [
                    "bbq",
                    "bench",
                    "dog_toilet",
                    "dressing_room",
                    "drinking_water",
                    "give_box",
                    "mailroom",
                    "parcel_locker",
                    "shelter",
                    "shower",
                    "telephone",
                    "toilets",
                    "water_point",
                    "watering_place",
                ]:
                    # 公用设施用地
                    return "U"
                # Waste Management
                elif amenity_type in [
                    "sanitary_dump_station",
                    "recycling",
                    "waste_basket",
                    "waste_disposal",
                    "waste_transfer_station",
                ]:
                    # 环保设施用地
                    return "U23"
                # Other omenity
                else:
                    # 其他公用设施用地
                    return "U9"
            # https://wiki.openstreetmap.org/wiki/Key:building
            elif main_land_type.startswith("building"):
                building_type = main_land_type.split("|")[-1]
                # Accommodation
                if building_type in [
                    "apartments",
                    "barracks",
                    "bungalow",
                    "cabin",
                    "detached",
                    "dormitory",
                    "farm",
                    "ger",
                    "hotel",
                    "house",
                    "houseboat",
                    "residential",
                    "semidetached_house",
                    "static_caravan",
                    "stilt_house",
                    "terrace",
                    "tree_house",
                    "trullo",
                ]:
                    # 居住用地
                    return "R"
                # Commercial
                elif building_type in [
                    "commercial",
                    "industrial",
                    "kiosk",
                    "office",
                    "retail",
                    "supermarket",
                    "warehouse",
                ]:
                    # 商业设施用地
                    return "B1"
                # Religious
                elif building_type in [
                    "religious",
                    "cathedral",
                    "chapel",
                    "church",
                    "kingdom_hall",
                    "monastery",
                    "mosque",
                    "presbytery",
                    "shrine",
                    "synagogue",
                    "temple",
                ]:
                    # 宗教设施用地
                    return "A9"
                # Civic/amenity
                elif building_type in [
                    "bakehouse",
                    "bridge",
                    "civic",
                    "college",
                    "fire_station",
                    "government",
                    "gatehouse",
                    "hospital",
                    "kindergarten",
                    "museum",
                    "public",
                    "school",
                    "toilets",
                    "train_station",
                    "transportation",
                    "university",
                ]:
                    if building_type in [
                        "college",
                        "kindergarten",
                        "school",
                        "university",
                    ]:
                        # 教育科研用地
                        return "A3"
                    elif building_type in [
                        "bridge",
                        "train_station",
                        "transportation",
                    ]:
                        # 交通场站用地
                        return "S4"
                    elif building_type in ["government", "civic"]:
                        # 行政办公用地
                        return "A1"
                    elif building_type in [
                        "fire_station",
                    ]:
                        # 安全设施用地
                        return "U4"
                    elif building_type in ["hospital"]:
                        # 医疗卫生用地
                        return "A5"
                    else:
                        # 其他公用设施用地
                        return "U9"
                # Agricultural/plant production
                elif building_type in [
                    "barn",
                    "conservatory",
                    "cowshed",
                    "farm_auxiliary",
                    "greenhouse",
                    "slurry_tank",
                    "stable",
                    "sty",
                    "livestock",
                ]:
                    # 农林用地
                    return "E2"
                # Sports
                elif building_type in [
                    "grandstand",
                    "pavilion",
                    "riding_hall",
                    "sports_hall",
                    "sports_centre",
                    "stadium",
                ]:
                    # 体育用地
                    return "A4"
                # Storage
                elif building_type in [
                    "allotment_house",
                    "boathouse",
                    "hangar",
                    "hut",
                    "shed",
                ]:
                    # 物流仓储用地
                    return "W"
                # Cars
                elif building_type in ["carport", "garage", "garages", "parking"]:
                    # 交通场站用地
                    return "S4"
                # Power/technical buildings
                elif building_type in [
                    "digester",
                    "service",
                    "tech_cab",
                    "transformer_tower",
                    "water_tower",
                    "storage_tank",
                    "silo",
                ]:
                    # 供应设施用地
                    return "U1"
                # Other
                else:
                    # 其他公用设施用地
                    return "U9"
            # https://wiki.openstreetmap.org/wiki/Key:landuse
            else:
                # developed land
                if main_land_type in [
                    "commercial",
                    "construction",
                    "education",
                    "fairground",
                    "industrial",
                    "residential",
                    "retail",
                    "institutional",
                ]:
                    if main_land_type in [
                        "commercial",
                        "construction",
                        "retail",
                        "fairground",
                    ]:
                        # 商业服务业设施用地
                        return "B"
                    elif main_land_type in ["education"]:
                        # 教育科研用地
                        return "A3"
                    elif main_land_type in ["industrial"]:
                        # 工业用地
                        return "M"
                    elif main_land_type in ["residential"]:
                        # 居住用地
                        return "R"
                    elif main_land_type in ["institutional"]:
                        # 社会福利设施用地
                        return "C9"
                    else:
                        # 其他非建设用地
                        return "E3"
                # rural and agricultural land
                elif main_land_type in [
                    "aquaculture",
                    "allotments",
                    "farmland",
                    "farmyard",
                    "paddy",
                    "animal_keeping",
                    "flowerbed",
                    "forest",
                    "greenhouse_horticulture",
                    "meadow",
                    "orchard",
                    "plant_nursery",
                    "vineyard",
                ]:
                    if main_land_type in ["aquaculture", "farmyard", "animal_keeping"]:
                        # 农林用地
                        return "E2"
                    elif main_land_type in [
                        "allotments",
                        "farmland",
                        "paddy",
                        "flowerbed",
                        "greenhouse_horticulture",
                    ]:
                        # 农林用地
                        return "E2"
                    elif main_land_type in ["forest"]:
                        # 农林用地
                        return "E2"
                    elif main_land_type in ["meadow"]:
                        # 农林用地
                        return "E2"
                    elif main_land_type in ["orchard" "plant_nursery", "vineyard"]:
                        # 农林用地
                        return "E2"
                    else:
                        # 农林用地
                        return "E2"
                # waterbody
                elif main_land_type in ["basin", "reservoir", "salt_pond"]:
                    # 水域
                    return "E1"
                # Other
                else:
                    # 其他非建设用地
                    return "E3"
        else:
            # from `land_use`
            land_use = aoi["land_use"] if "land_use" in aoi else 12
            mapper = {
                5: "B1",
                6: "M",
                7: "R",
                8: "U",
                10: "S",
                12: "E3",
            }
            return mapper.get(land_use, "E3")

    for _, aoi in aois.items():
        aoi["urban_land_use"] = get_urban_land_use(aoi)
    return aois


def _add_aoi_name(aois):
    def get_aoi_name(aoi: dict) -> str:
        names = (
            aoi["external"]["names"]
            if "external" in aoi and "names" in aoi["external"]
            else {}
        )
        main_name = (
            max(
                [(name, area) for name, area in names.items()],
                key=lambda x: x[1],
            )[0]
            if names
            else None
        )
        if main_name:
            return main_name
        else:
            return ""

    for _, aoi in aois.items():
        aoi["name"] = get_aoi_name(aoi)
    return aois


def _add_pois(aois, pois, projstr):
    """
    add POIs
    Add poi ids information to aoi
    """
    projector = pyproj.Proj(projstr)
    # 读取poi原始数据，构建poi数据
    # {
    #   "id": "10001234807567065271",
    #   "name": "好孩子幼儿园",
    #   "catg": "241300",
    #   "gps": [
    #     116.55439111960065,
    #     39.75337111705692
    #   ]
    # }
    # ===>
    # message Poi {
    #     // Poi id (starting from 7_0000_0000)
    #     int32 id = 1;
    #     // Poi name
    #     string name = 2;
    #     // Poi category code
    #     string category = 3;
    #     // Poi original position
    #     city.geo.v2.XYPosition position = 4;
    #     // Aoi to which the Poi belongs
    #     int32 aoi_id = 5;
    #     // The capacity of Poi (the number of people it can accommodate at the same time), if none, it means there is no limit on the number of people
    #     optional int32 capacity = 6;
    #     // The functions the Poi can offer
    #     repeated string functions = 7;
    # }
    # Traverse aoi and find the corresponding poi id from external.ex_poi_ids
    poi_id_set = set()
    poi_id_to_aoi_id = {}
    for aoi in aois.values():
        for poi_id in aoi["external"].get("ex_poi_ids", []):
            poi_id_set.add(poi_id)
            assert poi_id not in poi_id_to_aoi_id
            poi_id_to_aoi_id[poi_id] = aoi["id"]
    # generate pois
    out_pois = {}
    poi_uid = POI_START_ID
    for poi_id in poi_id_set:
        poi = pois[poi_id]
        x, y = poi["coords"][0]
        external = poi["external"]
        poi_name = external.get("name", "")
        out_pois[poi_uid] = {
            "id": poi_uid,
            "name": poi_name,
            "category": poi.get("catg", ""),
            "position": {"x": x, "y": y},
            "aoi_id": poi_id_to_aoi_id[poi_id],
            "external": {
                "tencent_poi_id": poi_id,
            },
        }
        poi_uid += 1
    # add poi_ids for aois
    for aoi in aois.values():
        aoi["poi_ids"] = []
    for poi in out_pois.values():
        aoi = aois[poi["aoi_id"]]
        aoi["poi_ids"].append(poi["id"])
    return aois, out_pois


def _merge_covered_aoi(aois, workers):
    """
    Blend the contained small poly aoi into the large poly aoi
    At the same time, cut off the overlapping parts between aoi
    """
    global aois_to_merge, aois_with_overlap
    logging.info("Merging Covered Aoi")
    # Pre-compute geometric properties
    for aoi in aois:
        geo = aoi["geo"]
        aoi["point"] = geo_coords(geo.centroid)[0]  # Geometric center
        aoi["length"] = geo.length  # Perimeter
        aoi["area"] = geo.area  # area
        aoi["valid"] = geo.is_valid
        aoi["grid_idx"] = tuple(x // AOI_MERGE_GRID for x in aoi["point"])
    aois_to_merge = aois
    aois = [(i, a) for i, a in enumerate(aois)]
    aois_result = []
    for i in range(0, len(aois), MAX_BATCH_SIZE):
        aois_batch = aois[i : i + MAX_BATCH_SIZE]
        with Pool(processes=workers) as pool:
            aois_result += pool.map(
                _find_aoi_parent_unit,
                aois_batch,
                chunksize=min(ceil(len(aois_batch) / workers), 1000),
            )
    aois = aois_result
    parent2children = defaultdict(list)
    child2parent = {}
    for i, aoi in enumerate(aois):
        if aoi["has_parent"]:
            child2parent[i] = aoi["parent"]
    for child, parent in child2parent.items():
        while parent in child2parent.keys():
            parent = child2parent[parent]
        parent2children[parent].append(child)
    for parent, children in parent2children.items():
        aoi_parent = aois[parent]
        external = aoi_parent["external"]
        if "inner_poi" not in aoi_parent["external"]:
            external["inner_poi"] = []
        if "population" not in aoi_parent["external"]:
            external["population"] = 0
        for c in children:
            a = aois[c]
            external["inner_poi"] += a["external"].get("inner_poi", [])
            external["population"] += a["external"].get("population", 0)
            child_land_types = external["land_types"]
            child_names = external["names"]
            for land_type, area in child_land_types.items():
                external["land_types"][land_type] += area
            for child_name, area in child_names.items():
                external["names"][child_name] += area
    aois = [a for a in aois if not a["has_parent"]]
    aois_with_overlap = aois
    aois = [(i, a) for i, a in enumerate(aois)]
    aois_result = []
    for i in range(0, len(aois), MAX_BATCH_SIZE):
        aois_batch = aois[i : i + MAX_BATCH_SIZE]
        with Pool(processes=workers) as pool:
            aois_result += pool.map(
                _find_aoi_overlap_unit,
                aois_batch,
                chunksize=max(
                    min(ceil(len(aois_batch) / workers), 1000),
                    1,
                ),
            )
    aois = aois_result
    # get difference set of larger aoi
    has_overlap_aids = defaultdict(list)
    for i, aoi in enumerate(aois):
        for j in aoi["overlaps"]:
            has_overlap_aids[j].append(i)
    for i, aids in has_overlap_aids.items():
        aoi = aois[i]
        for j in aids:
            geo = aoi["geo"]
            overlap_geo = aois[j]["geo"]
            diff_geo_i = geo.difference(overlap_geo)
            diff_geo_j = overlap_geo.difference(geo)
            if diff_geo_i:
                if isinstance(diff_geo_i, Polygon):
                    aoi["geo"] = diff_geo_i
                    continue
                elif isinstance(diff_geo_i, MultiPolygon):
                    # AOI may be cut off, take the part with the largest area
                    candidate_geos = [(p.area, p) for p in diff_geo_i.geoms if p]
                    if candidate_geos:
                        aoi["geo"] = max(candidate_geos, key=lambda x: x[0])[1]
                        continue
            if diff_geo_j:
                if isinstance(diff_geo_j, Polygon):
                    aois[j]["geo"] = diff_geo_j
                    continue
                elif isinstance(diff_geo_j, MultiPolygon):
                    # AOI may be cut off, take the part with the largest area
                    candidate_geos = [(p.area, p) for p in diff_geo_j.geoms if p]
                    if candidate_geos:
                        aois[j]["geo"] = max(candidate_geos, key=lambda x: x[0])[1]
                        continue
    return aois


def _add_aoi(aois, stops, workers, merge_aoi: bool = False):
    """
    aois matches the rightmost lane
    """
    global aoi_uid, d_matcher, w_matcher
    global d_tree, w_tree
    # Preprocessing
    d_tree = STRtree([l["geo"] for l in d_matcher])
    w_tree = STRtree([l["geo"] for l in w_matcher])
    aois_poly, aois_poi, aois_stop = [], [], []
    logging.info("Pre Compute")
    for aoi in aois:
        if len(aoi["coords"]) > 1:
            geo = aoi["geo"]
            external = aoi["external"]
            external["land_types"] = defaultdict(
                float
            )  # All included land types and areas
            external["names"] = defaultdict(float)
            osm_tags = external["osm_tags"]
            for tags in osm_tags:
                if "landuse" in tags:
                    value = tags["landuse"]
                    # Exclude invalid fields
                    if not "yes" in value:
                        external["land_types"][value] += geo.area
                if "leisure" in tags:
                    value = tags["leisure"]
                    if not "yes" in value:
                        external["land_types"]["leisure|" + value] += geo.area
                if "amenity" in tags:
                    value = tags["amenity"]
                    if not "yes" in value:
                        external["land_types"]["amenity|" + value] += geo.area
                if "building" in tags:
                    value = tags["building"]
                    if not "yes" in value:
                        external["land_types"]["building|" + value] += geo.area
                if "name" in tags:
                    value = tags["name"]
                    external["names"][value] += geo.area
            aois_poly.append(aoi)
        else:
            x, y = aoi["coords"][0][:2]
            geo = Point(x, y)  # The essence is aoi of poi, geo is a single point
            aoi["geo"] = geo
            aois_poi.append(aoi)
    merged_aois, aois_poi = _merge_point_aoi(aois_poi, workers)
    aois_poly.extend(merged_aois)
    # Expand single-point AOI into a square
    SQUARE_SIDE_LENGTH = (
        10  # The side length of the rectangle formed by a single point POI
    )
    logging.info("Extending aois_poi to Square")
    for p in aois_poi:
        center = p["geo"]
        half_side = SQUARE_SIDE_LENGTH / 2
        bottom_left = (center.x - half_side, center.y - half_side)
        top_left = (center.x - half_side, center.y + half_side)
        top_right = (center.x + half_side, center.y + half_side)
        bottom_right = (center.x + half_side, center.y - half_side)
        p["geo"] = Polygon(
            [bottom_left, top_left, top_right, bottom_right, bottom_left]
        )
        external = p["external"]
        external["land_types"] = defaultdict(
            float
        )  # All included land types and areas. Since the single-point AOI original data has no land type, it is an empty dict.
        external["names"] = defaultdict(float)

    aois_poly.extend(aois_poi)

    if merge_aoi:
        aois_poly = _merge_covered_aoi(aois_poly, workers)
    # The convex hull may fail, check it
    for a in aois_poly:
        assert isinstance(a["geo"], Polygon)

    for stop in stops:
        aois_stop.append(stop)
    # multiprocessing
    logging.info("Multiprocessing to match aoi to lanes")
    logging.info(f"aois_poly, aois_stop:{len(aois_poly)},{len(aois_stop)}")
    logging.info(f"lanes: {len(d_matcher) + len(w_matcher)}")

    # bus stop first
    args = [(aoi, AoiType.AOI_TYPE_BUS_STATION) for aoi in aois_stop]
    results_stop = []
    for i in range(0, len(args), MAX_BATCH_SIZE):
        args_batch = args[i : i + MAX_BATCH_SIZE]
        with Pool(processes=workers) as pool:
            results_stop += pool.map(
                _add_aoi_stop_unit,
                args_batch,
                chunksize=min(ceil(len(args_batch) / workers), 500),
            )
    results_stop = [r for r in results_stop if r]
    logging.info(f"matched aois_stop: {len(results_stop)}")
    results_poly = []
    args = [(aoi, AoiType.AOI_TYPE_OTHER) for aoi in aois_poly]
    for i in range(0, len(args), MAX_BATCH_SIZE):
        args_batch = args[i : i + MAX_BATCH_SIZE]
        with Pool(processes=workers) as pool:
            results_poly += pool.map(
                _add_poly_aoi_unit,
                args_batch,
                chunksize=min(ceil(len(args_batch) / workers), 500),
            )
    results_poly = [r for r in results_poly if r]
    logging.info(f"matched aois_poly: {len(results_poly)}")

    # Post-compute
    aois = {}
    for aoi in results_poly + results_stop:
        aoi["id"] = aoi_uid
        aois[aoi_uid] = aoi
        aoi_uid += 1
    return aois


def add_aoi_to_map(
    matchers,
    input_aois: list,
    input_pois: list,
    input_stops: list,
    bbox,
    projstr: str,
    shp_path: Optional[str],
    workers: int = 32,
):
    """match AOIs to lanes"""
    global aoi_uid, d_matcher, w_matcher, road_lane_matcher
    d_matcher, w_matcher, road_lane_matcher = (
        matchers["drive"],
        matchers["walk"],
        matchers["road_lane"],
    )
    # AOI UID
    aoi_uid = AOI_START_ID
    # raw POIs
    raw_pois = {doc["id"]: doc for doc in input_pois}

    aois = _add_aoi(aois=input_aois, stops=input_stops, workers=workers, merge_aoi=True)
    added_tencent_poi = []
    for _, aoi in aois.items():
        added_tencent_poi.extend(aoi["external"].get("ex_poi_ids", []))
    logging.info(f"Added Tencent POI num: {len(added_tencent_poi)}")
    # Post-compute
    logging.info("Post Compute")
    aois = _add_aoi_land_use(aois, shp_path, bbox, projstr)
    aois = _add_aoi_name(aois)
    aois = _add_aoi_urban_land_use(aois)
    (aois, pois) = _add_pois(aois, raw_pois, projstr)
    return (aois, pois)


def add_sumo_aoi_to_map(
    matchers: dict,
    input_aois: list,
    input_pois: list,
    input_stops: list,
    projstr: str,
    merge_aoi: bool,
    workers: int = 32,
):
    """for SUMO converter, match AOI to lanes"""
    global d_matcher, w_matcher, road_lane_matcher
    d_matcher, w_matcher, road_lane_matcher = (
        matchers["drive"],
        matchers["walk"],
        matchers["road_lane"],
    )
    global D_DIS_GATE, D_HUGE_GATE, W_DIS_GATE, W_HUGE_GATE, LENGTH_PER_DOOR, MAX_DOOR_NUM, AOI_GATE_OFFSET
    global aoi_uid
    # AOI UID
    aoi_uid = AOI_START_ID
    # projection parameter
    global projector
    # raw POIs
    raw_pois = {doc["id"]: doc for doc in input_pois}
    projector = pyproj.Proj(projstr)
    if sys.platform == "win32" and (
        len(input_aois) > 0 or len(input_stops) > 0 or len(input_pois) > 0
    ):
        logging.warning("Adding aoi cannot run on Windows platform!")
        return {}, {}
    if not input_aois and not input_stops and not input_pois:
        return {}, {}

    aois = _add_aoi(
        aois=input_aois,
        stops=input_stops,
        workers=workers,
        merge_aoi=merge_aoi,
    )
    added_ex_pois = []
    for _, aoi in aois.items():
        added_ex_pois.extend(aoi["external"].get("ex_poi_ids", []))
    logging.info(f"Added POI num: {len(added_ex_pois)}")
    # Post-compute
    logging.info("Post Compute")
    (aois, pois) = _add_pois(aois, raw_pois, projstr)
    return (aois, pois)
