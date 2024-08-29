import logging
from collections import Counter, defaultdict
from copy import deepcopy
from math import atan2
from typing import (Callable, Dict, List, Literal, Optional, Set, Tuple, Union,
                    cast)

import numpy as np
import pycityproto.city.map.v2.map_pb2 as mapv2
from shapely.geometry import LineString, MultiPoint, Point

from ..const import *
from ..._util.angle import abs_delta_angle, delta_angle
from ..._util.line import get_line_angle, get_start_vector


def classify_main_auxiliary_wid(
    wids: List[int],
    way_angle: float,
    group_type: Union[Literal["in_ways"], Literal["out_ways"]],
    map_roads: dict,
):
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
        right_lanes = {wid: map_roads[wid]["lanes"][-1] for wid in wids}
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
        left_lanes = {wid: map_roads[wid]["lanes"][0] for wid in wids}
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


def check_1_n_available_turns(
    in_main_lanes: list,
    in_auxiliary_lanes: list,
    in_straight_main_lanes: list,
    in_straight_auxiliary_lanes: list,
    out_main_group,
    out_left_groups: list,
    out_right_groups: list,
    map_roads: dict,
    main_left_lane_num: int,
    auxiliary_left_lane_num: int,
    main_right_lane_num,
    auxiliary_right_lane_num: int,
):
    res = {in_type: defaultdict(bool) for in_type in ["main", "auxiliary"]}
    # Straight
    if out_main_group is not None:
        out_angle, out_way_ids = out_main_group
        out_main_wid, out_auxiliary_wids = classify_main_auxiliary_wid(
            out_way_ids, out_angle, "out_ways", map_roads
        )
        out_main_lanes = map_roads[out_main_wid]["lanes"]
        out_auxiliary_lanes = [
            l for wid in out_auxiliary_wids for l in map_roads[wid]["lanes"]
        ]
        # main road to main road
        if len(in_straight_main_lanes) > 0 and len(out_main_lanes) > 0:
            res["main"]["straight"] = True
        # auxiliary road to auxiliary road
        if len(in_straight_auxiliary_lanes) > 0 and len(out_auxiliary_lanes) > 0:
            res["auxiliary"]["straight"] = True
        # When there is no in auxiliary road, connect the in main road to the out auxiliary road
        if (
            not len(in_straight_auxiliary_lanes) > 0
            and len(out_auxiliary_lanes) > 0
            and len(in_straight_main_lanes) > 0
        ):
            res["main"]["straight"] = True
        # When there is no out auxiliary road, connect the in auxiliary road to the main road
        if (
            len(in_straight_auxiliary_lanes) > 0
            and not len(out_auxiliary_lanes) > 0
            and len(out_main_lanes) > 0
        ):
            res["auxiliary"]["straight"] = True
    # Left
    if len(out_left_groups) > 0:
        for out_angle, out_way_ids in out_left_groups:
            (
                out_main_wid,
                out_auxiliary_wids,
            ) = classify_main_auxiliary_wid(
                out_way_ids, out_angle, "out_ways", map_roads
            )
            out_main_lanes = map_roads[out_main_wid]["lanes"]
            out_auxiliary_lanes = [
                l for wid in out_auxiliary_wids for l in map_roads[wid]["lanes"]
            ]
            # main road to main road
            if len(in_main_lanes[:main_left_lane_num]) > 0 and len(out_main_lanes) > 0:
                res["main"]["left"] = True
            # auxiliary road to auxiliary road
            if (
                len(in_auxiliary_lanes[:auxiliary_left_lane_num]) > 0
                and len(out_auxiliary_lanes) > 0
            ):
                res["auxiliary"]["left"] = True
            # When there is no in auxiliary road, connect the in main road to the out auxiliary road
            if (
                not len(in_auxiliary_lanes[:main_left_lane_num]) > 0
                and len(out_auxiliary_lanes) > 0
                and len(in_main_lanes[:main_left_lane_num]) > 0
            ):
                res["main"]["left"] = True
            # When there is no out auxiliary road, connect the in auxiliary road to the main road
            if (
                len(in_auxiliary_lanes[:auxiliary_left_lane_num]) > 0
                and not len(out_auxiliary_lanes) > 0
                and len(out_main_lanes) > 0
            ):
                res["auxiliary"]["left"] = True
    # Right
    if len(out_right_groups) > 0:
        for out_angle, out_way_ids in out_right_groups:
            (
                out_main_wid,
                out_auxiliary_wids,
            ) = classify_main_auxiliary_wid(
                out_way_ids, out_angle, "out_ways", map_roads
            )
            out_main_lanes = map_roads[out_main_wid]["lanes"]
            out_auxiliary_lanes = [
                l for wid in out_auxiliary_wids for l in map_roads[wid]["lanes"]
            ]
            # main road to main road
            if (
                len(in_main_lanes[-main_right_lane_num:]) > 0
                and len(out_main_lanes) > 0
            ):
                res["main"]["right"] = True
            # auxiliary road to auxiliary road
            if (
                len(in_auxiliary_lanes[-auxiliary_right_lane_num:]) > 0
                and len(out_auxiliary_lanes) > 0
            ):
                res["auxiliary"]["right"] = True
            # When there is no in auxiliary road, connect the in main road to the out auxiliary road
            if (
                not len(in_auxiliary_lanes[-main_right_lane_num:]) > 0
                and len(out_auxiliary_lanes) > 0
                and len(in_main_lanes[-main_right_lane_num:]) > 0
            ):
                res["main"]["right"] = True
            # When there is no out auxiliary road, connect the in auxiliary road to the main road
            if (
                len(in_auxiliary_lanes[-auxiliary_right_lane_num:]) > 0
                and not len(out_auxiliary_lanes) > 0
                and len(out_main_lanes) > 0
            ):
                res["auxiliary"]["right"] = True
    return res


def check_n_n_available_turns(
    in_main_lanes: list,
    in_auxiliary_lanes: list,
    main_count: int,
    auxiliary_main_count: int,
    main_out_start: int,
    main_out_end: int,
    auxiliary_main_out_start: int,
    auxiliary_main_out_end: int,
    left_count: int,
    auxiliary_left_count: int,
    right_count: int,
    auxiliary_right_count: int,
    out_main_group,
    out_left_groups: list,
    out_right_groups: list,
    map_roads: dict,
):
    res = {in_type: defaultdict(bool) for in_type in ["main", "auxiliary"]}
    if main_count > 0 and out_main_group is not None:
        out_angle, out_way_ids = out_main_group
        (
            out_main_wid,
            out_auxiliary_wids,
        ) = classify_main_auxiliary_wid(out_way_ids, out_angle, "out_ways", map_roads)
        out_main_lanes = map_roads[out_main_wid]["lanes"]
        out_auxiliary_lanes = [
            l for wid in out_auxiliary_wids for l in map_roads[wid]["lanes"]
        ]
        # main road to main road
        if (len(in_main_lanes[main_out_start:main_out_end])) > 0 and len(
            out_main_lanes
        ) > 0:
            res["main"]["straight"] = True
        # auxiliary road to auxiliary road
        if (
            len(in_auxiliary_lanes[auxiliary_main_out_start:auxiliary_main_out_end]) > 0
            and len(out_auxiliary_lanes) > 0
        ):
            res["auxiliary"]["straight"] = True
        # When there is no in auxiliary road, connect the in main road to the out auxiliary road
        if (
            not len(in_auxiliary_lanes) > 0
            and len(out_auxiliary_lanes) > 0
            and len(in_main_lanes[main_out_start:main_out_end]) > 0
        ):
            res["main"]["straight"] = True
        # When there is no out auxiliary road, connect the in auxiliary road to the main road
        if (
            len(in_auxiliary_lanes[auxiliary_main_out_start:auxiliary_main_out_end]) > 0
            and not len(out_auxiliary_lanes) > 0
            and len(out_main_lanes) > 0
        ):
            res["auxiliary"]["straight"] = True
    if right_count > 0:
        for out_angle, out_way_ids in out_right_groups:
            (
                out_main_wid,
                out_auxiliary_wids,
            ) = classify_main_auxiliary_wid(
                out_way_ids, out_angle, "out_ways", map_roads
            )
            out_main_lanes = map_roads[out_main_wid]["lanes"]
            out_auxiliary_lanes = [
                l for wid in out_auxiliary_wids for l in map_roads[wid]["lanes"]
            ]
            # main road to main road
            if (
                len(in_main_lanes[-right_count:]) > 0
                and len(out_main_lanes[-right_count:]) > 0
            ):
                res["main"]["right"] = True
            # auxiliary road to auxiliary road
            if (
                len(in_auxiliary_lanes[-auxiliary_right_count:]) > 0
                and len(out_auxiliary_lanes[-auxiliary_right_count:]) > 0
            ):
                res["auxiliary"]["right"] = True
            # When there is no out auxiliary road, connect the in auxiliary road to the main road
            if (
                len(in_auxiliary_lanes[-auxiliary_right_count:]) > 0
                and not len(out_auxiliary_lanes) > 0
                and len(out_main_lanes[-right_count:]) > 0
            ):
                res["auxiliary"]["right"] = True
    if left_count > 0:
        for out_angle, out_way_ids in out_left_groups:
            (
                out_main_wid,
                out_auxiliary_wids,
            ) = classify_main_auxiliary_wid(
                out_way_ids, out_angle, "out_ways", map_roads
            )
            out_main_lanes = map_roads[out_main_wid]["lanes"]
            out_auxiliary_lanes = [
                l for wid in out_auxiliary_wids for l in map_roads[wid]["lanes"]
            ]
            # main road to main road
            if (
                len(in_main_lanes[:left_count]) > 0
                and len(out_main_lanes[:left_count]) > 0
            ):
                res["main"]["left"] = True
            # auxiliary road to auxiliary road
            if (
                len(in_auxiliary_lanes[:auxiliary_left_count]) > 0
                and len(out_auxiliary_lanes[:auxiliary_left_count]) > 0
            ):
                res["auxiliary"]["left"] = True
            # When there is no out auxiliary road, connect the in auxiliary road to the main road
            if (
                len(in_auxiliary_lanes[:auxiliary_left_count]) > 0
                and not len(out_auxiliary_lanes) > 0
                and len(out_main_lanes[:left_count])
            ):
                res["auxiliary"]["left"] = True
    return res


def add_overlaps(output_junctions: dict, output_lanes: dict) -> None:
    for _, junc in output_junctions.items():
        junc_poly_lanes = [
            (l, LineString([[n["x"], n["y"]] for n in l["center_line"]["nodes"]]))
            for l in (output_lanes[lid] for lid in junc["lane_ids"])
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


def add_driving_groups(
    output_junctions: dict, output_lanes: dict, output_roads: dict
) -> None:
    # Shallow copy
    lanes = {lid: d for lid, d in output_lanes.items()}
    roads = {rid: d for rid, d in output_roads.items()}
    juncs = {jid: d for jid, d in output_junctions.items()}

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
            group_turn = turns[0]
            if not all(t == group_turn for t in turns):
                # Reset group turn as the most common turn
                group_turn = Counter(turns).most_common(1)[0][0]
                logging.warning(
                    f"Not all lane turns are the same at junction {jid} driving group, reset all lane turn to {group_turn}"
                )
                for lid in lane_ids:
                    lanes[lid]["turn"] = group_turn
            group["turn"] = group_turn
