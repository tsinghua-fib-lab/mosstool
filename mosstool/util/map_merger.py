"""
merge all input maps into one map
"""

import logging
import time
from copy import deepcopy
from typing import List, Optional

from ..map._map_util.format_checker import output_format_check
from ..type import Map
from .format_converter import dict2pb, pb2dict


def _filter_map(map_dict: dict):
    """
    Filter invalid values in output map
    """
    output_map = deepcopy(map_dict)
    lanes = output_map["lanes"]
    roads = output_map["roads"]
    juncs = output_map["junctions"]
    aois = output_map["aois"]
    all_lane_ids = set(l["id"] for l in lanes)
    # aoi
    filter_aois = []
    for aoi in aois:
        aoi_id = aoi["id"]
        aoi_pos_gate_key_tuples = (
            ("driving_positions", "driving_gates", "Driving"),
            ("walking_positions", "walking_gates", "Walking"),
        )
        for pos_key, gate_key, pos_type in aoi_pos_gate_key_tuples:
            a_pos = aoi[pos_key]
            a_gate = aoi[gate_key]
            assert len(a_pos) == len(
                a_gate
            ), f"Different {pos_type} position and gates length in Aoi {aoi_id}"
            pos_idxes = []
            for i, pos in enumerate(a_pos):
                pos_l_id = pos["lane_id"]
                if pos_l_id in all_lane_ids:
                    pos_idxes.append(i)
                else:
                    logging.warning(
                        f"{pos_type} position in Aoi {aoi_id} has lane {pos_l_id} not in map!"
                    )
            aoi[pos_key] = [pos for i, pos in enumerate(a_pos) if i in pos_idxes]
            aoi[gate_key] = [gate for i, gate in enumerate(a_gate) if i in pos_idxes]
        if all(len(aoi[pos_key]) == 0 for pos_key, _, _ in aoi_pos_gate_key_tuples):
            # not connected to roadnet
            logging.warning(f"Aoi {aoi_id} has no gates connected to the roadnet!")
            continue
        else:
            filter_aois.append(aoi)
    output_map["aois"] = filter_aois
    return output_map


def merge_map(
    partial_maps: List[Map],
    output_path: Optional[str] = None,
    output_lane_length_check: bool = False,
) -> Map:
    """
    Args:
    - partial_maps (list[Map]): maps to be merged.
    - output_path (str): merged map output path.
    - output_lane_length_check (bool): whether to check lane length.

    Returns:
    - merged map.
    """
    if len(partial_maps) == 0:
        raise ValueError("Input maps should have at least one map")
    list_dict_maps = [pb2dict(m) for m in partial_maps]
    output_map_dict = {
        "header": None,
        "lanes": [],
        "roads": [],
        "junctions": [],
        "aois": [],
        "pois": [],
    }
    projstr_set = set()
    map_name_list = list()
    for map_dict in list_dict_maps:
        map_header = map_dict["header"]
        projstr_set.add(map_header["projection"])
        map_name_list.append(map_header["name"].split("_")[0])
        for class_name in [
            "lanes",
            "roads",
            "junctions",
            "aois",
            "pois",
        ]:
            output_map_dict[class_name].extend(map_dict[class_name])
    if len(projstr_set) > 1:
        raise ValueError("More than one projection str in input maps")
    # generate header
    xy = [
        [i["x"], i["y"]]
        for j in output_map_dict["lanes"]
        for k in ["center_line"]
        for i in j[k]["nodes"]
    ]
    x, y = [*zip(*xy)]
    output_map_dict["header"] = {
        "name": map_name_list[0],
        "date": time.strftime("%a %b %d %H:%M:%S %Y"),
        "north": max(y),
        "south": min(y),
        "west": min(x),
        "east": max(x),
        "projection": list(projstr_set)[0],
    }
    output_map_dict = _filter_map(output_map_dict)
    output_format_check(output_map_dict, output_lane_length_check)
    map_pb = dict2pb(output_map_dict, Map())
    if output_path is not None:
        with open(output_path, "wb") as f:
            f.write(map_pb.SerializeToString())

    return map_pb
