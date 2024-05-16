"""
merge all input maps into one map
"""

import time
from typing import List, Optional

from ..map._map_util.format_checker import output_format_check
from ..type import Map
from .format_converter import dict2pb, pb2dict


def merge_map(
    partial_maps: List[Map],
    output_path: Optional[str] = None,
) -> Map:
    """
    Args:
    - partial_maps (list[Map]): maps to be merged.
    - output_path (str): merged map output path.

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
    output_format_check(output_map_dict)
    map_pb = dict2pb(output_map_dict, Map())
    if output_path is not None:
        with open(output_path, "wb") as f:
            f.write(map_pb.SerializeToString())

    return map_pb
