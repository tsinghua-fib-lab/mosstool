import logging
from math import ceil
from typing import Any, Dict, List

__all__ = ["parse_osm_way_tags", "merge_way_nodes"]


def parse_osm_way_tags(
    tags: Dict[str, str], default_settings: Dict[str, Any]
) -> Dict[str, Any]:
    """
    tags: tags field for OSM units whose type==way
    default_settings: default # of lanes, the max speed(max_speed), and turns within a junction(turn), etc.
    """
    highway = tags.get("highway", "default")
    result = {
        "highway": highway,
        # default values
        "lanes": default_settings["lane"][highway],
        "max_speed": default_settings["max_speed"][highway],
        "oneway": tags.get("oneway", None) == "yes",
        "name": tags.get("name", ""),
    }
    # number of lanes
    if "lanes" in tags:
        try:
            # Rounding up after float parsing
            lanes = int(ceil(float(tags["lanes"])))
            if lanes > 0:
                result["lanes"] = lanes
            else:
                logging.warning(f"unexpected lanes: {tags['lanes']} in {tags}")
        except:
            logging.warning(f"unexpected lanes: {tags['lanes']} in {tags}")
    if "maxspeed" in tags:
        try:
            if "mph" in tags["maxspeed"]:
                result["max_speed"] = float(tags["maxspeed"].split(" ")[0]) * 0.44704
            else:
                result["max_speed"] = float(tags["maxspeed"]) / 3.6  # to m/s
        except:
            logging.warning(f"unexpected max speed: {tags['maxspeed']} in {tags}")
    return result


def merge_way_nodes(a_nodes: List[int], b_nodes: List[int]) -> List[int]:
    """
    Merge two groups of nodes, it is required that at least one pair of endpoints of the two groups of nodes are the same, otherwise an error will be reported.
    eg:
    [1,2,3], [3,4,5] -> [1,2,3,4,5]
    [3,2,1], [3,4,5] -> [3,2,1,4,5]
    """
    # Determine the endpoints of two sets of nodes
    a_start = a_nodes[0]
    a_end = a_nodes[-1]
    b_start = b_nodes[0]
    b_end = b_nodes[-1]

    # Check whether the endpoints of two sets of nodes have the same
    if a_start == b_start:
        # [1,2,3], [1,4,5] -> [3,2,1,4,5]
        return a_nodes[::-1] + b_nodes[1:]
    elif a_start == b_end:
        # [1,2,3], [4,5,1] -> [4,5,1,2,3]
        return b_nodes + a_nodes[1:]
    elif a_end == b_start:
        # [1,2,3], [3,4,5] -> [1,2,3,4,5]
        return a_nodes + b_nodes[1:]
    elif a_end == b_end:
        # [1,2,3], [5,4,3] -> [1,2,3,4,5]
        return a_nodes + b_nodes[::-1][1:]
    else:
        raise ValueError(
            f"The endpoints of the two sets of nodes do not match: {a_nodes}, {b_nodes}"
        )
