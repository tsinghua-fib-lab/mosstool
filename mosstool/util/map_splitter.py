"""
split map into multiple parts 
"""

import logging
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pyproj
from geojson import FeatureCollection
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree

from ..type import Map
from .format_converter import dict2pb, pb2dict

__all__ = ["split_map"]


def _center_point(lanes_dict: Dict[int, dict], lane_ids: List[int]) -> Point:
    lane_coords = []
    for lane_id in lane_ids:
        lane_coords.extend(
            [[n["x"], n["y"]] for n in lanes_dict[lane_id]["center_line"]["nodes"]]
        )
    lane_coords = np.array(lane_coords)
    x_center, y_center = np.mean(lane_coords, axis=0)
    return Point(x_center, y_center)


def _gen_header(map_name: str, poly_id: int, proj_str: str, lanes: List[dict]) -> dict:
    xy = [
        [i["x"], i["y"]] for j in lanes for k in ["center_line"] for i in j[k]["nodes"]
    ]
    x, y = [*zip(*xy)]
    return {
        "name": f"{map_name}_{poly_id}",
        "date": time.strftime("%a %b %d %H:%M:%S %Y"),
        "north": max(y),
        "south": min(y),
        "west": min(x),
        "east": max(x),
        "projection": proj_str,
    }


def split_map(
    geo_data: FeatureCollection,
    map: Map,
    output_path: Optional[str] = None,
    distance_threshold: float = 50.0,
) -> List[Map]:
    """
    Args:
    - geo_data (FeatureCollection): polygon geo files.
    - map (Map): the map.
    - output_path (str): splitted map output path.
    - distance_threshold (float): maximum distance considered to be contained in a bounding box.

    Returns:
    - List of splitted maps.
    """
    map_dict = pb2dict(map)
    lanes_dict = {d["id"]: d for d in map_dict["lanes"]}
    roads_dict = {d["id"]: d for d in map_dict["roads"]}
    road_center_points = {
        road_id: _center_point(lanes_dict, road["lane_ids"])
        for road_id, road in roads_dict.items()
    }
    junctions_dict = {d["id"]: d for d in map_dict["junctions"]}
    junction_center_points = {
        junction_id: _center_point(lanes_dict, junction["lane_ids"])
        for junction_id, junction in junctions_dict.items()
    }
    aois_dict = {d["id"]: d for d in map_dict["aois"]}
    pois_dict = {d["id"]: d for d in map_dict["pois"]}
    # keys: 'header', 'lanes', 'roads', 'junctions', 'aois', 'pois'
    proj_str = map_dict["header"]["projection"]
    projector = pyproj.Proj(proj_str)
    map_name = map_dict["header"]["name"]
    all_geos = []
    tree_id2poly_id = {}
    for i, feature in enumerate(geo_data["features"]):
        if not feature["geometry"]["type"] == "Polygon":
            raise ValueError("bad geometry type: " + feature)
        if "properties" not in feature:
            raise ValueError("no properties in feature: " + feature)
        coords = np.array(
            feature["geometry"]["coordinates"][0], dtype=np.float64
        )  # inner poly is not supported
        xy_coords = np.stack(projector(*coords.T[:2]), axis=1)
        poly_id = feature["properties"].get("id", len(geo_data["features"]) + i)
        polygon = Polygon(xy_coords)
        tree_id2poly_id[len(all_geos)] = poly_id
        all_geos.append(polygon)
    all_geos_tree = STRtree(all_geos)
    output_map_dict = {
        poly_id: {
            "header": None,
            "lanes": [],
            "roads": [],
            "junctions": [],
            "aois": [],
            "pois": [],
        }
        for _, poly_id in tree_id2poly_id.items()
    }
    # split junctions
    for junc_id, junc_center_point in junction_center_points.items():
        tree_ids = all_geos_tree.query_nearest(
            geometry=junc_center_point,
            max_distance=distance_threshold,
            return_distance=False,
        )
        if not len(tree_ids) > 0:
            raise ValueError(
                f"Center point of junction {junc_id} is not inside any bounding box"
            )
        else:
            inner_poly_id = tree_id2poly_id[tree_ids[0]]
            output_map_dict[inner_poly_id]["junctions"].append(junctions_dict[junc_id])
    # split roads
    for road_id, road_center_point in road_center_points.items():
        tree_ids = all_geos_tree.query_nearest(
            geometry=road_center_point,
            max_distance=distance_threshold,
            return_distance=False,
        )
        if not len(tree_ids) > 0:
            raise ValueError(
                f"Center point of road {road_id} is not inside any bounding box"
            )
        else:
            inner_poly_id = tree_id2poly_id[tree_ids[0]]
            output_map_dict[inner_poly_id]["roads"].append(roads_dict[road_id])
    output_map_pbs: Dict[int, Map] = {}
    added_aoi_ids = set()
    # add lane, aoi, poi
    for poly_id, partial_map_data in output_map_dict.items():
        # Areas has no map units inside
        if (
            len(partial_map_data["roads"]) == 0
            and len(partial_map_data["junctions"]) == 0
        ):
            logging.info(f"Geo {poly_id} has no map unit inside")
            continue
        else:
            all_lane_ids = []
            for road in partial_map_data["roads"]:
                all_lane_ids.extend(road["lane_ids"])
            for junc in partial_map_data["junctions"]:
                all_lane_ids.extend(junc["lane_ids"])
            for lane_id in all_lane_ids:
                lane = lanes_dict[lane_id]
                # add lane
                partial_map_data["lanes"].append(lane)
                # add aoi
                for aoi_id in lane["aoi_ids"]:
                    if aoi_id in added_aoi_ids:
                        continue
                    else:
                        added_aoi_ids.add(aoi_id)
                    aoi = aois_dict[aoi_id]
                    partial_map_data["aois"].append(aoi)
                    # add poi
                    for poi_id in aoi["poi_ids"]:
                        poi = pois_dict[poi_id]
                        partial_map_data["pois"].append(poi)
                # header
                partial_map_data["header"] = _gen_header(
                    map_name, poly_id, proj_str, partial_map_data["lanes"]
                )
            output_map_pbs[poly_id] = dict2pb(partial_map_data, Map())
    if output_path is not None:
        for poly_id, pb in output_map_pbs.items():
            with open(os.path.join(output_path, f"{map_name}_{poly_id}.pb"), "wb") as f:
                f.write(pb.SerializeToString())
    return list(output_map_pbs.values())
