"""
match aois from map.pb
"""

from math import ceil
from multiprocessing import Pool

from shapely.geometry import Polygon
from shapely.strtree import STRtree

from ...type import Map
from ...util.format_converter import pb2dict
from .aoi_matcher import _process_matched_result, _str_tree_matcher_unit
from .const import *


def _map_aoi2geo(aoi: dict) -> Polygon:
    coords = [(c["x"], c["y"]) for c in aoi["positions"]]
    return Polygon(coords)


def _add_aoi_unit(aoi):
    global d_matcher, w_matcher
    global D_DIS_GATE, D_HUGE_GATE
    global W_DIS_GATE, W_HUGE_GATE
    global d_tree, w_tree
    geo = _map_aoi2geo(aoi)
    # d_matched = _matcher_unit(geo, d_matcher, D_DIS_GATE, D_HUGE_GATE)
    # w_matched = _matcher_unit(geo, w_matcher, W_DIS_GATE, W_HUGE_GATE)
    d_matched = _str_tree_matcher_unit(geo, d_matcher, d_tree, D_DIS_GATE, D_HUGE_GATE)
    w_matched = _str_tree_matcher_unit(geo, w_matcher, w_tree, W_DIS_GATE, W_HUGE_GATE)
    aoi["external"] = {}
    if d_matched or w_matched:
        return _process_matched_result(
            aoi=aoi, d_matched=d_matched, w_matched=w_matched
        )


def match_map_aois(net: Map, matchers: dict, workers: int):
    global d_matcher, w_matcher
    global d_tree, w_tree
    net_dict = pb2dict(net)
    orig_aois = net_dict["aois"]
    orig_pois = net_dict["pois"]
    d_matcher, w_matcher = (
        matchers["drive"],
        matchers["walk"],
    )
    d_tree = STRtree([l["geo"] for l in d_matcher])
    w_tree = STRtree([l["geo"] for l in w_matcher])
    results_aois = []
    for i in range(0, len(orig_aois), MAX_BATCH_SIZE):
        args_batch = orig_aois[i : i + MAX_BATCH_SIZE]
        with Pool(processes=workers) as pool:
            results_aois += pool.map(
                _add_aoi_unit,
                args_batch,
                chunksize=min(ceil(len(args_batch) / workers), 500),
            )
    aois = [r for r in results_aois if r]
    # filter pois
    all_poi_ids = set(pid for a in aois for pid in a["poi_ids"])
    pois = [p for p in orig_pois if p["id"] in all_poi_ids]
    # rearrange aoi id
    aois_dict = {}
    _poi_uid_dict = {
        poi["id"]: poi_id for poi_id, poi in enumerate(pois, start=POI_START_ID)
    }
    for aoi_id, aoi in enumerate(aois, start=AOI_START_ID):
        aoi["id"] = aoi_id
        aois_dict[aoi_id] = aoi
        # update poi_ids
        aoi["poi_ids"] = [_poi_uid_dict[pid] for pid in aoi["poi_ids"]]
    # rearrange poi id
    pois_dict = {}
    _valid_aois = [(a["id"], aoi) for aoi, a in zip(results_aois, orig_aois) if aoi]
    _aoi_uid_dict = {
        aid: AOI_START_ID + idx for idx, (aid, _) in enumerate(_valid_aois)
    }
    for poi_id, poi in enumerate(pois, start=POI_START_ID):
        poi["id"] = poi_id
        pois_dict[poi_id] = poi
        # update aoi_id
        poi["aoi_id"] = _aoi_uid_dict[poi["aoi_id"]]
    return (aois_dict, pois_dict)
