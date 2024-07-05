import asyncio
from collections import defaultdict
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from typing import Optional

import numpy as np
from pycityproto.city.routing.v2.routing_service_pb2 import GetRouteRequest
import pycityproto.city.map.v2.map_pb2 as mapv2

from tqdm import tqdm

from mosstool.trip.route import RoutingClient

from .._map_util.const import *

__all__ = [
    "public_transport_process",
]
ETA_FACTOR = 5
TAZ_LENGTH = 1500


async def _fill_public_lines(m: dict, server_address: str):
    m = deepcopy(m)
    public_line_id = PUBLIC_LINE_START_ID
    aois = {a["id"]: a for a in m["aois"]}
    lanes = {l["id"]: l for l in m["lanes"]}
    roads = {r["id"]: r for r in m["roads"]}
    aoi_center_point = {}
    for aoi_id, aoi in aois.items():
        coords = np.array([[n["x"], n["y"]] for n in aoi["positions"]])
        x, y = np.mean(coords, axis=0)[:2]
        aoi_center_point[aoi_id] = (x, y)

    def get_subline_type(pub_type: str):
        if pub_type == "SUBWAY":
            return mapv2.SUBLINE_TYPE_SUBWAY
        elif pub_type == "BUS":
            return mapv2.SUBLINE_TYPE_BUS
        else:
            return mapv2.SUBLINE_TYPE_UNSPECIFIED

    def get_aoi_dis(aoi_id1, aoi_id2):
        x1, y1 = aoi_center_point[aoi_id1]
        x2, y2 = aoi_center_point[aoi_id2]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    pub_data = m["public_transport"]
    client = RoutingClient(server_address)
    aoi_id2route_position = defaultdict(list)
    sublines_data = []

    def route_length(
        road_ids, route_start: Optional[dict] = None, route_end: Optional[dict] = None
    ) -> float:
        res = 0
        if len(road_ids) == 0:
            return res
        if route_start is not None and route_end is not None:
            for road_id in road_ids[:-1]:
                road = roads[road_id]
                center_lane_id = road["lane_ids"][len(road["lane_ids"]) // 2]
                center_lane = lanes[center_lane_id]
                res += center_lane["length"]
            res -= route_start["lane_position"]["s"]
            res += route_end["lane_position"]["s"]
        else:
            for road_id in road_ids:
                road = roads[road_id]
                center_lane_id = road["lane_ids"][len(road["lane_ids"]) // 2]
                center_lane = lanes[center_lane_id]
                res += center_lane["length"]
        return res

    # Filter sublines that are reachable in the map
    for pub in tqdm(pub_data):
        for subline in pub["sublines"]:
            if pub["type"] == "BUS":
                station_connection_road_ids = subline["station_connection_road_ids"]
                aoi_ids = subline["aoi_ids"]
                if station_connection_road_ids:
                    continue
                else:
                    aoi_ids = [
                        a_id
                        for a_id in aoi_ids
                        if len(aois[a_id]["driving_positions"]) >= 1
                    ]
                    if len(aoi_ids) < 2:
                        continue
                    aoi_start = aoi_ids[0]
                    start_positions = [
                        {"lane_position": p}
                        for p in aois[aoi_start]["driving_positions"]
                    ]
                    res_conn_road_ids = []
                    res_aoi_ids = [aoi_start]
                    res_etas = []
                    for aoi_end in aoi_ids[1:]:
                        loop_res = []
                        end_positions = [
                            {"lane_position": p}
                            for p in aois[aoi_end]["driving_positions"]
                        ]
                        for route_start in start_positions:
                            for route_end in end_positions:
                                res = await client.GetRoute(
                                    req=GetRouteRequest(
                                        type=1,  # driving
                                        start=route_start,
                                        end=route_end,
                                        time=0,
                                    )
                                )
                                if res and res.journeys:
                                    road_ids = list(res.journeys[0].driving.road_ids)
                                    route_len = route_length(
                                        road_ids, route_start, route_end
                                    )
                                    if len(
                                        road_ids
                                    ) >= 1 and route_len < 1.6 * get_aoi_dis(
                                        aoi_start, aoi_end
                                    ):
                                        eta = res.journeys[0].driving.eta
                                        loop_res.append(
                                            {
                                                "route_start": route_start,
                                                "route_end": route_end,
                                                "road_ids": road_ids,
                                                "route_length": route_len,
                                                "eta": eta * ETA_FACTOR,
                                            }
                                        )
                        if len(loop_res) > 0:
                            # there is a path
                            min_loop_res = min(
                                loop_res, key=lambda r: r["route_length"]
                            )
                            aoi_id2route_position[aoi_start].append(
                                min_loop_res["route_start"]
                            )
                            aoi_id2route_position[aoi_end].append(
                                min_loop_res["route_end"]
                            )
                            road_ids = min_loop_res["road_ids"]
                            start_positions = [min_loop_res["route_end"]]
                            aoi_start = aoi_end
                            res_conn_road_ids.append(
                                {
                                    "road_ids": road_ids,
                                }
                            )
                            res_aoi_ids.append(aoi_end)
                            res_etas.append(min_loop_res["eta"])
                        else:
                            pass
                    if len(res_aoi_ids) >= 2:
                        subline["station_connection_road_ids"] = res_conn_road_ids
                        subline["aoi_ids"] = res_aoi_ids
                        subline["etas"] = res_etas
            if len(subline["station_connection_road_ids"]) > 0:
                # write to new subline data
                if "etas" in subline:
                    offset_times = subline["etas"]
                else:
                    offset_times = []
                    for d in subline["station_connection_road_ids"]:
                        road_ids = d["road_ids"]
                        offset_times.append(
                            route_length(road_ids) / DEFAULT_MAX_SPEED["SUBWAY"]
                        )
                departure_times = subline["schedules"]
                aoi_ids = subline["aoi_ids"]
                for aoi_id in aoi_ids:
                    aois[aoi_id]["subline_ids"].append(public_line_id)
                sublines_data.append(
                    {
                        "id": public_line_id,
                        "name": subline["name"],
                        "aoi_ids": aoi_ids,
                        "station_connection_road_ids": subline[
                            "station_connection_road_ids"
                        ],
                        "type": get_subline_type(pub["type"]),
                        "parent_name": subline["parent_name"],
                        "schedules": {
                            "departure_times": departure_times,
                            "offset_times": offset_times,
                        },
                        "taz_costs": [],
                    }
                )

                public_line_id += 1
    for aoi_id, aoi in aois.items():
        if len(aoi["subline_ids"]) == 0:
            continue
        # update walking_positions walking_gates
        w_pos = aoi["walking_positions"]
        w_gates = aoi["walking_gates"]
        external = aoi["external"]
        if w_pos:
            w_idx = min(
                [
                    {
                        "idx": i,
                        "distance": dis,
                    }
                    for i, dis in enumerate(external["walking_distances"])
                ],
                key=lambda x: x["distance"],
            )["idx"]
            aoi["walking_positions"] = [d for i, d in enumerate(w_pos) if i == w_idx]
            aoi["walking_gates"] = [d for i, d in enumerate(w_gates) if i == w_idx]
            external["walking_distances"] = [
                d for i, d in enumerate(external["walking_distances"]) if i == w_idx
            ]
            external["walking_lane_project_point"] = [
                d
                for i, d in enumerate(external["walking_lane_project_point"])
                if i == w_idx
            ]
        if aoi_id not in aoi_id2route_position:
            continue
        route_lane_ids = {
            p["lane_position"]["lane_id"] for p in aoi_id2route_position[aoi_id]
        }
        # update driving_positions driving_gates
        d_pos = aoi["driving_positions"]
        d_gates = aoi["driving_gates"]
        d_idxs = []
        for i, pos in enumerate(d_pos):
            if pos["lane_id"] in route_lane_ids:
                d_idxs.append(i)
        aoi["driving_positions"] = [d for i, d in enumerate(d_pos) if i in d_idxs]
        aoi["driving_gates"] = [d for i, d in enumerate(d_gates) if i in d_idxs]
        external = aoi["external"]
        for ex_key in ["driving_distances", "driving_lane_project_point"]:
            if ex_key in external:
                external[ex_key] = [
                    d for i, d in enumerate(external[ex_key]) if i in d_idxs
                ]
    # clear lane的aoi ids
    for _, lane in lanes.items():
        lane["aoi_ids"] = []
    # add aoi id
    for aoi_id, aoi in aois.items():
        for pos in aoi["driving_positions"]:
            lanes[pos["lane_id"]]["aoi_ids"].append(aoi_id)
        for pos in aoi["walking_positions"]:
            lanes[pos["lane_id"]]["aoi_ids"].append(aoi_id)
    # add public transportation lines
    del m["public_transport"]
    m["sublines"] = sublines_data
    return m


def _get_taz_cost_unit(arg):
    subline, station_aois, (x_min, x_step, y_min, y_step), route_lengths = arg
    station_aoi_ids = list(station_aois.keys())
    station_durations = [aoi["external"]["duration"] for _, aoi in station_aois.items()]
    subline_id = subline["id"]
    subline_type = subline["type"]
    if subline_type == mapv2.SUBLINE_TYPE_SUBWAY:
        vehicle_speed = DEFAULT_MAX_SPEED["SUBWAY"]
    else:
        vehicle_speed = DEFAULT_MAX_SPEED["BUS"]
    walk_speed = DEFAULT_MAX_SPEED["WALK"]
    taz_costs = []

    # build TAZ
    def get_taz_id(x, y):
        return round((x - x_min) / x_step), round((y - y_min) / y_step)

    aoi_id2taz_id = {}
    for aoi_id, aoi in station_aois.items():
        aoi_ps = np.array([[p["x"], p["y"]] for p in aoi["positions"]])
        p_aoi_x, p_aoi_y = np.mean(aoi_ps, axis=0)[:2]
        aoi_idx, aoi_idy = get_taz_id(p_aoi_x, p_aoi_y)
        aoi_id2taz_id[aoi_id] = (aoi_idx, aoi_idy)
    min_tazs = defaultdict(list)
    for i in range(len(station_aoi_ids)):
        cur_aoi_ids = station_aoi_ids[i:]
        for i_offset, taz_id in [
            (i_offset, t_id)
            for i_offset, (aoi_id, t_id) in enumerate(aoi_id2taz_id.items())
            if aoi_id in cur_aoi_ids
        ]:
            # The cost required to get off at the station closest to the target TAZ
            total_drive_length = sum(route_lengths[i : i + i_offset])
            drive_eta_time = total_drive_length / vehicle_speed + sum(
                station_durations[i : i + i_offset]
            )
            key = (taz_id[0], taz_id[1], cur_aoi_ids[0])
            min_tazs[key].append(drive_eta_time)
    for (taz_x_id, taz_y_id, cur_aoi_id), etas in min_tazs.items():
        taz_costs.append(
            {
                "taz_x_id": taz_x_id,
                "taz_y_id": taz_y_id,
                "aoi_id": cur_aoi_id,
                "cost": np.mean(etas),
            }
        )
    return (subline_id, taz_costs)


def _post_compute(m: dict, workers: int):
    header = m["header"]
    aois = {a["id"]: a for a in m["aois"]}
    lanes = {l["id"]: l for l in m["lanes"]}
    roads = {r["id"]: r for r in m["roads"]}
    x_max, x_min = header["east"], header["west"]
    y_max, y_min = header["north"], header["south"]
    _, x_step = np.linspace(
        x_min, x_max, max(int((x_max - x_min) / TAZ_LENGTH), 2), retstep=True
    )
    _, y_step = np.linspace(
        y_min, y_max, max(int((y_max - y_min) / TAZ_LENGTH), 2), retstep=True
    )
    # update header
    header["taz_x_step"] = x_step
    header["taz_y_step"] = y_step
    # calculate the cost to specific TAZ
    sublines_data = m["sublines"]
    taz_cost_args = []

    def station_distance(road_ids, s_start: float, s_end: float) -> float:
        res = 0
        for road_id in road_ids[:-1]:
            road = roads[road_id]
            center_lane_id = road["lane_ids"][len(road["lane_ids"]) // 2]
            center_lane = lanes[center_lane_id]
            res += center_lane["length"]
        res -= s_start
        res += s_end
        return res

    for subline in sublines_data:
        station_aois = {aoi_id: aois[aoi_id] for aoi_id in subline["aoi_ids"]}
        sta_aoi_ids = list(station_aois.keys())
        route_lengths = []
        sta_conn_rids = [d["road_ids"] for d in subline["station_connection_road_ids"]]
        for i in range(len(sta_aoi_ids) - 1):
            road_ids = sta_conn_rids[i]
            aoi_start = station_aois[sta_aoi_ids[i]]
            s_start = 0
            for d in aoi_start["driving_positions"]:
                lane = lanes[d["lane_id"]]
                if lane["parent_id"] == road_ids[0]:
                    s_start = d["s"]
                    break
            aoi_end = station_aois[sta_aoi_ids[i + 1]]
            s_end = 0
            for d in aoi_end["driving_positions"]:
                lane = lanes[d["lane_id"]]
                if lane["parent_id"] == road_ids[-1]:
                    s_end = d["s"]
                    break
            route_lengths.append(station_distance(road_ids, s_start, s_end))
        arg = (subline, station_aois, (x_min, x_step, y_min, y_step), route_lengths)
        taz_cost_args.append(arg)

    with Pool(processes=workers) as pool:
        taz_results = pool.map(
            _get_taz_cost_unit,
            taz_cost_args,
            chunksize=max((len(taz_cost_args) // workers), 500),
        )
    subline_id2taz_costs = {r[0]: r[1] for r in taz_results}
    for subline in sublines_data:
        subline_id = subline["id"]
        subline["taz_costs"] = subline_id2taz_costs[subline_id]
    for sl in sublines_data:
        if not sl["type"] == mapv2.SUBLINE_TYPE_SUBWAY:
            continue
        transfer_stations = []
        for aid in sl["aoi_ids"]:
            station = aois[aid]
            if len(station["subline_ids"]) > 1:
                transfer_stations.append(station)

    return m


def public_transport_process(m: dict, server_address: str, workers: int = cpu_count()):
    m = asyncio.run(_fill_public_lines(m, server_address))
    m = _post_compute(m, workers)
    return m
