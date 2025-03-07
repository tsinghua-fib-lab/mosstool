import logging
from collections import defaultdict
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from math import ceil
from multiprocessing import Pool, cpu_count
from typing import Any, Literal, Optional, Union, cast

import numpy as np
import pyproj
import shapely.geometry as geometry
from geopandas.geodataframe import GeoDataFrame

from ...map._map_util.const import *
from ...type import (AoiPosition, LanePosition, Map, Person, PersonProfile,
                     Position, Schedule, Trip, TripMode)
from ...util.format_converter import dict2pb, pb2dict
from ._util.const import *
from ._util.utils import (extract_HWEO_from_od_matrix, gen_bus_drivers,
                          gen_departure_times, gen_profiles, gen_taxi_drivers,
                          recalculate_trip_mode_prob, recalculate_trip_modes)
from .template import default_person_template_generator


# from ...util.geo_match_pop import geo2pop
def geo_coords(geo):
    if isinstance(geo, geometry.Polygon):
        return list(geo.exterior.coords)
    elif isinstance(geo, geometry.MultiPolygon):
        all_coords = []
        for p_geo in geo.geoms:
            all_coords.extend(geo_coords(p_geo))
        return all_coords
    else:
        return list(geo.coords)


# determine trip mode
def _get_mode(p1, p2):
    (x1, y1), (x2, y2) = p1, p2
    dis = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    if dis > DIS_CAR:
        return CAR
    elif dis > DIS_BIKE:
        return BIKE
    else:
        return WALK


def _get_mode_with_distribution(
    partial_args: tuple[
        list[str],
        tuple[
            float, float, float, float, float, float, float, float, float, float, float
        ],
    ],
    p1: tuple[float, float],
    p2: tuple[float, float],
    profile: dict,
    seed: int = 0,
):
    available_trip_modes, (
        SUBWAY_EXPENSE,
        BUS_EXPENSE,
        DRIVING_SPEED,
        DRIVING_PENALTY,
        SUBWAY_SPEED,
        SUBWAY_PENALTY,
        BUS_SPEED,
        BUS_PENALTY,
        BIKE_SPEED,
        BIKE_PENALTY,
        PARKING_FEE,
    ) = partial_args
    (x1, y1), (x2, y2) = p1, p2
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    subway_expense = SUBWAY_EXPENSE
    bus_expense = BUS_EXPENSE
    driving_duration = distance / DRIVING_SPEED + DRIVING_PENALTY
    subway_duration = distance / SUBWAY_SPEED + SUBWAY_PENALTY
    bus_duration = distance / BUS_SPEED + BUS_PENALTY
    bicycle_duration = distance / BIKE_SPEED + BIKE_PENALTY
    parking_fee = PARKING_FEE
    age = 0.384  # proportion of ages from 18 to 35 population
    income = 0.395  # proportion of low-income population
    if bus_expense > 0:
        V_bus = -0.0516 * bus_duration / 60 - 0.4810 * bus_expense
    else:
        V_bus = -np.inf
    if subway_expense > 0:
        V_subway = -0.0512 * subway_duration / 60 - 0.0833 * subway_expense
    else:
        V_subway = -np.inf
    V_fuel = (
        -0.0705 * driving_duration / 60
        + 0.5680 * age
        - 0.8233 * income
        - 0.0941 * parking_fee
    )
    V_elec = -0.0339 * driving_duration / 60 - 0.1735 * parking_fee
    if distance > 15000:
        V_bicycle = -np.inf
    else:
        V_bicycle = -0.1185 * bicycle_duration / 60
    V = np.array([V_bus, V_subway, V_fuel, V_elec, V_bicycle])
    V = np.exp(V)
    _all_trip_modes = recalculate_trip_modes(
        profile, ALL_TRIP_MODES, available_trip_modes
    )
    V = recalculate_trip_mode_prob(profile, _all_trip_modes, V, available_trip_modes)
    V = V / sum(V)
    rng = np.random.default_rng(seed)
    choice_index = rng.choice(len(V), p=V)
    return _all_trip_modes[choice_index]


def _match_aoi_unit(partial_args: tuple[list[Any],], aoi):
    (shapes,) = partial_args
    p = aoi["shapely"].centroid
    for id, j in enumerate(shapes):
        if j.contains(p):
            return (aoi["id"], id)
    return None


def _generate_unit(
    partial_args: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[int, list[dict[str, Any]]],
        dict[int, dict[str, Any]],
        dict[str, list[int]],
        int,
        Any,
        list[float],
        int,
    ],
    get_mode_partial_args: tuple[
        list[str],
        tuple[
            float, float, float, float, float, float, float, float, float, float, float
        ],
    ],
    a_home_region: int,
    a_profile: dict[str, Any],
    modes: list[
        Union[Literal["H"], Literal["W"], Literal["E"], Literal["O"], Literal["+"]]
    ],
    p_mode: list[float],
    seed: int,
):
    # Three steps
    # 1.determine person activity mode
    # 2.determine departure times
    # 3.generate trip according to OD-matrix
    (
        H,
        W,
        E,
        O,
        region2aoi,
        aoi_map,
        aoi_type2ids,
        n_region,
        projector,
        departure_prob,
        LEN_OD_TIMES,
    ) = partial_args
    OD_TIME_INTERVAL = 24 / LEN_OD_TIMES  # (hour)

    def choose_aoi_with_type(
        region_id,
        aoi_type: Union[
            Literal["work"],
            Literal["home"],
            Literal["education"],
            Literal["other"],
        ],
    ):
        region_aois = region2aoi[region_id]
        if len(region_aois) > 0:
            if len(region2aoi[region_id]) == 1:
                return region2aoi[region_id][0]
            popu = np.zeros(len(region_aois))
            for i, id in enumerate(region_aois):
                popu[i] = aoi_map[id]["external"]["population"]  # type:ignore
            if sum(popu) == 0:
                idx = rng.choice(len(region_aois))
                return region_aois[idx]
            p = popu / sum(popu)
            idx = rng.choice(len(region_aois), p=p)
            return region_aois[idx]
        else:
            return rng.choice(aoi_type2ids[aoi_type])

    rng = np.random.default_rng(seed)
    if a_home_region is None:
        home_region = rng.choice(n_region, p=H)
    else:
        home_region = a_home_region
    mode_idx = rng.choice(len(modes), p=p_mode)
    mode = modes[mode_idx]
    aoi_list = []
    now_region = home_region
    # arrange time   times[i]: departure time from region i
    n_trip = len(mode) - 1 if mode[-1] == "H" else len(mode)
    times = gen_departure_times(rng, n_trip, mode, departure_prob)
    for t in times:
        assert t >= 0 and t <= 24

    # add hidden H for further process
    # HWH+ ->HWH+H
    if mode[-1] != "H":
        mode = mode + "H"
    # home aoi
    home_aoi = choose_aoi_with_type(home_region, "home")
    person_home = home_aoi
    person_work = None
    # work aoi
    work_indexes = [idx for (idx, m) in enumerate(mode) if m == "W"]
    if len(work_indexes) > 0:
        p = np.zeros((n_region))
        p = W[
            home_region,
            :,
            min(int(times[work_indexes[0] - 1] / OD_TIME_INTERVAL), LEN_OD_TIMES - 1),
        ]  # hour to int index
        p = p / sum(p)
        work_region = rng.choice(n_region, p=p)
        work_aoi = choose_aoi_with_type(work_region, "work")
        person_work = work_aoi
    else:
        work_region = None
        work_aoi = None
    # Study Aoi
    educate_indexes = [idx for (idx, m) in enumerate(mode) if m == "S"]
    if len(educate_indexes) > 0:
        p = np.zeros((n_region))
        p = E[
            home_region,
            :,
            min(
                int(times[educate_indexes[0] - 1] / OD_TIME_INTERVAL), LEN_OD_TIMES - 1
            ),
        ]  # hour to int index
        p = p / sum(p)
        educate_region = rng.choice(n_region, p=p)
        educate_aoi = choose_aoi_with_type(educate_region, "education")
        person_work = educate_aoi
    else:
        educate_region = None
        educate_aoi = None
    for idx, mode_i in enumerate(mode):
        if mode_i == "H":
            aoi_list.append(home_aoi)
            now_region = home_region
        elif mode_i == "W":
            aoi_list.append(work_aoi)
            now_region = work_region
        elif mode_i == "S":
            aoi_list.append(educate_aoi)
            now_region = educate_region
        elif mode_i == "+" or mode_i == "O":
            p = np.zeros((n_region))
            p = O[
                now_region,
                :,
                min(int(times[idx - 1] / OD_TIME_INTERVAL), LEN_OD_TIMES - 1),
            ]  # hour to int index
            p = p / sum(p)
            other_region = rng.choice(n_region, p=p)
            other_aoi = choose_aoi_with_type(other_region, "other")
            aoi_list.append(other_aoi)
            now_region = other_region
    trip_modes = []
    for cur_aoi, next_aoi in zip(aoi_list[:-1], aoi_list[1:]):
        lon1, lat1 = aoi_map[cur_aoi]["geo"][0][:2]
        lon2, lat2 = aoi_map[next_aoi]["geo"][0][:2]
        p1 = projector(longitude=lon1, latitude=lat1)
        p2 = projector(longitude=lon2, latitude=lat2)
        trip_modes.append(
            _get_mode_with_distribution(get_mode_partial_args, p1, p2, a_profile, seed)
        )

    # determine activity
    activities = []
    for next_aoi in aoi_list[1:]:
        activities.append(aoi_map[next_aoi]["external"]["catg"])
    # neither work or study
    if person_work is None:
        person_work = aoi_list[1]
    assert len(aoi_list) == len(times) + 1
    trip_models = ["" for _ in range(n_trip)]
    return (
        aoi_list,
        person_home,
        person_work,
        a_profile,
        times,
        trip_modes,
        trip_models,
        activities,
    )


def _process_agent_unit(
    partial_args_tuple: tuple[
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            dict[int, list[dict[str, Any]]],
            dict[int, dict[str, Any]],
            dict[str, list[int]],
            int,
            Any,
            list[float],
            int,
        ],
        tuple[
            list[str],
            tuple[
                float,
                float,
                float,
                float,
                float,
                float,
                float,
                float,
                float,
                float,
                float,
            ],
        ],
    ],
    arg: tuple[
        int,
        int,
        int,
        dict[str, Any],
        list[
            Union[Literal["H"], Literal["W"], Literal["E"], Literal["O"], Literal["+"]]
        ],
        list[float],
    ],
):
    partial_args, get_mode_partial_args = partial_args_tuple
    _, seed, home_region, profile, mode, p_mode = arg
    return _generate_unit(
        partial_args,
        get_mode_partial_args,
        home_region,
        profile,
        mode,
        p_mode,
        seed,
    )


def _fill_sch_unit(
    partial_args: tuple[
        np.ndarray,
        dict[int, list[dict[str, Any]]],
        dict[int, dict[str, Any]],
        dict[str, list[int]],
        int,
        Any,
        list[float],
        int,
    ],
    get_mode_partial_args: tuple[
        list[str],
        tuple[
            float, float, float, float, float, float, float, float, float, float, float
        ],
    ],
    p_home: int,
    p_home_region: int,
    p_work: int,
    p_work_region: int,
    p_profile: dict[str, Any],
    modes: list[
        Union[Literal["H"], Literal["W"], Literal["E"], Literal["O"], Literal["+"]]
    ],
    p_mode: list[float],
    seed: int,
):
    (
        O,
        region2aoi,
        aoi_map,
        aoi_type2ids,
        n_region,
        projector,
        departure_prob,
        LEN_OD_TIMES,
    ) = partial_args
    OD_TIME_INTERVAL = 24 / LEN_OD_TIMES  # (hour)

    def choose_aoi_with_type(
        region_id,
        aoi_type: Union[
            Literal["work"],
            Literal["home"],
            Literal["education"],
            Literal["other"],
        ],
    ):
        region_aois = region2aoi[region_id]
        if len(region_aois) > 0:
            if len(region2aoi[region_id]) == 1:
                return region2aoi[region_id][0]
            popu = np.zeros(len(region_aois))
            for i, id in enumerate(region_aois):
                popu[i] = aoi_map[id]["external"]["population"]  # type:ignore
            if sum(popu) == 0:
                idx = rng.choice(len(region_aois))
                return region_aois[idx]
            p = popu / sum(popu)
            idx = rng.choice(len(region_aois), p=p)
            return region_aois[idx]
        else:
            return rng.choice(aoi_type2ids[aoi_type])

    rng = np.random.default_rng(seed)
    mode_idx = rng.choice(len(modes), p=p_mode)
    mode = modes[mode_idx]
    aoi_list = []
    now_region = p_home_region
    # arrange time   times[i]: departure time from region i
    n_trip = len(mode) - 1 if mode[-1] == "H" else len(mode)
    times = gen_departure_times(rng, n_trip, mode, departure_prob)
    for t in times:
        assert t >= 0 and t <= 24

    # add hidden H for further process
    # HWH+ ->HWH+H
    if mode[-1] != "H":
        mode = mode + "H"
    # home aoi
    home_aoi = p_home
    home_region = p_home_region
    # work aoi
    work_aoi = p_work
    work_region = p_work_region
    # add aois
    for idx, mode_i in enumerate(mode):
        if mode_i == "H":
            aoi_list.append(home_aoi)
            now_region = home_region
        elif mode_i == "W" or mode_i == "S":
            aoi_list.append(work_aoi)
            now_region = work_region
        elif mode_i == "+" or mode_i == "O":
            p = np.zeros((n_region))
            p = O[
                now_region,
                :,
                min(int(times[idx - 1] / OD_TIME_INTERVAL), LEN_OD_TIMES - 1),
            ]  # hour to int index
            p = p / sum(p)
            other_region = rng.choice(n_region, p=p)
            other_aoi = choose_aoi_with_type(other_region, "other")
            aoi_list.append(other_aoi)
            now_region = other_region
    trip_modes = []
    for cur_aoi, next_aoi in zip(aoi_list[:-1], aoi_list[1:]):
        lon1, lat1 = aoi_map[cur_aoi]["geo"][0][:2]
        lon2, lat2 = aoi_map[next_aoi]["geo"][0][:2]
        p1 = projector(longitude=lon1, latitude=lat1)
        p2 = projector(longitude=lon2, latitude=lat2)
        trip_modes.append(
            _get_mode_with_distribution(get_mode_partial_args, p1, p2, p_profile, seed)
        )

    # determine activity
    activities = []
    for next_aoi in aoi_list[1:]:
        activities.append(aoi_map[next_aoi]["external"]["catg"])
    assert len(aoi_list) == len(times) + 1
    trip_models = ["" for _ in range(n_trip)]
    return (
        aoi_list,
        times,
        trip_modes,
        trip_models,
        activities,
    )


def _fill_person_schedule_unit(
    partial_args_tuple: tuple[
        tuple[
            np.ndarray,
            dict[int, list[dict[str, Any]]],
            dict[int, dict[str, Any]],
            dict[str, list[int]],
            int,
            Any,
            list[float],
            int,
        ],
        tuple[
            list[str],
            tuple[
                float,
                float,
                float,
                float,
                float,
                float,
                float,
                float,
                float,
                float,
                float,
            ],
        ],
    ],
    arg: tuple[
        int,
        int,
        int,
        int,
        int,
        dict[str, Any],
        list[
            Union[Literal["H"], Literal["W"], Literal["E"], Literal["O"], Literal["+"]]
        ],
        list[float],
        int,
    ],
):
    (
        p_id,
        p_home,
        p_home_region,
        p_work,
        p_work_region,
        p_profile,
        modes,
        p_mode,
        seed,
    ) = arg
    partial_args, get_mode_partial_args = partial_args_tuple
    return _fill_sch_unit(
        partial_args,
        get_mode_partial_args,
        p_home,
        p_home_region,
        p_work,
        p_work_region,
        p_profile,
        modes,
        p_mode,
        seed,
    )


__all__ = ["TripGenerator"]


class TripGenerator:
    """
    generate trip from OD matrix.
    """

    def __init__(
        self,
        m: Map,
        pop_tif_path: Optional[str] = None,
        activity_distributions: Optional[dict] = None,
        driving_speed: float = 30 / 3.6,
        parking_fee: float = 20.0,
        driving_penalty: float = 0.0,
        subway_speed: float = 35 / 3.6,
        subway_penalty: float = 600.0,
        subway_expense: float = 10.0,
        bus_speed: float = 15 / 3.6,
        bus_penalty: float = 600.0,
        bus_expense: float = 5.0,
        bike_speed: float = 10 / 3.6,
        bike_penalty: float = 0.0,
        template_func: Callable[[], Person] = default_person_template_generator,
        add_pop: bool = False,
        multiprocessing_chunk_size: int = 500,
        workers: int = cpu_count(),
    ):
        """
        Args:
        - m (Map): The Map.
        - pop_tif_path (str): path to population tif file.
        - activity_distributions (dict): human mobility mode and its probability. e.g. {"HWH": 18.0, "HWH+": 82.0,}. H for go home, W for go to work, O or + for other activities
        - driving_speed (float): vehicle speed(m/s) for traffic mode assignment.
        - parking_fee (float): money cost(￥) of parking a car for traffic mode assignment.
        - driving_penalty (float): extra cost(s) of vehicle for traffic mode assignment.
        - subway_speed (float): subway speed(m/s) for traffic mode assignment.
        - subway_penalty (float): extra cost(s) of subway for traffic mode assignment.
        - subway_expense (float): money cost(￥) of subway for traffic mode assignment.
        - bus_speed (float): bus speed(m/s) for traffic mode assignment.
        - bus_penalty (float): extra cost(s) of bus for traffic mode assignment.
        - bus_expense (float): money  cost(￥) of bus for traffic mode assignment.
        - bike_speed (float): extra cost(s) of bike for traffic mode assignment.
        - bike_penalty (float): money  cost(￥) of bike for traffic mode assignment.
        - template_func (Callable[[],Person]): The template function of generated person object, whose `schedules`, `home` will be replaced and others will be copied.
        - add_pop (bool): Add population to aois.
        - multiprocessing_chunk_size (int): the maximum size of each multiprocessing chunk
        - workers (int): number of workers.
        """
        SUBWAY_EXPENSE, BUS_EXPENSE = subway_expense, bus_expense
        DRIVING_SPEED, DRIVING_PENALTY, PARKING_FEE = (
            driving_speed,
            driving_penalty,
            parking_fee,
        )
        SUBWAY_SPEED, SUBWAY_PENALTY = subway_speed, subway_penalty
        BUS_SPEED, BUS_PENALTY = bus_speed, bus_penalty
        BIKE_SPEED, BIKE_PENALTY = bike_speed, bike_penalty
        self._trip_mode_cost_partial_args = (
            SUBWAY_EXPENSE,
            BUS_EXPENSE,
            DRIVING_SPEED,
            DRIVING_PENALTY,
            SUBWAY_SPEED,
            SUBWAY_PENALTY,
            BUS_SPEED,
            BUS_PENALTY,
            BIKE_SPEED,
            BIKE_PENALTY,
            PARKING_FEE,
        )
        self.m = m
        self.pop_tif_path = pop_tif_path
        self.add_pop = add_pop
        self.projector = pyproj.Proj(m.header.projection)
        self.max_chunk_size = multiprocessing_chunk_size
        self.workers = workers
        self.template = template_func
        self.persons = []
        # activity proportion
        if activity_distributions is not None:
            ori_modes_stat = {
                k: float(v)
                for k, v in activity_distributions.items()
                if k in HUMAN_MODE_STATS
            }
        else:
            ori_modes_stat = HUMAN_MODE_STATS
        self.modes = [mode for mode in ori_modes_stat.keys()]
        self.p = np.array([prob for prob in ori_modes_stat.values()])
        self.p_mode = self.p / sum(self.p)

    def _read_aois(self):
        aois = []
        # read aois
        for i in self.m.aois:
            a = {}
            a["id"] = i.id
            a["external"] = {
                "area": i.area,
                "urban_land_use": i.urban_land_use,
            }
            a["geo"] = [self.projector(p.x, p.y, inverse=True) for p in i.positions]
            a["shapely"] = geometry.Polygon(
                [
                    p.x,
                    p.y,
                ]
                for p in i.positions
            )
            a["has_driving_gates"] = len(i.driving_gates) > 0
            aois.append(a)

        def get_aoi_catg(urban_land_use: str):
            if urban_land_use in {"R"}:
                return "residential"
            elif urban_land_use in {"B29"}:
                return "business"
            elif urban_land_use in {
                "B1",
                "B",
            }:
                return "commercial"
            elif urban_land_use in {"M"}:
                return "industrial"
            elif urban_land_use in {"S4", "S"}:
                return "transportation"
            elif urban_land_use in {"A", "A1"}:
                return "administrative"
            elif urban_land_use in {"A3"}:
                return "education"
            elif urban_land_use in {"A5"}:
                return "medical"
            elif urban_land_use in {"B32", "A4", "A2"}:
                return "sport and cultual"
            elif urban_land_use in {"B31", "G1", "B3", "B13"}:
                return "park and leisure"
            return "other"

        # add catg
        for aoi in aois:
            aoi["external"]["catg"] = get_aoi_catg(aoi["external"]["urban_land_use"])
        # population
        if self.add_pop and self.pop_tif_path is not None:
            raise NotImplementedError(
                "Adding population to AOIs in trip_generator has been removed!"
            )
            # geos = []
            # for aoi_data in aois:
            #     geos.append(
            #         Feature(
            #             geometry=Polygon([[list(c) for c in aoi_data["geo"]]]),
            #             properties={
            #                 "id": aoi_data["id"],
            #             },
            #         )
            #     )
            # geos = FeatureCollection(geos)
            # geos = geo2pop(geos, self.pop_tif_path)

            # geos = cast(FeatureCollection, geos)
            # aoi_id2pop = defaultdict(int)
            # for feature in geos["features"]:
            #     aoi_id = feature["properties"]["id"]
            #     pop = feature["properties"]["population"]
            #     aoi_id2pop[aoi_id] = pop
            # for aoi in aois:
            #     aoi_id = aoi["id"]
            #     aoi["external"]["population"] = aoi_id2pop.get(aoi_id, 0)
        else:
            for aoi in aois:
                aoi["external"]["population"] = aoi["external"]["area"]
        self.aois = aois

    def _read_regions(self):
        self.regions = []
        for i, poly in enumerate(self.areas.geometry.to_crs(self.m.header.projection)):
            self.area_shapes.append(poly)
            r = {
                "geometry": geo_coords(poly),  # xy coords
                "ori_id": i,
                "region_id": i,
            }
            self.regions.append(r)

    def _read_od_matrix(self):
        logging.info("Reading original ods")
        n_region = len(self.regions)
        assert (
            n_region == self.od_matrix.shape[0] and n_region == self.od_matrix.shape[1]
        )
        # OD-matrix contains time axis = 2
        if len(self.od_matrix.shape) > 2:
            od = self.od_matrix
            self.LEN_OD_TIMES = od.shape[2]
        else:
            orig_od = np.expand_dims(self.od_matrix, axis=2)
            self.LEN_OD_TIMES = 1
            od = np.broadcast_to(
                orig_od, (n_region, n_region, self.LEN_OD_TIMES)
            ).astype(np.int64)
        sum_od_j = np.sum(od, axis=1)
        od_prob = od / sum_od_j[:, np.newaxis]
        od_prob = np.nan_to_num(od_prob)
        self.od = od
        self.od_prob = od_prob

    def _match_aoi2region(self):
        aois = self.aois
        results = []
        _match_aoi_unit_with_arg = partial(_match_aoi_unit, (self.area_shapes,))
        for i in range(0, len(aois), MAX_BATCH_SIZE):
            aois_batch = aois[i : i + MAX_BATCH_SIZE]
            with Pool(processes=self.workers) as pool:
                results += pool.map(
                    _match_aoi_unit_with_arg,
                    aois_batch,
                    chunksize=min(
                        ceil(len(aois_batch) / self.workers), self.max_chunk_size
                    ),
                )
        results = [r for r in results if r is not None]
        for r in results:
            aoi_id, reg_id = r[:2]
            self.aoi2region[aoi_id] = reg_id
            self.region2aoi[reg_id].append(aoi_id)
        logging.info(f"AOI matched: {len(self.aoi2region)}")

    def _generate_mobi(
        self,
        agent_num: int = 10000,
        area_pops: Optional[list] = None,
        person_profiles: Optional[list] = None,
        seed: int = 0,
        max_chunk_size: int = 500,
    ):
        available_trip_modes = self.available_trip_modes
        if "bus" in available_trip_modes and "subway" in available_trip_modes:
            available_trip_modes.append("bus_subway")
        get_mode_partial_args = (
            available_trip_modes,
            self._trip_mode_cost_partial_args,
        )
        region2aoi = self.region2aoi
        aoi_map = {d["id"]: d for d in self.aois}
        n_region = len(self.regions)
        home_dist, work_od, educate_od, other_od = extract_HWEO_from_od_matrix(
            self.aois,
            n_region,
            self.aoi2region,
            self.aoi_type2ids,
            self.od_prob,
            self.LEN_OD_TIMES,
        )
        aoi_type2ids = self.aoi_type2ids
        agent_args = []
        a_home_regions = []
        if area_pops is not None:
            for ii, pop in enumerate(area_pops):
                pop_num = int(pop)
                if pop_num > 0:
                    a_home_regions += [ii for _ in range(pop_num)]
            agent_num = sum(a_home_regions)
        else:
            a_home_regions = [None for _ in range(agent_num)]
        rng = np.random.default_rng(seed)
        if person_profiles is not None:
            a_profiles = person_profiles
        else:
            a_profiles = gen_profiles(agent_num, self.workers, max_chunk_size)
        a_modes = [self.modes for _ in range(agent_num)]
        a_p_modes = [self.p_mode for _ in range(agent_num)]
        for i, (a_home_region, a_profile, a_mode, a_p_mode) in enumerate(
            zip(a_home_regions, a_profiles, a_modes, a_p_modes)
        ):
            agent_args.append(
                (
                    i,
                    rng.integers(0, 2**16 - 1),
                    a_home_region,
                    a_profile,
                    a_mode,
                    a_p_mode,
                )
            )
        partial_args = (
            home_dist,
            work_od,
            educate_od,
            other_od,
            region2aoi,
            aoi_map,
            aoi_type2ids,
            len(self.regions),
            self.projector,
            self.departure_prob,
            self.LEN_OD_TIMES,
        )
        _process_agent_unit_with_arg = partial(
            _process_agent_unit, (partial_args, get_mode_partial_args)  # type:ignore
        )
        raw_persons = []
        for i in range(0, len(agent_args), MAX_BATCH_SIZE):
            agent_args_batch = agent_args[i : i + MAX_BATCH_SIZE]
            with Pool(processes=self.workers) as pool:
                raw_persons += pool.map(
                    _process_agent_unit_with_arg,
                    agent_args_batch,
                    chunksize=min(
                        ceil(len(agent_args_batch) / self.workers), max_chunk_size
                    ),
                )
        raw_persons = [r for r in raw_persons]
        for agent_id, (
            aoi_list,
            person_home,
            person_work,
            a_profile,
            times,
            trip_modes,
            trip_models,
            activities,
        ) in enumerate(raw_persons):
            times = np.array(times) * 3600  # hour->second
            p = Person()
            p.CopyFrom(self.template())
            p.ClearField("schedules")
            p.id = agent_id
            p.home.CopyFrom(Position(aoi_position=AoiPosition(aoi_id=person_home)))
            p.work.CopyFrom(Position(aoi_position=AoiPosition(aoi_id=person_work)))
            p.profile.CopyFrom(dict2pb(a_profile, PersonProfile()))
            for time, aoi_id, trip_mode, activity, trip_model in zip(
                times, aoi_list[1:], trip_modes, activities, trip_models
            ):
                schedule = cast(Schedule, p.schedules.add())
                schedule.departure_time = time
                schedule.loop_count = 1
                trip = Trip(
                    mode=cast(
                        TripMode,
                        trip_mode,
                    ),
                    end=Position(aoi_position=AoiPosition(aoi_id=aoi_id)),
                    activity=activity,
                    model=trip_model,
                )
                schedule.trips.append(trip)
            self.persons.append(p)

    def generate_persons(
        self,
        od_matrix: np.ndarray,
        areas: GeoDataFrame,
        available_trip_modes: list[str] = ["drive", "walk", "bus", "subway", "taxi"],
        departure_time_curve: Optional[list[float]] = None,
        area_pops: Optional[list] = None,
        person_profiles: Optional[list[dict]] = None,
        seed: int = 0,
        agent_num: Optional[int] = None,
    ) -> list[Person]:
        """
        Args:
        - od_matrix (numpy.ndarray): The OD matrix.
        - areas (GeoDataFrame): The area data. Must contain a 'geometry' column with geometric information and a defined `crs` string.
        - available_trip_modes (list[str]): available trip modes for person schedules.
        - departure_time_curve (Optional[list[float]]): The departure time of a day (24h). The resolution must >=1h.
        - area_pops (list): list of populations in each area. If is not None, # of the persons departs from each home position is exactly equal to the given pop num.
        - person_profiles (Optional[list[dict]]): list of profiles in dict format.
        - seed (int): The random seed.
        - agent_num (int): number of agents to generate.

        Returns:
        - list[Person]: The generated person objects.
        """
        # init
        self.area_shapes = []
        self.aoi2region = defaultdict(list)
        self.region2aoi: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self.aoi_type2ids: dict[str, list[int]] = defaultdict(list)
        self.persons = []
        # user input time curve
        if departure_time_curve is not None:
            assert len(departure_time_curve) >= 24
            sum_times = sum(departure_time_curve)
            self.departure_prob = np.array(
                [d / sum_times for d in departure_time_curve]
            )
        else:
            self.departure_prob = None
        self.od_matrix = od_matrix
        self.areas = areas
        self.available_trip_modes = available_trip_modes
        self._read_aois()
        self._read_regions()
        self._read_od_matrix()
        self._match_aoi2region()
        if agent_num is None:
            agent_num = int(np.sum(self.od) / self.LEN_OD_TIMES)
        if not agent_num >= 1:
            logging.warning("agent_num should >=1")
            return []
        self._generate_mobi(
            agent_num, area_pops, person_profiles, seed, self.max_chunk_size
        )
        return self.persons

    def _get_driving_pos_dict(self) -> dict[tuple[int, int], LanePosition]:
        road_aoi_id2d_pos = {}
        road_id2d_lane_ids = {}
        lane_id2parent_road_id = {}
        m_lanes = {l.id: l for l in self.m.lanes}
        m_roads = {r.id: r for r in self.m.roads}
        m_aois = {a.id: a for a in self.m.aois}
        for road_id, road in m_roads.items():
            d_lane_ids = [
                lid
                for lid in road.lane_ids
                if m_lanes[lid].type == mapv2.LANE_TYPE_DRIVING
            ]
            road_id2d_lane_ids[road_id] = d_lane_ids
        for lane_id, lane in m_lanes.items():
            parent_id = lane.parent_id
            # road lane
            if parent_id in m_roads:
                lane_id2parent_road_id[lane_id] = parent_id
        for aoi_id, aoi in m_aois.items():
            for d_pos in aoi.driving_positions:
                pos_lane_id, _ = d_pos.lane_id, d_pos.s
                # junction lane
                assert (
                    pos_lane_id in lane_id2parent_road_id
                ), f"Bad lane position {d_pos} at AOI {aoi_id}"
                parent_road_id = lane_id2parent_road_id[pos_lane_id]
                road_aoi_key = (parent_road_id, aoi_id)
                road_aoi_id2d_pos[road_aoi_key] = d_pos
        return road_aoi_id2d_pos

    def generate_public_transport_drivers(
        self,
        template_func: Optional[Callable[[], Person]] = None,
        stop_duration_time: float = 30.0,
        seed: int = 0,
    ) -> list[Person]:
        """
        Args:
        - template_func (Optional[Callable[[],Person]]): The template function of generated person object, whose `schedules`, `home` will be replaced and others will be copied. If not provided, the `temp_func` provided in `__init__` will be utilized.
        - stop_duration_time (float): The duration time (in second) for bus at each stop.
        - seed (int): The random seed.

        Returns:
        - list[Person]: The generated driver objects.
        """
        self.persons = []
        road_aoi_id2d_pos = self._get_driving_pos_dict()
        person_id = PT_START_ID
        _template = template_func if template_func is not None else self.template
        for sl in self.m.sublines:
            departure_times = list(sl.schedules.departure_times)
            if not sl.type in {mapv2.SUBLINE_TYPE_BUS, mapv2.SUBLINE_TYPE_SUBWAY}:
                continue
            person_id, generated_drivers = gen_bus_drivers(
                person_id,
                _template,
                departure_times,
                stop_duration_time,
                road_aoi_id2d_pos,
                sl,
            )
            self.persons.extend(generated_drivers)
        return self.persons

    def _generate_schedules(self, input_persons: list[Person], seed: int):
        available_trip_modes = self.available_trip_modes
        if "bus" in available_trip_modes and "subway" in available_trip_modes:
            available_trip_modes.append("bus_subway")
        region2aoi = self.region2aoi
        aoi_map = {d["id"]: d for d in self.aois}
        n_region = len(self.regions)
        home_dist, work_od, educate_od, other_od = extract_HWEO_from_od_matrix(
            self.aois,
            n_region,
            self.aoi2region,
            self.aoi_type2ids,
            self.od_prob,
            self.LEN_OD_TIMES,
        )
        orig_persons = deepcopy(input_persons)
        person_args = []
        bad_person_indexes = set()
        rng = np.random.default_rng(seed)
        for idx, p in enumerate(orig_persons):
            try:
                p_home = p.home.aoi_position.aoi_id
                if not p_home >= AOI_START_ID:
                    logging.warning(
                        f"Person {p.id} has no home AOI ID, use random home instead!"
                    )
                    p_home = rng.choice(self.aoi_type2ids["home"])
                p_work = p.work.aoi_position.aoi_id
                if not p_work >= AOI_START_ID:
                    logging.warning(
                        f"Person {p.id} has no work AOI ID, use random work instead!"
                    )
                    p_work = rng.choice(self.aoi_type2ids["other"])
                p_profile = pb2dict(p.profile)
                person_args.append(
                    [
                        p.id,
                        p_home,
                        self.aoi2region[p_home],  # possibly key error
                        p_work,
                        self.aoi2region[p_work],  # possibly key error
                        p_profile,
                        self.modes,
                        self.p_mode,
                        rng.integers(0, 2**16 - 1),
                    ]
                )
            except Exception as e:
                bad_person_indexes.add(idx)
                logging.warning(f"{e} when handling Person {p.id}, Skip!")
        to_process_persons = [
            p for idx, p in enumerate(orig_persons) if idx not in bad_person_indexes
        ]
        # return directly
        no_process_persons = [
            p for idx, p in enumerate(orig_persons) if idx in bad_person_indexes
        ]
        get_mode_partial_args = (
            available_trip_modes,
            self._trip_mode_cost_partial_args,
        )
        partial_args = (
            other_od,
            region2aoi,
            aoi_map,
            self.aoi_type2ids,
            len(self.regions),
            self.projector,
            self.departure_prob,
            self.LEN_OD_TIMES,
        )
        filled_schedules = []
        _fill_person_schedule_unit_with_arg = partial(
            _fill_person_schedule_unit,
            (partial_args, get_mode_partial_args),  # type:ignore
        )

        for i in range(0, len(person_args), MAX_BATCH_SIZE):
            person_args_batch = person_args[i : i + MAX_BATCH_SIZE]
            with Pool(processes=self.workers) as pool:
                filled_schedules += pool.map(
                    _fill_person_schedule_unit_with_arg,
                    person_args_batch,
                    chunksize=min(
                        ceil(len(person_args_batch) / self.workers), self.max_chunk_size
                    ),
                )
        for (
            aoi_list,
            times,
            trip_modes,
            trip_models,
            activities,
        ), orig_p in zip(filled_schedules, to_process_persons):
            times = np.array(times) * 3600  # hour->second
            p = Person()
            p.CopyFrom(orig_p)
            p.ClearField("schedules")
            for time, aoi_id, trip_mode, activity, trip_model in zip(
                times, aoi_list[1:], trip_modes, activities, trip_models
            ):
                schedule = cast(Schedule, p.schedules.add())
                schedule.departure_time = time
                schedule.loop_count = 1
                trip = Trip(
                    mode=cast(
                        TripMode,
                        trip_mode,
                    ),
                    end=Position(aoi_position=AoiPosition(aoi_id=aoi_id)),
                    activity=activity,
                    model=trip_model,
                )
                schedule.trips.append(trip)
            self.persons.append(p)
        len_filled_persons, len_no_process_persons = len(to_process_persons), len(
            no_process_persons
        )
        logging.info(f"Filled schedules of {len_filled_persons} persons")
        if len_no_process_persons > 0:
            logging.warning(
                f"Unprocessed persons: {len_no_process_persons}, index range [{len_filled_persons}:] in returned results"
            )
        self.persons.extend(no_process_persons)

    def fill_person_schedules(
        self,
        input_persons: list[Person],
        od_matrix: np.ndarray,
        areas: GeoDataFrame,
        available_trip_modes: list[str] = ["drive", "walk", "bus", "subway", "taxi"],
        departure_time_curve: Optional[list[float]] = None,
        seed: int = 0,
    ) -> list[Person]:
        """
        Generate person schedules.

        Args:
        - input_persons (list[Person]): Input Person objects.
        - od_matrix (numpy.ndarray): The OD matrix.
        - areas (GeoDataFrame): The area data. Must contain a 'geometry' column with geometric information and a defined `crs` string.
        - available_trip_modes (Optional[list[str]]): available trip modes for person schedules.
        - departure_time_curve (Optional[list[float]]): The departure time of a day (24h). The resolution must >=1h.
        - seed (int): The random seed.

        Returns:
        - list[Person]: The person objects with generated schedules.
        """
        # init
        self.area_shapes = []
        self.aoi2region = defaultdict(list)
        self.region2aoi = defaultdict(list)
        self.aoi_type2ids = defaultdict(list)
        self.persons = []
        # user input time curve
        if departure_time_curve is not None:
            assert len(departure_time_curve) >= 24
            sum_times = sum(departure_time_curve)
            self.departure_prob = np.array(
                [d / sum_times for d in departure_time_curve]
            )
        else:
            self.departure_prob = None
        self.od_matrix = od_matrix
        self.areas = areas
        self.available_trip_modes = available_trip_modes
        self._read_aois()
        self._read_regions()
        self._read_od_matrix()
        self._match_aoi2region()
        self._generate_schedules(input_persons, seed)

        return self.persons

    def generate_taxi_drivers(
        self,
        template_func: Optional[Callable[[], Person]] = None,
        parking_positions: Optional[list[Union[LanePosition, AoiPosition]]] = None,
        agent_num: Optional[int] = None,
        seed: int = 0,
    ) -> list[Person]:
        """
        Args:
        - template_func (Optional[Callable[[],Person]]): The template function of generated person object, whose `schedules`, `home` will be replaced and others will be copied. If not provided, the `temp_func` provided in `__init__` will be utilized.
        - parking_positions (Optional[list[Union[LanePosition,AoiPosition]]]): The parking positions of each taxi.
        - agent_num (Optional[int]): The taxi driver num.
        - seed (int): The random seed.

        Returns:
        - list[Person]: The generated driver objects.
        """
        self.persons = []
        person_id = TAXI_START_ID
        _template = template_func if template_func is not None else self.template
        self._read_aois()
        if parking_positions is not None:
            logging.info(f"")
            _taxi_home_positions = parking_positions
        elif agent_num is not None:
            logging.info(f"")
            if not agent_num >= 1:
                logging.warning("agent_num should >=1")
                return []
            rng = np.random.default_rng(seed)
            has_driving_aoi_ids = [a["id"] for a in self.aois if a["has_driving_gates"]]
            _taxi_home_positions = [
                AoiPosition(aoi_id=_id)
                for _id in rng.choice(has_driving_aoi_ids, int(agent_num))
            ]
        else:
            logging.warning(
                "Either `agent_num` or `parking_positions` should be provided!"
            )
            return []
        for _pos in _taxi_home_positions:
            person_id, generated_drivers = gen_taxi_drivers(
                person_id,
                _template,
                _pos,
            )
            self.persons.extend(generated_drivers)
        return self.persons
