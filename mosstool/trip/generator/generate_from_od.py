import logging
from collections import defaultdict
from functools import partial
from math import ceil
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Literal, Optional, Set, Tuple, Union, cast

import numpy as np
import pyproj
import shapely.geometry as geometry
from geojson import Feature, FeatureCollection, Polygon
from geopandas.geodataframe import GeoDataFrame
from pycityproto.city.geo.v2.geo_pb2 import AoiPosition, Position
from pycityproto.city.map.v2.map_pb2 import Map
from pycityproto.city.person.v1.person_pb2 import (
    Consumption,
    Education,
    Gender,
    Person,
    PersonProfile,
)
from pycityproto.city.trip.v2.trip_pb2 import Schedule, Trip, TripMode

from ...map._map_util.aoiutils import geo_coords
from ...map._map_util.const import *
from ...util.format_converter import dict2pb
from ...util.geo_match_pop import geo2pop
from ._util.const import *
from ._util.utils import gen_profiles, recalculate_trip_mode_prob
from .template import DEFAULT_PERSON


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
    p1: Tuple[float, float], p2: Tuple[float, float], profile: dict, seed: int = 0
):
    (x1, y1), (x2, y2) = p1, p2
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    subway_expense = SUBWAY_EXPENSE
    bus_expense = BUS_EXPENSE
    driving_duration = distance / DRIVING_SPEED + DRIVING_PENALTY
    subway_duration = distance / SUBWAY_SPEED + SUBWAY_PENALTY
    bus_duration = distance / BUS_SPEED + BUS_PENALTY
    bicycle_duration = distance / BIKE_SPEED + BIKE_PENALTY
    parking_fee = 20
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
    V = recalculate_trip_mode_prob(profile, V)
    V = V / sum(V)
    rng = np.random.default_rng(seed)
    choice_index = rng.choice(len(V), p=V)
    return ALL_TRIP_MODES[choice_index]


def _match_aoi_unit(projector, aoi):
    global shapes
    p = aoi["shapely"].centroid
    for id, j in enumerate(shapes):
        if j.contains(p):
            return (aoi["id"], id)
    return None


def _generate_unit(H, W, E, O, a_home_region, a_profile, modes, p_mode, seed, args):
    # Three steps
    # 1.determine person activity mode
    # 2.determine departure times
    # 3.generate trip according to OD-matrix
    global region2aoi, aoi_map, aoi_type2ids
    n_region, projector, departure_prob, LEN_OD_TIMES = args
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
                popu[i] = aoi_map[id]["external"]["population"]
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
    mode = rng.choice(len(modes), p=p_mode)
    mode = modes[mode]
    aoi_list = []
    now_region = home_region
    # arrange time   times[i]: departure time from region i
    n_trip = len(mode) - 1 if mode[-1] == "H" else len(mode)
    times = [0.0 for _ in range(n_trip)]
    if departure_prob is not None:
        LEN_TIMES = len(departure_prob)
        TIME_INTERVAL = 24 / LEN_TIMES  # (hour)
        for i in range(n_trip):
            time_center = (
                rng.choice(range(LEN_TIMES), p=departure_prob)
            ) * TIME_INTERVAL
            times[i] = rng.uniform(
                time_center - 0.5 * TIME_INTERVAL, time_center + 0.5 * TIME_INTERVAL
            )
    else:
        # Defaults
        ## Go to work
        if mode[1] == "W" or mode[1] == "S":
            times[0] = rng.normal(8, 2)
        else:
            times[0] = rng.normal(10.5, 2.5)
        ## Go home
        end_idx = -1
        for i in range(len(mode)):
            if mode[len(mode) - i - 1] == "H":
                if mode[len(mode) - i - 2] == "W" or mode[len(mode) - i - 2] == "S":
                    t = rng.normal(17, 2)
                else:
                    t = rng.normal(14.5, 3)

                times[len(mode) - i - 2] = t
                end_idx = len(mode) - i - 2
                break
        if times[0] > times[end_idx]:
            times[0], times[end_idx] = times[end_idx], times[0]
        ## uniform distribute from work and home
        for i in range(1, end_idx):
            if times[end_idx] - times[0] > 4:
                times[i] = rng.uniform(times[0] + 1, times[end_idx] - 1)
            elif times[end_idx] - times[0] > 2:
                times[i] = rng.uniform(times[0] + 0.5, times[end_idx] - 0.5)
            else:
                times[i] = rng.uniform(times[0], times[end_idx])
        if mode[-1] != "H":
            sleep_time = 22 + rng.exponential(2)
            if sleep_time - times[end_idx] > 3:
                times[-2:] = rng.uniform(times[end_idx] + 0.5, sleep_time - 0.5, size=2)
            elif sleep_time > times[end_idx]:
                times[-2:] = rng.uniform(times[end_idx], sleep_time, size=2)
            else:
                times[-2:] = rng.uniform(sleep_time, times[end_idx], size=2)

    times = np.sort(times)
    times = times % 24

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
        trip_modes.append(_get_mode_with_distribution(p1, p2, a_profile, seed))

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


def _process_agent_unit(args, d):
    global home_dist, work_od, educate_od, other_od
    _, seed, home_region, profile, mode, p_mode = d
    return _generate_unit(
        home_dist,
        work_od,
        educate_od,
        other_od,
        home_region,
        profile,
        mode,
        p_mode,
        seed,
        args,
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
        driving_penalty: float = 0.0,
        subway_speed: float = 35 / 3.6,
        subway_penalty: float = 600.0,
        subway_expense: float = 10.0,
        bus_speed: float = 15 / 3.6,
        bus_penalty: float = 600.0,
        bus_expense: float = 5.0,
        bike_speed: float = 10 / 3.6,
        bike_penalty: float = 0.9,
        template: Person = DEFAULT_PERSON,
        add_pop: bool = False,
        workers: int = cpu_count(),
    ):
        """
        Args:
        - m (Map): The Map.
        - pop_tif_path (str): path to population tif file.
        - activity_distributions (dict): human mobility mode and its probability. e.g. {"HWH": 18.0, "HWH+": 82.0,}. H for go home, W for go to work, O or + for other activities
        - driving_speed (float): vehicle speed(m/s) for traffic mode assignment.
        - driving_penalty (float): extra cost(s) of vehicle for traffic mode assignment.
        - subway_speed (float): subway speed(m/s) for traffic mode assignment.
        - subway_penalty (float): extra cost(s) of subway for traffic mode assignment.
        - subway_expense (float): money  cost(￥) of subway for traffic mode assignment.
        - bus_speed (float): bus speed(m/s) for traffic mode assignment.
        - bus_penalty (float): extra cost(s) of bus for traffic mode assignment.
        - bus_expense (float): money  cost(￥) of bus for traffic mode assignment.
        - bike_speed (float): extra cost(s) of bike for traffic mode assignment.
        - bike_penalty (float): money  cost(￥) of bike for traffic mode assignment.
        - template (Person): The template of generated person object, whose `schedules`, `home` will be replaced and others will be copied.
        - add_pop (bool): Add population to aois.
        - workers (int): number of workers.
        """
        global SUBWAY_EXPENSE, BUS_EXPENSE, DRIVING_SPEED, DRIVING_PENALTY, SUBWAY_SPEED, SUBWAY_PENALTY, BUS_SPEED, BUS_PENALTY, BIKE_SPEED, BIKE_PENALTY
        SUBWAY_EXPENSE, BUS_EXPENSE = subway_expense, bus_expense
        DRIVING_SPEED, DRIVING_PENALTY = driving_speed, driving_penalty
        SUBWAY_SPEED, SUBWAY_PENALTY = subway_speed, subway_penalty
        BUS_SPEED, BUS_PENALTY = bus_speed, bus_penalty
        BIKE_SPEED, BIKE_PENALTY = bike_speed, bike_penalty
        self.m = m
        self.pop_tif_path = pop_tif_path
        self.add_pop = add_pop
        self.projector = pyproj.Proj(m.header.projection)
        self.workers = workers
        self.template = template
        self.template.ClearField("schedules")
        self.template.ClearField("home")
        self.area_shapes = []
        self.aoi2region = defaultdict(list)
        self.region2aoi = defaultdict(list)
        self.aoi_type2ids = defaultdict(list)
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
            geos = []
            for aoi_data in aois:
                geos.append(
                    Feature(
                        geometry=Polygon([[list(c) for c in aoi_data["geo"]]]),
                        properties={
                            "id": aoi_data["id"],
                        },
                    )
                )
            geos = FeatureCollection(geos)
            geos = geo2pop(geos, self.pop_tif_path)

            geos = cast(FeatureCollection, geos)
            aoi_id2pop = defaultdict(int)
            for feature in geos["features"]:
                aoi_id = feature["properties"]["id"]
                pop = feature["properties"]["population"]
                aoi_id2pop[aoi_id] = pop
            for aoi in aois:
                aoi_id = aoi["id"]
                aoi["external"]["population"] = aoi_id2pop.get(aoi_id, 0)
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
        logging.info("reading original ods")
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
        global shapes
        shapes = self.area_shapes
        aois = self.aois
        results = []
        _match_aoi_unit_with_arg = partial(_match_aoi_unit, self.projector)
        for i in range(0, len(aois), MAX_BATCH_SIZE):
            aois_batch = aois[i : i + MAX_BATCH_SIZE]
            with Pool(processes=self.workers) as pool:
                results += pool.map(
                    _match_aoi_unit_with_arg,
                    aois_batch,
                    chunksize=min(ceil(len(aois_batch) / self.workers), 500),
                )
        results = [r for r in results if r is not None]
        for r in results:
            aoi_id, reg_id = r[:2]
            self.aoi2region[aoi_id] = reg_id
            self.region2aoi[reg_id].append(aoi_id)
        logging.info(f"aoi_matched: {len(self.aoi2region)}")

    def _generate_mobi(
        self,
        agent_num: int = 10000,
        area_pops: Optional[list] = None,
        person_profiles: Optional[list] = None,
        seed: int = 0,
    ):
        global region2aoi, aoi_map, aoi_type2ids
        global home_dist, work_od, other_od, educate_od
        region2aoi = self.region2aoi
        aoi_map = {d["id"]: d for d in self.aois}
        n_region = len(self.regions)
        total_popu = np.zeros(n_region)
        work_popu = np.zeros(n_region)
        educate_popu = np.zeros(n_region)
        home_popu = np.zeros(n_region)
        for aoi in self.aois:
            # aoi type identify
            external = aoi["external"]
            aoi_id = aoi["id"]
            if external["catg"] in HOME_CATGS:
                self.aoi_type2ids["home"].append(aoi_id)
            elif external["catg"] in WORK_CATGS:
                self.aoi_type2ids["work"].append(aoi_id)
            elif external["catg"] in EDUCATION_CATGS:
                self.aoi_type2ids["education"].append(aoi_id)
            else:
                self.aoi_type2ids["other"].append(aoi_id)
            # pop calculation
            if aoi_id not in self.aoi2region:
                continue
            reg_idx = self.aoi2region[aoi_id]
            total_popu[reg_idx] += external["population"]
            work_popu[reg_idx] += (
                external["population"] if external["catg"] in WORK_CATGS else 0
            )
            home_popu[reg_idx] += (
                external["population"] if external["catg"] in HOME_CATGS else 0
            )
            educate_popu[reg_idx] += (
                external["population"] if external["catg"] in EDUCATION_CATGS else 0
            )
        home_dist = home_popu / sum(home_popu)
        # initialization
        work_od = np.zeros((n_region, n_region, self.LEN_OD_TIMES))
        educate_od = np.zeros((n_region, n_region, self.LEN_OD_TIMES))
        other_od = np.zeros((n_region, n_region, self.LEN_OD_TIMES))
        # calculate frequency
        ## work
        work_od[:, :, :] = (
            self.od_prob[:, :, :] * work_popu[:, None] / total_popu[:, None]
        )
        work_od[:, total_popu <= 0, :] = 0
        ## study
        educate_od[:, :, :] = (
            self.od_prob[:, :, :] * educate_popu[:, None] / total_popu[:, None]
        )
        educate_od[:, total_popu <= 0, :] = 0
        ## other
        other_od[:, :, :] = (
            self.od_prob[:, :, :]
            * (total_popu[:, None] - work_popu[:, None] - educate_popu[:, None])
            / total_popu[:, None]
        )
        other_od[:, total_popu <= 0, :] = 0
        # to probabilities
        ## work
        sum_work_od_j = np.sum(work_od, axis=1)
        work_od = work_od / sum_work_od_j[:, np.newaxis]
        work_od = np.nan_to_num(work_od)
        work_od[np.where(sum_work_od_j == 0)] = 1
        ## study
        sum_educate_od_j = np.sum(educate_od, axis=1)
        educate_od = educate_od / sum_educate_od_j[:, np.newaxis]
        educate_od = np.nan_to_num(educate_od)
        educate_od[np.where(sum_educate_od_j == 0)] = 1
        ## other
        sum_other_od_j = np.sum(other_od, axis=1)
        other_od = other_od / sum_other_od_j[:, np.newaxis]
        other_od = np.nan_to_num(other_od)
        other_od[np.where(sum_other_od_j == 0)] = 1
        # global variables
        ## home_dist, work_od, educate_od, other_od = home_dist, work_od, educate_od, other_od
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
            a_profiles = gen_profiles(agent_num, self.workers)
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
            len(self.regions),
            self.projector,
            self.departure_prob,
            self.LEN_OD_TIMES,
        )
        _process_agent_unit_with_arg = partial(_process_agent_unit, partial_args)
        raw_persons = []
        for i in range(0, len(agent_args), MAX_BATCH_SIZE):
            agent_args_batch = agent_args[i : i + MAX_BATCH_SIZE]
            with Pool(processes=self.workers) as pool:
                raw_persons += pool.map(
                    _process_agent_unit_with_arg,
                    agent_args_batch,
                    chunksize=min(ceil(len(agent_args_batch) / self.workers), 500),
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
            p.CopyFrom(self.template)
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
        departure_time_curve: Optional[list[float]] = None,
        area_pops: Optional[list] = None,
        person_profiles: Optional[list[dict]] = None,
        seed: int = 0,
        agent_num: Optional[int] = None,
    ) -> List[Person]:
        """
        Args:
        - od_matrix (numpy.ndarray): The OD matrix.
        - areas (GeoDataFrame): The area data.
        - departure_time_curve (list[float]): The departure time of a day (24h). The resolution must >=1h.
        - area_pops (list): list of populations in each area. If is not None, # of the persons departs from each home position is exactly equal to the given pop num.
        - person_profiles (list[dict]): list of profiles in dict format.
        - agent_num (int): number of agents to generate.
        - seed (int): The random seed.

        Returns:
        - List[Person]: The generated person objects.
        """
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
        self._read_aois()
        self._read_regions()
        self._read_od_matrix()
        self._match_aoi2region()
        if agent_num is None:
            agent_num = int(np.sum(self.od) / self.LEN_OD_TIMES)
        if not agent_num >= 1:
            logging.warning("agent_num should >=1")
            return []
        self._generate_mobi(agent_num, area_pops, person_profiles, seed)
        return self.persons
