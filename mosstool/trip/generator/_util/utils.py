from multiprocessing import Pool, cpu_count
from typing import (Callable, Dict, List, Literal, Optional, Set, Tuple, Union,
                    cast)

import numpy as np
from pycityproto.city.person.v2.person_pb2 import (BusAttribute, BusType,
                                                   Person, PersonAttribute,
                                                   VehicleAttribute)
from pycityproto.city.routing.v2.routing_pb2 import (DrivingJourneyBody,
                                                     Journey, JourneyType)
from pycityproto.city.trip.v2.trip_pb2 import TripStop

from ....map._map_util.const import *
from ....type import (AoiPosition, LanePosition, Position, Schedule, Trip,
                      TripMode)
from ....util.format_converter import dict2pb, pb2dict
from .const import *

__all__ = [
    "is_walking",
    "gen_profiles",
    "recalculate_trip_mode_prob",
    "gen_bus_drivers",
]


def is_walking(trip_mode: TripMode) -> bool:
    """
    Determine if it is walking mode

    Args:
    - trip_mode (TripMode): mode.

    Returns:
    - bool: Whether it is walking mode.
    """
    return trip_mode in (
        TripMode.TRIP_MODE_BIKE_WALK,
        TripMode.TRIP_MODE_WALK_ONLY,
        TripMode.TRIP_MODE_BUS_WALK,
    )


def _in_range(a, l, u):
    return l - 1e-2 < a < u - 1e-2


def _suitable_profile(age, edu_level):
    ## rules for filtering abnormal profile
    suitable = True
    if _in_range(age, 0, 15):
        if edu_level in {HIGH_SCHOOL, COLLEGE, BACHELOR, MASTER, DOCTOR}:
            suitable = False
    elif _in_range(age, 15, 18):
        if edu_level in {COLLEGE, BACHELOR, MASTER, DOCTOR}:
            suitable = False
    return suitable


def _gen_profile_unit(seed: int):
    rng = np.random.default_rng(seed)
    age = rng.choice(AGES, p=AGE_STATS)
    gender = rng.choice(GENDERS, p=GENDER_STATS)
    consumption = rng.choice(CONSUMPTION_LEVELS, p=CONSUMPTION_STATS)
    edu_prob = []
    for edu_level, prob in zip(EDUCATION_LEVELS, EDUCATION_STATS):
        if _suitable_profile(age, edu_level):
            edu_prob.append(prob)
        else:
            edu_prob.append(0.0)
    edu_prob = np.array(edu_prob) / sum(edu_prob)
    education = rng.choice(EDUCATION_LEVELS, p=edu_prob)
    return {
        "age": age,
        "education": education,
        "gender": gender,
        "consumption": consumption,
    }


def gen_profiles(
    agent_num: int, workers: int, multiprocessing_chunk_size: int
) -> List[Dict]:
    """
    Randomly generate PersonProfile

    Args:
    - agent_num (int): number of agents to generate.
    - workers (int): number of workers.

    Returns:
    - list(dict): List of PersonProfile dict.
    """
    profiles = []
    MAX_CHUNK_SIZE = multiprocessing_chunk_size
    profile_args = [i for i in range(agent_num)]
    for i in range(0, len(profile_args), MAX_BATCH_SIZE):
        profile_batch = profile_args[i : i + MAX_BATCH_SIZE]
        with Pool(processes=workers) as pool:
            profiles += pool.map(
                _gen_profile_unit,
                profile_batch,
                chunksize=min(len(profile_batch) // workers, MAX_CHUNK_SIZE),
            )
    return profiles


def recalculate_trip_mode_prob(profile: dict, V: np.ndarray):
    """
    Filter some invalid trip modes according to the PersonProfile
    """
    return V


def gen_departure_times(
    rng: np.random.Generator, n_trip: int, mode: str, departure_prob
):
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
    return np.array(times)


def extract_HWEO_from_od_matrix(
    aois: list,
    n_region: int,
    aoi2region: dict,
    aoi_type2ids: dict,
    od_prob: np.ndarray,
    od_times_length: int,
):
    total_popu = np.zeros(n_region)
    work_popu = np.zeros(n_region)
    educate_popu = np.zeros(n_region)
    home_popu = np.zeros(n_region)
    LEN_OD_TIMES = od_times_length
    for aoi in aois:
        # aoi type identify
        external = aoi["external"]
        aoi_id = aoi["id"]
        if external["catg"] in HOME_CATGS:
            aoi_type2ids["home"].append(aoi_id)
        elif external["catg"] in WORK_CATGS:
            aoi_type2ids["work"].append(aoi_id)
        elif external["catg"] in EDUCATION_CATGS:
            aoi_type2ids["education"].append(aoi_id)
        else:
            aoi_type2ids["other"].append(aoi_id)
        # pop calculation
        if aoi_id not in aoi2region:
            continue
        reg_idx = aoi2region[aoi_id]
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
    work_od = np.zeros((n_region, n_region, LEN_OD_TIMES))
    educate_od = np.zeros((n_region, n_region, LEN_OD_TIMES))
    other_od = np.zeros((n_region, n_region, LEN_OD_TIMES))
    # calculate frequency
    ## work
    work_od[:, :, :] = od_prob[:, :, :] * work_popu[:, None] / total_popu[:, None]
    work_od[:, total_popu <= 0, :] = 0
    ## study
    educate_od[:, :, :] = od_prob[:, :, :] * educate_popu[:, None] / total_popu[:, None]
    educate_od[:, total_popu <= 0, :] = 0
    ## other
    other_od[:, :, :] = (
        od_prob[:, :, :]
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
    return home_dist, work_od, educate_od, other_od


def gen_bus_drivers(
    person_id: int,
    person_template_generator: Callable[[], Person],
    depart_times: List[float],
    stop_duration_time: float,
    road_aoi_id2d_pos: Dict[Tuple[int, int], geov2.LanePosition],
    subline,
) -> Tuple[int, List[Person]]:
    def _transfer_conn_road_ids(
        station_connection_road_ids: List[List[int]],
    ) -> List[int]:
        assert (
            len(station_connection_road_ids) > 0
            and len(station_connection_road_ids[0]) > 0
        ), f"Bad conn_road_ids {station_connection_road_ids}"
        route_road_ids = []
        for next_road_ids in station_connection_road_ids:
            if len(route_road_ids) > 0 and route_road_ids[-1] == next_road_ids[0]:
                route_road_ids += next_road_ids[1:]
            else:
                route_road_ids += next_road_ids
        return route_road_ids

    def _aoi_road_ids(station_connection_road_ids) -> List[int]:
        cur_road_ids = [rids[0] for rids in station_connection_road_ids]
        # road that connect to the end station
        cur_road_ids.append(station_connection_road_ids[-1][-1])
        return cur_road_ids

    sl_id = subline.id
    # basic attributes
    sl_type = subline.type
    bus_type = BusType.BUS_TYPE_UNSPECIFIED
    if sl_type == mapv2.SUBLINE_TYPE_BUS:
        sl_capacity = STATION_CAPACITY["BUS"]
        bus_type = BusType.BUS_TYPE_BUS
    elif sl_type == mapv2.SUBLINE_TYPE_SUBWAY:
        sl_capacity = STATION_CAPACITY["SUBWAY"]
        bus_type = BusType.BUS_TYPE_SUBWAY
    elif sl_type == mapv2.SUBLINE_TYPE_UNSPECIFIED:
        sl_capacity = STATION_CAPACITY["UNSPECIFIED"]
        bus_type = BusType.BUS_TYPE_UNSPECIFIED
    else:
        raise ValueError(f"Bad Subline Type {sl_type}")
    # road ids
    station_connection_road_ids = [
        [rid for rid in rids.road_ids] for rids in subline.station_connection_road_ids
    ]
    route_road_ids = _transfer_conn_road_ids(station_connection_road_ids)
    trip_stop_road_ids = _aoi_road_ids(
        station_connection_road_ids
    )  # stop road ids during the trip
    # trip stops
    aoi_ids = list(subline.aoi_ids)
    home_aoi_id, end_aoi_id = aoi_ids[0], aoi_ids[-1]
    trip_stop_aoi_ids = aoi_ids  # stop aoi ids during the trip
    trip_stop_lane_id_s = []
    for cur_road_id, cur_aoi_id in zip(trip_stop_road_ids, trip_stop_aoi_ids):
        road_aoi_key = (cur_road_id, cur_aoi_id)
        if road_aoi_key not in road_aoi_id2d_pos:
            raise ValueError(f"bad road and AOI pair {road_aoi_key}")
        d_pos = road_aoi_id2d_pos[road_aoi_key]
        d_lane_id, d_s = d_pos.lane_id, d_pos.s
        trip_stop_lane_id_s.append((d_lane_id, d_s))
    assert len(trip_stop_lane_id_s) == len(
        trip_stop_aoi_ids
    ), f"Bad PublicTransport Route at {aoi_ids}"
    p_trip_stops = []
    # bus attribute
    p_bus_attr = BusAttribute(
        subline_id=sl_id, capacity=sl_capacity, type=bus_type, model=""
    )
    for (d_lane_id, d_s), aoi_id in zip(trip_stop_lane_id_s, trip_stop_aoi_ids):
        trip_stop = TripStop()
        trip_stop.lane_position.CopyFrom(LanePosition(lane_id=d_lane_id, s=d_s))
        trip_stop.aoi_position.CopyFrom(AoiPosition(aoi_id=aoi_id))
        trip_stop.duration = stop_duration_time
        p_trip_stops.append(trip_stop)
    # eta for bus journey
    bus_eta = sum(subline.schedules.offset_times)
    sl_drivers = []
    if bus_type == BusType.BUS_TYPE_BUS:
        for tm in depart_times:
            p = Person()
            p.CopyFrom(person_template_generator())
            p.id = person_id
            p.bus_attribute.CopyFrom(p_bus_attr)
            p.home.CopyFrom(Position(aoi_position=AoiPosition(aoi_id=home_aoi_id)))
            schedule = cast(Schedule, p.schedules.add())
            schedule.departure_time = tm
            schedule.loop_count = 1
            trip = Trip(
                mode=cast(
                    TripMode,
                    CAR,
                ),
                end=Position(aoi_position=AoiPosition(aoi_id=end_aoi_id)),
                activity="",
                model="",
                trip_stops=p_trip_stops,
                routes=[
                    Journey(
                        driving=DrivingJourneyBody(
                            road_ids=route_road_ids, eta=bus_eta
                        ),
                        type=JourneyType.JOURNEY_TYPE_DRIVING,
                    )
                ],
            )
            schedule.trips.append(trip)
            person_id += 1
            sl_drivers.append(p)
    elif bus_type == BusType.BUS_TYPE_SUBWAY:
        # empty schedule
        p = Person()
        p.CopyFrom(person_template_generator())
        p.id = person_id
        p.bus_attribute.CopyFrom(p_bus_attr)
        p.home.CopyFrom(Position(aoi_position=AoiPosition(aoi_id=home_aoi_id)))
        person_id += 1

    return (person_id, sl_drivers)
