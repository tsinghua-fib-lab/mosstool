from multiprocessing import Pool, cpu_count
from typing import Dict, List

import numpy as np

from ....map._map_util.const import *
from ....type import Consumption, Education, TripMode
from .const import *

__all__ = [
    "is_walking",
    "gen_profiles",
    "recalculate_trip_mode_prob",
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


def gen_profiles(agent_num: int, workers: int) -> List[Dict]:
    """
    Randomly generate PersonProfile

    Args:
    - agent_num (int): number of agents to generate.
    - workers (int): number of workers.

    Returns:
    - list(dict): List of PersonProfile dict.
    """
    profiles = []
    profile_args = [i for i in range(agent_num)]
    for i in range(0, len(profile_args), MAX_BATCH_SIZE):
        profile_batch = profile_args[i : i + MAX_BATCH_SIZE]
        with Pool(processes=workers) as pool:
            profiles += pool.map(
                _gen_profile_unit,
                profile_batch,
                chunksize=min(len(profile_batch) // workers, 500),
            )
    return profiles


def recalculate_trip_mode_prob(profile: dict, V: np.ndarray):
    """
    Filter some invalid trip modes according to the PersonProfile
    """
    return V
