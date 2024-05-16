from enum import Enum
from typing import List, Optional, Tuple, Union, cast

import numpy as np
from pycityproto.city.geo.v2.geo_pb2 import AoiPosition, LanePosition, Position
from pycityproto.city.map.v2.map_pb2 import Aoi, Lane, LaneType, Map
from pycityproto.city.person.v1.person_pb2 import Person
from pycityproto.city.trip.v2.trip_pb2 import Schedule, Trip, TripMode

from ._util.utils import is_walking
from .template import DEFAULT_PERSON
from ...map import JUNC_START_ID

__all__ = ["PositionMode", "RandomGenerator"]


class PositionMode(Enum):
    AOI = 0
    LANE = 1


class RandomGenerator:
    """
    Random trip generator
    """

    def __init__(
        self,
        m: Map,
        position_modes: List[PositionMode],
        trip_mode: TripMode,
        template: Person = DEFAULT_PERSON,
    ):
        """
        Args:
        - m (Map): The Map.
        - position_modes (List[PositionMode]): The schedules generated will follow the position modes in this list. For example, if the list is [PositionMode.AOI, PositionMode.LANE, PositionMode.AOI], the generated person will start from an AOI, then go to a lane, and finally go to an AOI.
        - trip_mode (TripMode): The target trip mode.
        - template (Person): The template of generated person object, whose `schedules`, `home` will be replaced and others will be copied.
        """
        self.m = m
        self.modes = position_modes
        if len(self.modes) <= 1:
            raise ValueError("position_modes should have at least 2 elements")
        self.trip_mode = trip_mode
        self.template = template
        self.template.ClearField("schedules")
        self.template.ClearField("home")

        # Pre-process the map to build the randomly selected candidate set
        walk = is_walking(trip_mode)
        self._aoi_candidates = [
            aoi
            for aoi in self.m.aois
            if len(aoi.walking_positions if walk else aoi.driving_positions) > 0
        ]
        self._lane_candidates = [
            lane
            for lane in self.m.lanes
            if lane.parent_id < JUNC_START_ID
            and lane.type
            == (LaneType.LANE_TYPE_WALKING if walk else LaneType.LANE_TYPE_DRIVING)
        ]
        if PositionMode.AOI in position_modes and len(self._aoi_candidates) == 0:
            raise ValueError("No available AOI")
        if PositionMode.LANE in position_modes and len(self._lane_candidates) == 0:
            raise ValueError("No available lane")

    def _rand_position(self, candidates: Union[List[Aoi], List[Lane]]):
        index = np.random.randint(0, len(candidates))
        candidate = candidates[index]
        if isinstance(candidate, Aoi):
            return Position(aoi_position=AoiPosition(aoi_id=candidate.id))
        else:
            return Position(
                lane_position=LanePosition(
                    lane_id=candidate.id, s=np.random.rand() * candidate.length
                )
            )

    def uniform(
        self,
        num: int,
        first_departure_time_range: Tuple[float, float],
        schedule_interval_range: Tuple[float, float],
        seed: Optional[int] = None,
        start_id: Optional[int] = None,
    ) -> List[Person]:
        """
        Generate a person object by uniform random sampling

        Args:
        - num (int): The number of person objects to generate.
        - first_departure_time_range (Tuple[float, float]): The range of the first departure time (uniform random sampling).
        - schedule_interval_range (Tuple[float, float]): The range of the interval between schedules (uniform random sampling).
        - seed (Optional[int], optional): The random seed. Defaults to None.
        - start_id (Optional[int], optional): The start id of the generated person objects. Defaults to None. If None, the `id` will be NOT set.

        Returns:
        - List[Person]: The generated person objects.
        """
        if seed is not None:
            np.random.seed(seed)
        persons = []
        for i in range(num):
            p = Person()
            p.CopyFrom(self.template)
            if start_id is not None:
                p.id = start_id + i
            p.home.CopyFrom(
                self._rand_position(
                    self._aoi_candidates
                    if self.modes[0] == PositionMode.AOI
                    else self._lane_candidates
                )
            )
            departure_time = np.random.uniform(*first_departure_time_range)
            for mode in self.modes[1:]:
                schedule = cast(Schedule, p.schedules.add())
                schedule.departure_time = departure_time
                schedule.loop_count = 1
                trip = Trip(
                    mode=self.trip_mode,
                    end=self._rand_position(
                        self._aoi_candidates
                        if mode == PositionMode.AOI
                        else self._lane_candidates
                    ),
                )
                schedule.trips.append(trip)
                departure_time += np.random.uniform(*schedule_interval_range)
            persons.append(p)
        return persons
