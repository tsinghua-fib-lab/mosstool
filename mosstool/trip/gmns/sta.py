from copy import copy
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple, cast

import pandas as pd
import path4gmns as pg

from ...map.gmns import Convertor as MapConvertor
from ...type import (
    Map,
    Persons,
    Position,
    Schedule,
    Trip,
    TripMode,
    Journey,
    JourneyType,
    DrivingJourneyBody,
)

__all__ = ["STA"]


class STA:
    """
    run static traffic assignment on GMNS map with persons
    """

    def __init__(self, map: Map, work_dir: str):
        self._map_convertor = MapConvertor(map)
        self._work_dir = work_dir

    def _get_od(self, start: Position, end: Position):
        """
        get origin and destination of a trip

        Args:
        - t: Trip, trip

        Return:
        - (origin, destination): (str, str), origin and destination
        """

        oid, did = None, None
        if start.HasField("aoi_position"):
            oid = f"{start.aoi_position.aoi_id}-start"
        elif start.HasField("lane_position"):
            lane = self._map_convertor._lanes[start.lane_position.lane_id]
            if lane.parent_id not in self._map_convertor._road2nodes:
                raise ValueError(f"lane {lane.id} has invalid parent node")
            # choose the successor node as oid
            _, oid = self._map_convertor._road2nodes[lane.parent_id]
        else:
            raise ValueError("start position is invalid")

        if end.HasField("aoi_position"):
            did = f"{end.aoi_position.aoi_id}-end"
        elif end.HasField("lane_position"):
            lane = self._map_convertor._lanes[end.lane_position.lane_id]
            if lane.parent_id not in self._map_convertor._road2nodes:
                raise ValueError(f"lane {lane.id} has invalid parent node")
            # choose the predecessor node as did
            did, _ = self._map_convertor._road2nodes[lane.parent_id]
        else:
            raise ValueError("end position is invalid")

        return oid, did

    def _check_connection(self, start_road_id: int, end_road_id: int):
        """
        check if two road segments are connected

        Args:
        - start_road_id: int, start road segment id
        - end_road_id: int, end road segment id

        Return:
        - bool, True if two road segments are connected, False otherwise
        """
        return (start_road_id, end_road_id) in self._map_convertor._connected_road_pairs

    def run(
        self,
        persons: Persons,
        time_interval: int = 60,
        reset_routes: bool = False,
        column_gen_num: int = 10,
        column_update_num: int = 10,
    ):
        """
        run static traffic assignment on GMNS map with persons.
        trips of persons will be sliced into time intervals.
        static traffic assignment will be run for each time interval.
        route results will be saved into persons.

        EXPERIMENTAL: the method is only for trip with deterministic departure time. Other cases will be skipped.

        Args:
        - persons: Persons, persons with trips
        - time_interval: int, time interval (minutes) for static traffic assignment slice. Try to set time_interval larger than trip's travel time.
        - reset_routes: bool, reset routes of persons before running static traffic assignment
        - column_gen_num: int, number of column generation iterations for static traffic assignment
        - column_update_num: int, number of column update iterations for static traffic assignment

        Return:
        - Persons, persons with route results
        - dict, statistics of static traffic assignment
        """

        # step 1. convert map to GMNS map
        self._map_convertor.save(self._work_dir)

        # step 2. get all persons' driving trips with deterministic departure time
        from_trips = []  # (pi, si, ti, departure_time, start, end)
        for pi, p in enumerate(persons.persons):
            departure_time = None
            now = cast(Position, p.home)
            for si, s in enumerate(p.schedules):
                s = cast(Schedule, s)
                if s.HasField("departure_time"):
                    departure_time = s.departure_time
                for ti, t in enumerate(s.trips):
                    end = cast(Position, t.end)
                    if t.HasField("departure_time"):
                        departure_time = t.departure_time
                    if (
                        departure_time is not None
                        and t.mode == TripMode.TRIP_MODE_DRIVE_ONLY
                        and (reset_routes or len(t.routes) == 0)
                        and s.loop_count == 1
                    ):
                        t.ClearField("routes")
                        from_trips.append((pi, si, ti, departure_time, now, end))
                    departure_time = None
                    now = end
        from_trips.sort(key=lambda x: x[3])

        from_trips_i = 0
        success_cnt = 0
        total_volumes = 0
        valid_volumes = 0
        disjointed_cnt = 0
        while from_trips_i < len(from_trips):
            # choose first slice
            demands = defaultdict(int)  # (origin, destination) -> count
            start_t = from_trips[from_trips_i][3]
            end_t = start_t + time_interval * 60
            this_from_trips = []
            while from_trips_i < len(from_trips):
                pi, si, ti, departure_time, start, end = from_trips[from_trips_i]
                if departure_time >= end_t:
                    break
                this_from_trips.append((pi, si, ti, departure_time, start, end))
                oid, did = self._get_od(start, end)
                demands[(oid, did)] += 1
                from_trips_i += 1
            od_df = pd.DataFrame(
                [
                    {
                        "o_zone_id": o,
                        "d_zone_id": d,
                        "volume": volume,
                    }
                    for (o, d), volume in demands.items()
                ]
            )
            od_df.sort_values(["o_zone_id", "d_zone_id"], inplace=True)
            # write to demand.csv
            od_df.to_csv(os.path.join(self._work_dir, "demand.csv"), index=False)

            # step 3: run traffic assignment
            network = pg.read_network(
                input_dir=self._work_dir, length_unit="km", speed_unit="kph"
            )
            pg.load_demand(network, input_dir=self._work_dir, demand_period_str="AM")
            pg.perform_column_generation(column_gen_num, column_update_num, network)
            pg.output_columns(network, output_geometry=False, output_dir=self._work_dir)

            # step 4: get route results
            agent_df = pd.read_csv(
                os.path.join(self._work_dir, "agent.csv"), index_col=None
            )
            # (o_zone_id, d_zone_id) -> [{volume, node_sequence, link_sequence}]
            agent_pairs: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
            for _, row in agent_df.iterrows():
                volume = int(row["volume"])
                total_volumes += volume
                link_sequence = cast(List[str], row["link_sequence"].split(";"))
                # check path is valid and convert to road_ids for route .pb format
                start_with_aoi = link_sequence[0].find("start") != -1
                end_with_aoi = link_sequence[-1].find("end") != -1
                road_ids = []
                if start_with_aoi:
                    # {aoiid}-start-{roadid}-{junctionid}
                    road_id = link_sequence[0].split("-")[2]
                    road_ids.append(int(road_id))
                    link_sequence = link_sequence[1:]
                if end_with_aoi:
                    # replace the last link with the road id
                    road_id = link_sequence[-1].split("-")[2]
                    link_sequence[-1] = road_id
                road_ids.extend([int(link) for link in link_sequence])
                # check connection
                bad = False
                for i in range(1, len(road_ids)):
                    if not self._check_connection(road_ids[i - 1], road_ids[i]):
                        bad = True
                        break
                if bad:
                    disjointed_cnt += volume
                    continue
                valid_volumes += volume
                agent_pairs[(row["o_zone_id"], row["d_zone_id"])].append(
                    {
                        "volume": volume,
                        "road_ids": road_ids,
                        "start_with_aoi": start_with_aoi,
                        "end_with_aoi": end_with_aoi,
                    }
                )
            # assign route results to persons
            for pi, si, ti, departure_time, start, end in this_from_trips:
                oid, did = self._get_od(start, end)
                if (oid, did) not in agent_pairs:
                    continue
                t = cast(Trip, persons.persons[pi].schedules[si].trips[ti])
                result = agent_pairs[(oid, did)][0]
                road_ids: List[int] = copy(result["road_ids"])
                if not result["start_with_aoi"]:
                    start_lane = self._map_convertor._lanes[start.lane_position.lane_id]
                    road_ids.insert(0, start_lane.parent_id)
                    # check connection
                    if not self._check_connection(start_lane.id, road_ids[1]):
                        disjointed_cnt += 1
                        continue
                if not result["end_with_aoi"]:
                    end_lane = self._map_convertor._lanes[end.lane_position.lane_id]
                    road_ids.append(end_lane.parent_id)
                    # check connection
                    if not self._check_connection(road_ids[-2], end_lane.id):
                        disjointed_cnt += 1
                        continue
                assert len(road_ids) >= 2
                assert len(t.routes) == 0, f"routes should be empty, but got {t.routes}"
                t.routes.append(
                    Journey(
                        type=JourneyType.JOURNEY_TYPE_DRIVING,
                        driving=DrivingJourneyBody(road_ids=road_ids),
                    )
                )
                success_cnt += 1
                # update volume
                result["volume"] -= 1
                if result["volume"] <= 0:
                    agent_pairs[(oid, did)].pop(0)
                    if len(agent_pairs[(oid, did)]) == 0:
                        agent_pairs.pop((oid, did))

        return persons, {
            "trip_cnt": len(from_trips),
            "total_volumes": total_volumes,
            "valid_volumes": valid_volumes,
            "successful_cnt": success_cnt,
            "disjointed_cnt": disjointed_cnt,
        }
