import logging
import random
import xml.dom.minidom as minidom
from typing import List, Literal, Optional, Tuple, Union, cast
from xml.dom.minidom import parse

import numpy as np
import pycityproto.city.map.v2.map_pb2 as mapv2
import pycityproto.city.person.v1.person_pb2 as personv1
import pycityproto.city.routing.v2.routing_pb2 as routingv2
import pycityproto.city.trip.v2.trip_pb2 as tripv2
from pycityproto.city.map.v2.map_pb2 import Map
from shapely.geometry import LineString

__all__ = ["RouteConverter"]


class RouteConverter:
    def __init__(
        self,
        converted_map: Map,
        sumo_id_mappings: dict,
        route_path: str,
        additional_path: Optional[str] = None,
        seed: Optional[int] = 0,
    ):
        """
        Args:
        - converted_map: The converted map from SUMO network.
        - sumo_id_mappings: The mapping from SUMO id to the unique id in the converted map.
        - route_path: The path to the SUMO route file.
        - additional_path: The path to the additional file containing bus stops, charging stations, and parking areas.
        - seed: The random seed.
        """
        if seed is not None:
            random.seed(seed)
        self._additional_path = additional_path  # additional file with "busStop","chargingStation","parkingArea"
        self._id2uid = sumo_id_mappings
        m = converted_map
        logging.info("Reading converted map")
        self._lanes = {}
        self._juncs = {}
        self._roads = {}
        for l in m.lanes:
            lid = l.id
            pres = l.predecessors
            sucs = l.successors
            self._lanes[lid] = {
                "type": l.type,
                "geo": LineString([[p.x, p.y] for p in l.center_line.nodes]),
                "in_lids": [p.id for p in pres],
                "out_lids": [s.id for s in sucs],
                "parent_id": l.parent_id,
                "length": l.length,
            }
        for j in m.junctions:
            jid = j.id
            self._juncs[jid] = {
                "lane_ids": j.lane_ids,
            }
        for r in m.roads:
            rid = r.id
            self._roads[rid] = {
                "lane_ids": r.lane_ids,
                "length": self._lanes[r.lane_ids[len(r.lane_ids) // 2]]["length"],
                "driving_lane_ids": [
                    lid
                    for lid in r.lane_ids
                    if self._lanes[lid]["type"] == mapv2.LANE_TYPE_DRIVING
                ],
                "walking_lane_ids": [
                    lid
                    for lid in r.lane_ids
                    if self._lanes[lid]["type"] == mapv2.LANE_TYPE_WALKING
                ],
            }
        logging.info(f"Reading route from {route_path}")
        dom_tree = parse(route_path)
        # get the root node
        root_node = dom_tree.documentElement
        # read SUMO .route.xml
        self._vtype = (
            {}
        )  # vehicle_id -> (agent_attribute,vehicle_attribute,pedestrian_attribute,bike_attribute,agent_type,)
        self._routes = root_node.getElementsByTagName("route")
        self._trips = root_node.getElementsByTagName("trip")
        self._flows = root_node.getElementsByTagName("flow")
        self._intervals = root_node.getElementsByTagName(
            "interval"
        )  # interval contains multiple `flows`
        self._vehicles = root_node.getElementsByTagName("vehicle")
        # output data
        self._output_agents = []
        for v in root_node.getElementsByTagName("vType"):
            vid = v.getAttribute("id")
            max_acc = (
                np.float64(v.getAttribute("accel")) if v.hasAttribute("accel") else 3.0
            )
            max_dec = (
                -np.float64(v.getAttribute("decel"))
                if v.hasAttribute("decel")
                else -4.5
            )
            length = (
                np.float64(v.getAttribute("length"))
                if v.hasAttribute("length")
                else 5.0
            )
            max_speed = (
                np.float64(v.getAttribute("maxSpeed"))
                if v.hasAttribute("maxSpeed")
                else 41.6666666667
            )
            width = (
                np.float64(v.getAttribute("width")) if v.hasAttribute("width") else 2.0
            )
            min_gap = (
                np.float64(v.getAttribute("minGap"))
                if v.hasAttribute("minGap")
                else 1.0
            )
            v_class = v.getAttribute("vClass") if v.hasAttribute("vClass") else ""
            if v_class == "pedestrian":
                agent_type = "AGENT_TYPE_PERSON"
            elif v_class == "bus":
                agent_type = "AGENT_TYPE_BUS"
            elif v_class == "bicycle":
                agent_type = "AGENT_TYPE_BIKE"
            else:
                agent_type = "AGENT_TYPE_PRIVATE_CAR"

            usual_acc = 2.0
            usual_dec = -4.5
            LC_length = 2 * length
            # TODO: add other car-following-model
            # https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html#car-following_models
            # model_name = v.getAttribute("carFollowModel") if v.hasAttribute("carFollowModel") else "IDM"
            self._vtype[vid] = (
                {
                    "length": length,
                    "width": width,
                    "max_speed": max_speed,
                    "max_acceleration": max_acc,
                    "max_braking_acceleration": max_dec,
                    "usual_acceleration": usual_acc,
                    "usual_braking_acceleration": usual_dec,
                },
                {
                    "lane_change_length": LC_length,
                    "min_gap": min_gap,
                },
                {
                    "speed": 1.34,
                },
                {
                    "speed": 5.0,
                },
                agent_type,
            )
        self._additional_stops = {}
        if self._additional_path:
            add_dom_tree = parse(self._additional_path)
            add_root_node = add_dom_tree.documentElement
            for stop_type in ["busStop", "chargingStation", "parkingArea"]:
                for stop in add_root_node.getElementsByTagName(stop_type):
                    stop_id = stop.getAttribute("id")
                    stop_name = (
                        stop.getAttribute("name") if stop.hasAttribute("name") else ""
                    )
                    if stop.getAttribute("lane") in self._id2uid.keys():
                        stop_lid = self._id2uid[stop.getAttribute("lane")]
                        stop_lane = self._lanes[stop_lid]
                        start_pos = (
                            np.float64(stop.getAttribute("startPos"))
                            if stop.hasAttribute("startPos")
                            else 0.1 * stop_lane["length"]
                        )
                        if start_pos < 0:
                            start_pos += stop_lane["length"]
                        end_pos = (
                            np.float64(stop.getAttribute("endPos"))
                            if stop.hasAttribute("endPos")
                            else 0.9 * stop_lane["length"]
                        )
                        if end_pos < 0:
                            end_pos += stop_lane["length"]
                        stop_s = (end_pos + start_pos) / 2
                        stop_s = np.clip(
                            stop_s, 0.1 * stop_lane["length"], 0.9 * stop_lane["length"]
                        )
                        veh_stop = {
                            "lane_position": {
                                "lane_id": stop_lid,
                                "s": stop_s,
                            }
                        }
                        parent_rid = stop_lane["parent_id"]
                        self._additional_stops[stop_id] = (veh_stop, parent_rid)
                    else:
                        logging.warning(f"Invalid stop {stop_id} {stop_name}!")

    def _convert_time(self, time_str: str) -> np.float64:
        if ":" not in time_str:
            return np.float64(time_str)
        converted_time = np.float64(0)
        times = time_str.split(":")
        t_factor = 1.0
        for t in times[::-1][:3]:
            converted_time += t_factor * np.float64(t)
            t_factor *= 60
        return converted_time

    def _convert_route_trips(
        self, edges: list, repeat: int, cycle_time: np.float64, rid2stop: dict
    ):
        route_trips = []  # Road ids separated by stop
        last_stop_rid = None  # Ensure road ids contain the start and end points
        for n_repeat in range(repeat + 1):
            road_ids = []
            for eid in edges:
                if not eid in self._id2uid.keys():  # Indicates junction lane
                    continue
                else:
                    if last_stop_rid:
                        road_ids.append(last_stop_rid)
                        last_stop_rid = None
                    rid = self._id2uid[eid]
                    road_ids.append(rid)
                    if rid in rid2stop.keys():
                        stop = rid2stop[rid]
                        if stop["until"]:
                            stop["until"] += n_repeat * cycle_time
                        route_trips.append(
                            {
                                "road_ids": road_ids,
                                "stop": stop,
                            }
                        )
                        last_stop_rid = rid
                        road_ids = []
            if road_ids:
                route_trips.append(
                    {
                        "road_ids": road_ids,
                        "stop": None,
                    }
                )
        return route_trips

    def _process_route_trips(
        self,
        t: minidom.Element,
        route_trips: list,
        trip_id: int,
        pre_veh_end: dict,
        TRIP_MODE: int,
        ROAD_LANE_TYPE: Union[Literal["walking_lane_ids"], Literal["driving_lane_ids"]],
        SPEED: float,
        departure: np.float64,
        trip_type: Union[Literal["trip"], Literal["flow"], Literal["vehicle"]] = "flow",
    ):
        schedules = []
        for i, route_trip in enumerate(route_trips):
            road_ids = route_trip["road_ids"]
            stop = route_trip["stop"]
            from_rid = road_ids[0]
            from_road = self._roads[from_rid]
            to_rid = road_ids[-1]
            to_road = self._roads[to_rid]
            if not stop:
                veh_end = self._get_trip_position(
                    t,
                    trip_id,
                    to_road,
                    to_rid,
                    ROAD_LANE_TYPE,
                    trip_type=trip_type,
                    attribute="arrivalLane",
                )
            else:
                veh_end = stop["veh_end"]
            if TRIP_MODE in {
                tripv2.TRIP_MODE_WALK_ONLY,
                tripv2.TRIP_MODE_BIKE_WALK,
                tripv2.TRIP_MODE_BUS_WALK,
            }:
                pre_lid = pre_veh_end["lane_position"]["lane_id"]
                pre_geo = self._lanes[pre_lid]["geo"]
                cur_lid = veh_end["lane_position"]["lane_id"]
                cur_geo = self._lanes[cur_lid]["geo"]
                estimate_distance = np.sqrt(2) * cur_geo.distance(pre_geo)
                eta = max(estimate_distance / SPEED, 5)
                schedules.append(
                    {
                        "trips": [
                            {
                                "mode": TRIP_MODE,
                                "end": veh_end,
                                "activity": "other",
                            }
                        ],
                        "departure_time": departure,
                        "loop_count": 1,
                    }
                )
                pre_veh_end = veh_end
            else:
                route_len = 0
                for rid in road_ids:
                    route_len += self._roads[rid]["length"]
                eta = max(route_len / SPEED, 5)
                journey = {
                    "type": routingv2.JOURNEY_TYPE_DRIVING,
                    "driving": {
                        "road_ids": road_ids,
                        "eta": eta,
                    },
                }
                schedules.append(
                    {
                        "trips": [
                            {
                                "mode": TRIP_MODE,
                                "end": veh_end,
                                "activity": "other",
                                "routes": [journey],
                            }
                        ],
                        "departure_time": departure,
                        "loop_count": 1,
                    }
                )
                pre_veh_end = veh_end
            if stop:
                duration = stop["duration"]
                until = stop["until"]
                if duration:
                    departure += duration
                elif until:
                    departure = max(departure + eta, until)
            else:
                departure += eta
        return schedules

    def _convert_trips_with_route(
        self,
        t: minidom.Element,
        departure_times: list[np.float64],
        TRIP_MODE: int,
        ROAD_LANE_TYPE: Union[Literal["walking_lane_ids"], Literal["driving_lane_ids"]],
        SPEED: float,
        trip_id: int,
        trip_type: Union[Literal["trip"], Literal["flow"], Literal["vehicle"]] = "flow",
    ):
        if t.hasAttribute("route"):
            route_id = t.getAttribute("route")
            troute = self.route_dict[route_id]
        else:
            troute = t.getElementsByTagName("route")[0]
        for departure in departure_times:
            edges = troute.getAttribute("edges").split(" ")
            repeat = (
                int(troute.getAttribute("repeat"))
                if troute.hasAttribute("repeat")
                else 0
            )
            vstops = troute.getElementsByTagName("stop")
            rid2stop = self._convert_stops(
                all_stops=list(t.getElementsByTagName("stop")) + list(vstops),
                trip_id=trip_id,
                trip_type=trip_type,
            )
            cycle_time = (
                self._convert_time(troute.getAttribute("cycleTime"))
                if troute.hasAttribute("cycleTime")
                else np.float64(0)
            )
            route_trips = self._convert_route_trips(edges, repeat, cycle_time, rid2stop)

            if not route_trips:
                logging.warning(f"Bad route at {trip_type} {trip_id}")
                continue
            # processing route_trips
            self._route_trips_to_person(
                route_trips,
                t,
                trip_id,
                ROAD_LANE_TYPE,
                trip_type,
                TRIP_MODE,
                SPEED,
                departure,
            )

    def _convert_flows_with_from_to(
        self,
        f: minidom.Element,
        departure_times: list[np.float64],
        flow_id: int,
        ROAD_LANE_TYPE: Union[Literal["walking_lane_ids"], Literal["driving_lane_ids"]],
        TRIP_MODE: int,
    ):
        from_eid = f.getAttribute("from")
        from_rid = self._id2uid[from_eid]
        from_road = self._roads[from_rid]
        to_eid = f.getAttribute("to")
        to_rid = self._id2uid[to_eid]
        to_road = self._roads[to_rid]
        for departure in departure_times:
            flow_home = self._get_trip_position(
                f,
                flow_id,
                from_road,
                from_rid,
                ROAD_LANE_TYPE,
                trip_type="flow",
                attribute="departLane",
            )
            if not flow_home:
                break
            flow_end = self._get_trip_position(
                f,
                flow_id,
                to_road,
                to_rid,
                ROAD_LANE_TYPE,
                trip_type="flow",
                attribute="arrivalLane",
            )
            if not flow_end:
                break
            schedules = [
                {
                    "trips": [
                        {
                            "mode": TRIP_MODE,
                            "end": flow_end,
                            "activity": "other",
                        }
                    ],
                    "departure_time": departure,
                    "loop_count": 1,
                }
            ]
            self._output_agents.append(
                {
                    "id": self.agent_uid,
                    "home": flow_home,
                    "attribute": self.agent_attribute,
                    "vehicle_attribute": self.vehicle_attribute,
                    "pedestrian_attribute": self.pedestrian_attribute,
                    "bike_attribute": self.bike_attribute,
                    "schedules": schedules,
                }
            )
            self.agent_uid += 1

    def _convert_stops(
        self,
        all_stops: list,
        trip_id: int,
        trip_type: Union[Literal["trip"], Literal["flow"], Literal["vehicle"]] = "flow",
    ):
        rid2stop = {}
        for s in all_stops:
            if any(
                [
                    s.hasAttribute(stop_type)
                    for stop_type in [
                        "containerStop",
                    ]
                ]
            ):
                logging.warning(f"Unsupported stop type at {trip_type} {trip_id}")
                continue
            if s.hasAttribute("busStop"):
                stop_id = s.getAttribute("busStop")
                if stop_id in self._additional_stops.keys():
                    (veh_stop, parent_rid) = self._additional_stops[stop_id]
                else:
                    logging.warning(f"Invalid busStop {stop_id}")
                    continue
            elif s.hasAttribute("chargingStation"):
                stop_id = s.getAttribute("chargingStation")
                if stop_id in self._additional_stops.keys():
                    (veh_stop, parent_rid) = self._additional_stops[stop_id]
                else:
                    logging.warning(f"Invalid chargingStation {stop_id}")
                    continue
            elif s.hasAttribute("parkingArea"):
                stop_id = s.getAttribute("parkingArea")
                if stop_id in self._additional_stops.keys():
                    (veh_stop, parent_rid) = self._additional_stops[stop_id]
                else:
                    logging.warning(f"Invalid parkingArea {stop_id}")
                    continue
            elif s.getAttribute("lane") in self._id2uid.keys():
                stop_lid = self._id2uid[s.getAttribute("lane")]
                stop_lane = self._lanes[stop_lid]
                start_pos = (
                    np.float64(s.getAttribute("startPos"))
                    if s.hasAttribute("startPos")
                    else 0.1 * stop_lane["length"]
                )
                if start_pos < 0:
                    start_pos += stop_lane["length"]
                end_pos = (
                    np.float64(s.getAttribute("endPos"))
                    if s.hasAttribute("endPos")
                    else 0.9 * stop_lane["length"]
                )
                if end_pos < 0:
                    end_pos += stop_lane["length"]
                stop_s = (end_pos + start_pos) / 2
                stop_s = np.clip(
                    stop_s, 0.1 * stop_lane["length"], 0.9 * stop_lane["length"]
                )
                veh_stop = {
                    "lane_position": {
                        "lane_id": stop_lid,
                        "s": stop_s,
                    }
                }
                parent_rid = stop_lane["parent_id"]
            else:
                logging.warning(f"Unsupported stop type at {trip_type} {trip_id}")
                continue
            duration = (
                self._convert_time(s.getAttribute("duration"))
                if s.hasAttribute("duration")
                else None
            )
            until = (
                self._convert_time(s.getAttribute("until"))
                if s.hasAttribute("until")
                else None
            )
            if not duration and not until:
                continue
            else:
                rid2stop[parent_rid] = {
                    "veh_end": veh_stop,
                    "duration": duration,
                    "until": until,
                }
        return rid2stop

    def _get_trip_position(
        self,
        t: minidom.Element,
        trip_id: int,
        road: dict,
        road_id: int,
        ROAD_LANE_TYPE: Union[Literal["walking_lane_ids"], Literal["driving_lane_ids"]],
        trip_type: Union[Literal["trip"], Literal["flow"], Literal["vehicle"]],
        attribute: Union[Literal["departLane"], Literal["arrivalLane"]],
    ):
        lid = None
        res_pos = {}
        if t.hasAttribute(attribute):
            attribute_id = t.getAttribute(attribute)
            if attribute_id in self._id2uid.keys():
                lid = self._id2uid[attribute_id]
        if not lid or lid not in road[ROAD_LANE_TYPE]:
            lid = random.choice(road[ROAD_LANE_TYPE])
            if not road[ROAD_LANE_TYPE]:
                logging.warning(
                    f"Wrong Lane Type {ROAD_LANE_TYPE} at {road_id} at {trip_type} {trip_id}"
                )
                lid = None
        s_proj = random.uniform(0.1, 0.9) * self._lanes[lid]["length"]
        if lid is not None:
            res_pos = {
                "lane_position": {
                    "lane_id": lid,
                    "s": s_proj,
                }
            }
        return res_pos

    def _process_agent_type(self):
        agent_type = self.agent_type
        if agent_type == "AGENT_TYPE_PERSON":
            TRIP_MODE = tripv2.TRIP_MODE_WALK_ONLY
            SPEED = random.uniform(0.3, 0.8) * 2
            ROAD_LANE_TYPE = "walking_lane_ids"
        elif agent_type == "AGENT_TYPE_BIKE":
            TRIP_MODE = tripv2.TRIP_MODE_BIKE_WALK
            SPEED = random.uniform(3, 6)
            ROAD_LANE_TYPE = "walking_lane_ids"
        elif agent_type == "AGENT_TYPE_PRIVATE_CAR":
            TRIP_MODE = tripv2.TRIP_MODE_DRIVE_ONLY
            SPEED = random.uniform(0.3, 0.8) * 50 / 3.6
            ROAD_LANE_TYPE = "driving_lane_ids"
        else:
            TRIP_MODE = tripv2.TRIP_MODE_DRIVE_ONLY
            SPEED = random.uniform(0.3, 0.8) * 50 / 3.6
            ROAD_LANE_TYPE = "driving_lane_ids"
        return (TRIP_MODE, SPEED, ROAD_LANE_TYPE)

    def _route_trips_to_person(
        self,
        route_trips: list,
        t: minidom.Element,
        trip_id: int,
        ROAD_LANE_TYPE: Union[Literal["walking_lane_ids"], Literal["driving_lane_ids"]],
        trip_type: Union[Literal["trip"], Literal["flow"], Literal["vehicle"]],
        TRIP_MODE: int,
        SPEED: float,
        departure: np.float64,
    ):
        home_rid = route_trips[0]["road_ids"][0]
        for i in range(len(route_trips) - 1):
            cur_route = route_trips[i]
            next_route = route_trips[i + 1]
            if not cur_route["road_ids"][-1] == next_route["road_ids"][0]:
                assert (
                    home_rid == next_route["road_ids"][0]
                )  # only process when `repeat` is valid
                cur_route["road_ids"].append(home_rid)

        home_road = self._roads[home_rid]
        veh_home = self._get_trip_position(
            t,
            trip_id,
            home_road,
            home_rid,
            ROAD_LANE_TYPE,
            trip_type,
            attribute="departLane",
        )
        if not veh_home:
            return
        pre_veh_end = veh_home

        schedules = self._process_route_trips(
            t,
            route_trips,
            trip_id,
            pre_veh_end,
            TRIP_MODE,
            ROAD_LANE_TYPE,
            SPEED,
            departure,
            trip_type,
        )
        self._output_agents.append(
            {
                "id": self.agent_uid,
                "home": veh_home,
                "attribute": self.agent_attribute,
                "vehicle_attribute": self.vehicle_attribute,
                "pedestrian_attribute": self.pedestrian_attribute,
                "bike_attribute": self.bike_attribute,
                "schedules": schedules,
            }
        )
        self.agent_uid += 1

    def convert_route(self):
        self.agent_uid = 0
        DEFAULT_AGENT_ATTRIBUTE = {
            "length": 5,
            "width": 2,
            "max_speed": 41.6666666667,
            "max_acceleration": 3,
            "max_braking_acceleration": -10,
            "usual_acceleration": 2,
            "usual_braking_acceleration": -4.5,
        }
        DEFAULT_VEHICLE_ATTRIBUTE = {
            "lane_change_length": 10,
            "min_gap": 1,
        }
        DEFAULT_PEDESTRIAN_ATTRIBUTE = {
            "lane_change_length": 10,
            "min_gap": 1,
        }
        DEFAULT_BIKE_ATTRIBUTE = {
            "lane_change_length": 10,
            "min_gap": 1,
        }
        DEFAULT_AGENT_TYPE = "AGENT_TYPE_PRIVATE_CAR"

        # Route contains the edges that all vehicles pass through, that is, the complete trajectory
        # Route can be defined separately from vehicle or under vehicle, so additional judgment is required.
        self.route_dict = {}
        for r in self._routes:
            route_id = r.getAttribute("id")
            self.route_dict[route_id] = r
        if self._trips:
            logging.info("Converting trips")
        for t in self._trips:
            if t.hasAttribute("type"):
                trip_type = t.getAttribute("type")
                self.agent_attribute, self.vehicle_attribute = self._vtype[trip_type]
            else:
                self.agent_attribute, self.vehicle_attribute = (
                    DEFAULT_AGENT_ATTRIBUTE,
                    DEFAULT_VEHICLE_ATTRIBUTE,
                )
            (TRIP_MODE, SPEED, ROAD_LANE_TYPE) = self._process_agent_type()
            trip_id = t.getAttribute("id")
            departure = self._convert_time(t.getAttribute("depart"))
            if not t.hasAttribute("from") or not t.hasAttribute("to"):
                # no from and to means this item has route
                if t.hasAttribute("route"):
                    route_id = t.getAttribute("route")
                    troute = self.route_dict[route_id]
                else:
                    troute = t.getElementsByTagName("route")[0]
                edges = troute.getAttribute("edges").split(" ")
                repeat = (
                    int(troute.getAttribute("repeat"))
                    if troute.hasAttribute("repeat")
                    else 0
                )
                vstops = troute.getElementsByTagName("stop")
                rid2stop = self._convert_stops(
                    all_stops=list(t.getElementsByTagName("stop")) + list(vstops),
                    trip_id=trip_id,
                    trip_type="trip",
                )

                cycle_time = (
                    self._convert_time(troute.getAttribute("cycleTime"))
                    if troute.hasAttribute("cycleTime")
                    else np.float64(0)
                )
                route_trips = self._convert_route_trips(
                    edges, repeat, cycle_time, rid2stop
                )

                if not route_trips:
                    logging.warning(f"Bad route at trip {trip_id}")
                    continue
                # process route_trips
                self._route_trips_to_person(
                    route_trips,
                    t,
                    trip_id,
                    ROAD_LANE_TYPE,
                    "trip",
                    TRIP_MODE,
                    SPEED,
                    departure,
                )
            else:
                from_eid = t.getAttribute("from")
                via_eids = (
                    t.getAttribute("via").split(" ") if t.hasAttribute("via") else []
                )
                to_eid = t.getAttribute("to")
                from_rid = self._id2uid[from_eid]
                from_road = self._roads[from_rid]
                to_rid = self._id2uid[to_eid]
                to_road = self._roads[to_rid]
                trip_home = self._get_trip_position(
                    t,
                    trip_id,
                    from_road,
                    from_rid,
                    ROAD_LANE_TYPE,
                    trip_type="trip",
                    attribute="departLane",
                )
                if not trip_home:
                    continue
                trip_end = self._get_trip_position(
                    t,
                    trip_id,
                    to_road,
                    to_rid,
                    ROAD_LANE_TYPE,
                    trip_type="trip",
                    attribute="arrivalLane",
                )
                if not trip_end:
                    continue
                via_eids.append(to_eid)
                via_rids = []
                for eid in via_eids:
                    if eid in self._id2uid.keys():
                        via_rids.append(self._id2uid[eid])
                schedules = []
                via_ends = [{} for _ in range(len(via_rids))]
                via_ends[-1] = trip_end
                pre_via_end = trip_home
                for i, rid in enumerate(via_rids):
                    via_end = via_ends[i]
                    if not via_end:
                        via_road = self._roads[rid]
                        if not via_road[ROAD_LANE_TYPE]:
                            continue
                        via_lid = random.choice(via_road[ROAD_LANE_TYPE])
                        via_s = (
                            random.uniform(0.1, 0.9) * self._lanes[via_lid]["length"]
                        )
                        via_end = {
                            "lane_position": {
                                "lane_id": via_lid,
                                "s": via_s,
                            }
                        }
                    pre_lid = pre_via_end["lane_position"]["lane_id"]
                    pre_geo = self._lanes[pre_lid]["geo"]
                    cur_lid = via_end["lane_position"]["lane_id"]
                    cur_geo = self._lanes[cur_lid]["geo"]
                    estimate_distance = np.sqrt(2) * cur_geo.distance(pre_geo)
                    eta = max(estimate_distance / SPEED, 5)
                    schedules.append(
                        {
                            "trips": [
                                {
                                    "mode": TRIP_MODE,
                                    "end": via_end,
                                    "activity": "other",
                                }
                            ],
                            "departure_time": departure,
                            "loop_count": 1,
                        }
                    )
                    departure += eta
                    pre_via_end = via_end
                self._output_agents.append(
                    {
                        "id": self.agent_uid,
                        "home": trip_home,
                        "attribute": self.agent_attribute,
                        "vehicle_attribute": self.vehicle_attribute,
                        "pedestrian_attribute": self.pedestrian_attribute,
                        "bike_attribute": self.bike_attribute,
                        "schedules": schedules,
                    }
                )
                self.agent_uid += 1

        def get_flow_departure_times(
            f: minidom.Element, begin_time: np.float64, end_time: np.float64
        ) -> list[np.float64]:
            departure_times = []
            if f.hasAttribute("number"):
                number = int(f.getAttribute("number"))
                departure_times = list(
                    np.linspace(begin, end, number).astype(np.float64)
                )
            elif f.hasAttribute("period"):
                period = self._convert_time(f.getAttribute("period"))
                number = int((end - begin) / period)
                departure_times = list(
                    np.linspace(begin, end, number).astype(np.float64)
                )
            elif f.hasAttribute("vehsPerHour"):
                vehs_per_hour = int(f.getAttribute("vehsPerHour"))
                number = int(vehs_per_hour * (end_time - begin_time) / 3600)
                departure_times = list(
                    np.linspace(begin, end, number).astype(np.float64)
                )
            elif f.hasAttribute("probability"):
                prob = np.float64(f.getAttribute("probability"))
                for i in range(int(end - begin) + 1):
                    if random.random() < prob:
                        departure_times.append(np.float64(i + begin))
            return departure_times

        if self._flows or self._intervals:
            logging.info("Converting flows")
        for f in self._flows:
            flow_id = f.getAttribute("id")
            begin = self._convert_time(f.getAttribute("begin"))
            end = self._convert_time(f.getAttribute("end"))
            departure_times = get_flow_departure_times(f, begin, end)
            if len(departure_times) < 1:
                logging.warning(f"Incomplete flow {flow_id} at vehicle num!")
                continue
            if f.hasAttribute("type"):
                flow_type = f.getAttribute("type")
                self.agent_attribute, self.vehicle_attribute = self._vtype[flow_type]
            else:
                self.agent_attribute, self.vehicle_attribute = (
                    DEFAULT_AGENT_ATTRIBUTE,
                    DEFAULT_VEHICLE_ATTRIBUTE,
                )
            (TRIP_MODE, SPEED, ROAD_LANE_TYPE) = self._process_agent_type()
            if not f.hasAttribute("from") or not f.hasAttribute("to"):
                # no from and to means this item has route
                self._convert_trips_with_route(
                    f,
                    departure_times,
                    TRIP_MODE,
                    ROAD_LANE_TYPE,
                    SPEED,
                    trip_id=flow_id,
                    trip_type="flow",
                )
            else:
                self._convert_flows_with_from_to(
                    f, departure_times, flow_id, ROAD_LANE_TYPE, TRIP_MODE
                )
        for i in self._intervals:
            begin = self._convert_time(i.getAttribute("begin"))
            end = self._convert_time(i.getAttribute("end"))
            for f in i.getElementsByTagName("flow"):
                flow_id = f.getAttribute("id")
                departure_times = get_flow_departure_times(f, begin, end)
                if len(departure_times) < 1:
                    logging.warning(f"Incomplete flow {flow_id} at vehicle num!")
                    continue
                if f.hasAttribute("type"):
                    flow_type = f.getAttribute("type")
                    self.agent_attribute, self.vehicle_attribute = self._vtype[
                        flow_type
                    ]
                else:
                    self.agent_attribute, self.vehicle_attribute = (
                        DEFAULT_AGENT_ATTRIBUTE,
                        DEFAULT_VEHICLE_ATTRIBUTE,
                    )
                (TRIP_MODE, SPEED, ROAD_LANE_TYPE) = self._process_agent_type()
                if not f.hasAttribute("from") or not f.hasAttribute("to"):
                    # no from and to means this item has route
                    self._convert_trips_with_route(
                        f,
                        departure_times,
                        TRIP_MODE,
                        ROAD_LANE_TYPE,
                        SPEED,
                        flow_id,
                        trip_type="flow",
                    )
                else:
                    self._convert_flows_with_from_to(
                        f, departure_times, flow_id, ROAD_LANE_TYPE, TRIP_MODE
                    )
        if self._vehicles:
            logging.info("Converting routes")
        for v in self._vehicles:
            veh_id = v.getAttribute("id")
            if v.hasAttribute("type"):
                vehicle_type = v.getAttribute("type")
                (
                    self.agent_attribute,
                    self.vehicle_attribute,
                    self.pedestrian_attribute,
                    self.bike_attribute,
                    self.agent_type,
                ) = self._vtype[vehicle_type]
            else:
                (
                    self.agent_attribute,
                    self.vehicle_attribute,
                    self.pedestrian_attribute,
                    self.bike_attribute,
                    self.agent_type,
                ) = (
                    DEFAULT_AGENT_ATTRIBUTE,
                    DEFAULT_VEHICLE_ATTRIBUTE,
                    DEFAULT_PEDESTRIAN_ATTRIBUTE,
                    DEFAULT_BIKE_ATTRIBUTE,
                    DEFAULT_AGENT_TYPE,
                )
            (TRIP_MODE, SPEED, ROAD_LANE_TYPE) = self._process_agent_type()
            if v.hasAttribute("route"):
                route_id = v.getAttribute("route")
                vroute = self.route_dict[route_id]
            else:
                vroute = v.getElementsByTagName("route")[0]
            departure = self._convert_time(v.getAttribute("depart"))
            edges = vroute.getAttribute("edges").split(" ")
            repeat = (
                int(vroute.getAttribute("repeat"))
                if vroute.hasAttribute("repeat")
                else 0
            )
            vstops = vroute.getElementsByTagName("stop")
            rid2stop = self._convert_stops(
                all_stops=list(vstops), trip_id=veh_id, trip_type="vehicle"
            )
            cycle_time = (
                self._convert_time(vroute.getAttribute("cycleTime"))
                if vroute.hasAttribute("cycleTime")
                else np.float64(0)
            )
            route_trips = self._convert_route_trips(edges, repeat, cycle_time, rid2stop)
            if not route_trips:
                logging.warning(f"Bad route at vehicle {veh_id}")
                continue
            # process route_trips
            self._route_trips_to_person(
                route_trips,
                v,
                veh_id,
                ROAD_LANE_TYPE,
                "vehicle",
                TRIP_MODE,
                SPEED,
                departure,
            )

        return {"persons": self._output_agents}
