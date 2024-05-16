"""Generate traffic-light in junctions"""

import logging
from collections import defaultdict
from typing import Optional
from xml.dom.minidom import parse

import numpy as np
import pycityproto.city.map.v2.light_pb2 as lightv2
import pycityproto.city.map.v2.map_pb2 as mapv2
from shapely.geometry import LineString

__all__ = ["generate_traffic_light", "convert_traffic_light"]


def _gen_fixed_program(juncs: dict, lanes: dict, green_time: float, yellow_time: float):
    """
    Generate fixed program traffic-light
    """
    for junc_id, j in juncs.items():
        phases = j["phases"]  # all available phases
        if "fixed_program" in j and j["fixed_program"]:
            continue
        # outputs
        tl_phases = []
        walk_indexes_set = set()
        for index, lane_id in enumerate(j["lane_ids"]):
            if lanes[lane_id]["type"] == mapv2.LANE_TYPE_WALKING:
                walk_indexes_set.add(index)
        non_transition_phase_indexes = (
            []
        )  # index which is not yellow phase or all red phase.
        for i in range(len(phases)):
            light_states = phases[i]["states"]
            yellow_light_states = []
            all_red_light_states = []
            has_all_red_phase = False
            for index_offset, (pre_light, cur_light) in enumerate(
                zip(phases[i - 1]["states"], phases[i]["states"])
            ):
                if (
                    pre_light == lightv2.LIGHT_STATE_GREEN
                    and cur_light == lightv2.LIGHT_STATE_RED
                ):
                    yellow_light_states.append(
                        lightv2.LIGHT_STATE_YELLOW
                    )  # green➡red, add one yellow phase
                else:
                    if (
                        pre_light == lightv2.LIGHT_STATE_RED
                        and cur_light == lightv2.LIGHT_STATE_GREEN
                        and index_offset not in walk_indexes_set
                    ):
                        yellow_light_states.append(lightv2.LIGHT_STATE_RED)
                    else:
                        yellow_light_states.append(cur_light)
                if (
                    pre_light == lightv2.LIGHT_STATE_RED
                    and cur_light == lightv2.LIGHT_STATE_GREEN
                    and index_offset not in walk_indexes_set
                ):
                    all_red_light_states.append(
                        lightv2.LIGHT_STATE_RED
                    )  # red➡green add one all red phase
                    has_all_red_phase = True
                else:
                    all_red_light_states.append(cur_light)
            # yellow phase
            if any(l == lightv2.LIGHT_STATE_YELLOW for l in yellow_light_states):
                tl_phases.append(
                    {"duration": yellow_time, "states": yellow_light_states}
                )
            # all red phase
            if has_all_red_phase:
                tl_phases.append(
                    {"duration": yellow_time, "states": all_red_light_states}
                )
            non_transition_phase_indexes.append(len(tl_phases))
            tl_phases.append({"duration": green_time, "states": light_states})
        PEDESTRIAN_CLEAR_RATIO = 0.15  # The proportion of walk lanes advance change time to the entire green phase time
        index_2_new_phases = defaultdict(list)
        for index in non_transition_phase_indexes:
            cur_phase = tl_phases[index]
            cur_phase_duration = cur_phase["duration"]
            next_phase = tl_phases[(index + 1) % len(tl_phases)]
            ped_clear_indexes = set()
            for i, l in enumerate(next_phase["states"]):
                if l == lightv2.LIGHT_STATE_YELLOW and i in walk_indexes_set:
                    ped_clear_indexes.add(i)
            # there is no yellow phase walking lanes in next phase, skip
            if len(ped_clear_indexes) == 0:
                index_2_new_phases[index].append(cur_phase)
                continue
            else:
                orig_duration = cur_phase_duration * (1 - PEDESTRIAN_CLEAR_RATIO)
                index_2_new_phases[index].append(
                    {"duration": orig_duration, "states": cur_phase["states"]}
                )
                clear_duration = cur_phase_duration * PEDESTRIAN_CLEAR_RATIO
                clear_states = []
                for i, l in enumerate(cur_phase["states"]):
                    if i not in ped_clear_indexes:
                        clear_states.append(l)
                    else:
                        clear_states.append(lightv2.LIGHT_STATE_YELLOW)
                index_2_new_phases[index].append(
                    {"duration": clear_duration, "states": clear_states}
                )
        output_phases = []
        for index, phase in enumerate(tl_phases):
            if index not in index_2_new_phases:
                output_phases.append(phase)
            else:
                for new_phase in index_2_new_phases[index]:
                    output_phases.append(new_phase)
        j["fixed_program"] = {
            "junction_id": junc_id,
            "phases": output_phases,
        }


# Convert SUMO format signal lights and generate default fixed phase traffic-lights for the rest
def _convert_fixed_program(
    lanes: dict,
    roads: dict,
    juncs: dict,
    green_time: float,
    yellow_time: float,
    id_mapping: dict,
    traffic_light_path: Optional[str],
):
    for jid, j in juncs.items():
        j["fixed_program"] = {}
    # Generation
    if traffic_light_path is not None:
        # convert
        start_angle = (
            -np.pi / 2
        )  # SUMO's signal control order starts from -pi/2 clockwise
        # read
        tl_lanes = {}
        for lid, l in lanes.items():
            pres = l["predecessors"]
            sucs = l["successors"]
            tl_lanes[lid] = {
                "type": l["type"],
                "geo": LineString(
                    [[p["x"], p["y"]] for p in l["center_line"]["nodes"]]
                ),
                "in_lids": [p["id"] for p in pres],
                "out_lids": [s["id"] for s in sucs],
            }
        logging.info(f"Reading tl_logic from {traffic_light_path}")
        dom_tree = parse(traffic_light_path)
        # get the root node
        root_node = dom_tree.documentElement
        # read SUMO .add.xml
        tl_logics = root_node.getElementsByTagName("tlLogic")
        orig_tls = []

        def str2state(s: str):
            state = []
            for l in s:
                if l in ["G", "g"]:
                    state.append(lightv2.LIGHT_STATE_GREEN)
                elif l in ["Y", "y"]:
                    state.append(lightv2.LIGHT_STATE_YELLOW)
                elif l in ["R", "r"]:
                    state.append(lightv2.LIGHT_STATE_RED)
                else:
                    # default to green light
                    state.append(lightv2.LIGHT_STATE_GREEN)
            return state

        for tl in tl_logics:
            jid = tl.getAttribute("id")
            if not jid in id_mapping:
                logging.warning(f"junction {jid} not exist")
                continue
            else:
                tl_type = tl.getAttribute("type")
                if not tl_type == "static":
                    logging.warning(
                        f"Cannot convert {tl_type} TrafficLight, Regard as static"
                    )
                time_offset = np.float64(tl.getAttribute("offset"))
                junc_uid = id_mapping[jid]
                assert junc_uid >= 3_0000_0000
                sumo_phases = tl.getElementsByTagName("phase")
                phases = []
                for p in sumo_phases:
                    duration = np.float64(p.getAttribute("duration"))
                    state = str2state(p.getAttribute("state"))
                    phases.append((duration, state))
                # shift phases according to time_offset
                while time_offset > phases[-1][0]:  # duration
                    time_offset -= phases[-1][0]
                    phases = phases[-1:] + phases[:-1]
                orig_tls.append(
                    {
                        "junc_id": junc_uid,
                        "phases": phases,
                    }
                )

        def get_lane_angle(line):
            """
            对lane进行排序
            """

            def in_get_vector(line):
                return np.array(line.coords[-1]) - np.array(
                    line.coords[4 * len(line.coords) // 5 - 1]
                )

            v = in_get_vector(line)
            angle = np.arctan2(v[1], v[0])
            angle -= start_angle
            if angle > 2 * np.pi:
                angle -= 2 * np.pi
            elif angle < 0:
                angle += 2 * np.pi
            return angle

        for tl in orig_tls:
            junc_id = tl["junc_id"]
            phases = tl["phases"]
            lane_ids = juncs[junc_id]["lane_ids"]
            unique_pre_lids = [
                tl_lanes[lid]["in_lids"][0] for lid in lane_ids
            ]  # junc lane has one unique predecessor
            lids_index = [i for i in range(len(lane_ids))]
            if not len(phases) == len(lane_ids):
                logging.warning(f"Different junction from .net.xml {junc_id}")
                continue
            lids_index = sorted(
                lids_index,
                key=lambda x: -get_lane_angle(tl_lanes[unique_pre_lids[x]]["geo"]),
            )
            out_phases = []
            for phase in phases:
                out_state = [lightv2.LIGHT_STATE_GREEN for _ in range(len(lane_ids))]
                (duration, state) = phase
                for i, lid_index in enumerate(lids_index):
                    out_state[lid_index] = state[i]
                out_phases.append(
                    {
                        "duration": duration,
                        "states": out_state,
                    }
                )
            juncs[junc_id]["fixed_program"] = {
                "junction_id": junc_id,
                "phases": out_phases,
            }
    # Complete the default fixed program traffic-light for all junctions
    _gen_fixed_program(juncs, lanes, green_time, yellow_time)


def _gen_available_phases(lanes: dict, juncs: dict, min_direction_group: int):
    """
    Generate all available phases for Max Pressure algorithm
    """

    def get_predecessor_direc(lane_id):
        pre_lane_id = lanes[lane_id]["predecessors"][0]["id"]
        pre_lane_center_line = lanes[pre_lane_id]["center_line"]["nodes"]
        x_1, y_1 = pre_lane_center_line[-1]["x"], pre_lane_center_line[-1]["y"]
        x_0, y_0 = pre_lane_center_line[-2]["x"], pre_lane_center_line[-2]["y"]
        return np.arctan2(y_1 - y_0, x_1 - x_0)

    def is_left(lane_id):
        """if this lane is a right turn lane"""
        lane_type = lanes[lane_id]["type"]
        return lane_type == mapv2.LANE_TURN_LEFT or lane_type == mapv2.LANE_TURN_AROUND

    def is_no_overlap_drive_right(lane_id):
        """if this lane is a right turn lane without overlap"""
        lane = lanes[lane_id]
        if (
            not lane["turn"] == mapv2.LANE_TURN_RIGHT
            or not lane["type"] == mapv2.LANE_TYPE_DRIVING
        ):
            return False
        else:
            no_overlap = True
            for o in lane["overlaps"]:
                if o["self"]["s"] <= 0.8 * lane["length"]:
                    no_overlap = False
                    break
            return no_overlap

    def has_independent_left(group):
        """if there is independent left turn lanes in this group"""
        for lane_id in group:
            if not is_left(lane_id):
                continue
            # For a left-turn lane, check whether there are non-left-turn lanes in other successors of its predecessor. If so, it is not an independent left-turn lane.
            pre_lane_id = lanes[lane_id]["predecessors"][0]["id"]
            pre_lane = lanes[pre_lane_id]
            suc_lane_ids = [suc["id"] for suc in pre_lane["successors"]]
            if all(is_left(lid) for lid in suc_lane_ids):
                # If all subsequent turns are left turns/U-turns, it will be judged as an independent left turn.
                return True
        return False

    for _, j in juncs.items():
        junc_lane_ids = j["lane_ids"]
        driving_lane_ids = [
            lid
            for lid in junc_lane_ids
            if lanes[lid]["type"] == mapv2.LANE_TYPE_DRIVING
        ]
        if (
            len(driving_lane_ids) < 1
        ):  # No phase will be generated when # of driving_lanes = 0
            j["phases"] = []
            continue
        ref_angle = get_predecessor_direc(driving_lane_ids[0])
        # Prepare road angle data
        direc_groups = defaultdict(list)
        NP_2_PI = np.pi * 2
        NP_PI_2 = np.pi / 2
        NP_PI_4 = np.pi / 4
        for lane_id in driving_lane_ids:
            # 0: -pi/4 ~ pi/4
            # 1: pi/4 ~ 3pi/4
            # 2: 3pi/4 ~ 5pi/4
            # 3: 5pi/4 ~ 7pi/4
            direc_type = (
                int(
                    (get_predecessor_direc(lane_id) - ref_angle + NP_2_PI + NP_PI_4)
                    / NP_PI_2
                )
                % 4
            )
            direc_groups[direc_type].append(lane_id)
        if len(j["driving_lane_groups"]) < min_direction_group:
            j["phases"] = []
            continue
        # Generate phase table
        group0 = direc_groups[0] + direc_groups[2]  # 0-180 degree
        group90 = direc_groups[1] + direc_groups[3]  # 90-270 degree
        is_green_funcs = []  # List of functions to determine green light
        stand_alone_walking_index = set()
        if has_independent_left(direc_groups[0]) and has_independent_left(
            direc_groups[2]
        ):
            # Go straight in the opposite direction
            is_green_funcs.append(
                lambda lane_id: lanes[lane_id]["type"] == mapv2.LANE_TURN_STRAIGHT
                and lane_id in group0
            )
            # Turn left on the opposite side
            is_green_funcs.append(
                lambda lane_id: is_left(lane_id) and lane_id in group0
            )
        else:
            # If there is no independent left turn, neither the opposite direction nor the opposite left turn will be considered, only the current direction will be considered and all will be released.
            if len(direc_groups[0]) > 0:
                # Allow all outgoing directions of one incoming direction
                is_green_funcs.append(lambda lane_id: lane_id in direc_groups[0])
            if len(direc_groups[2]) > 0:
                is_green_funcs.append(lambda lane_id: lane_id in direc_groups[2])
        if has_independent_left(direc_groups[1]) and has_independent_left(
            direc_groups[3]
        ):
            # Go straight in the opposite direction
            is_green_funcs.append(
                lambda lane_id: lanes[lane_id]["type"] == mapv2.LANE_TURN_STRAIGHT
                and lane_id in group90
            )
            # Turn left on the opposite side
            is_green_funcs.append(
                lambda lane_id: is_left(lane_id) and lane_id in group90
            )
        else:
            # If there is no independent left turn, neither the opposite direction nor the opposite left turn will be considered, only the current direction will be considered and all will be released.
            if len(direc_groups[1]) > 0:
                # Allow all outgoing directions of one incoming direction
                is_green_funcs.append(lambda lane_id: lane_id in direc_groups[1])
            if len(direc_groups[3]) > 0:
                is_green_funcs.append(lambda lane_id: lane_id in direc_groups[3])
        lane_id2index = {lane_id: i for i, lane_id in enumerate(junc_lane_ids)}
        phases = []
        for is_green_func in is_green_funcs:
            phase = [lightv2.LIGHT_STATE_UNSPECIFIED for _ in range(len(junc_lane_ids))]
            walking_overlap_with_driving_index = []
            walking_index = []
            # driving lanes
            for i, lane_id in enumerate(junc_lane_ids):
                lane = lanes[lane_id]
                if lane["type"] == mapv2.LANE_TYPE_WALKING:
                    phase[i] = (
                        lightv2.LIGHT_STATE_GREEN
                    )  # Set it to green first and then process it later
                    for overlap in lane["overlaps"]:
                        other_lane_id = overlap["other"]["lane_id"]
                        other_lane = lanes[other_lane_id]
                        if other_lane["type"] == mapv2.LANE_TYPE_DRIVING:
                            walking_overlap_with_driving_index.append(i)
                    walking_index.append(i)
                    continue
                if lane["turn"] == mapv2.LANE_TURN_RIGHT:
                    phase[i] = lightv2.LIGHT_STATE_GREEN
                    continue
                if is_green_func(lane_id):
                    phase[i] = lightv2.LIGHT_STATE_GREEN
                else:
                    phase[i] = lightv2.LIGHT_STATE_RED
            # walking lanes
            stand_alone_walking_index = set(walking_index) - set(
                walking_overlap_with_driving_index
            )
            # Generate signal control for sidewalks that overlap with roadways.
            for i in walking_overlap_with_driving_index:
                lane_id = junc_lane_ids[i]
                lane = lanes[lane_id]
                conflict_turn_type = mapv2.LANE_TURN_STRAIGHT
                for overlap in lane["overlaps"]:
                    other_lane_id = overlap["other"]["lane_id"]
                    other_lane = lanes[other_lane_id]
                    if (
                        other_lane["type"] == mapv2.LANE_TYPE_DRIVING
                        and other_lane["turn"] == conflict_turn_type
                        and phase[lane_id2index[other_lane_id]]
                        == lightv2.LIGHT_STATE_GREEN
                    ):
                        phase[i] = lightv2.LIGHT_STATE_RED
                        break
            # The sidewalk that does not overlap the roadway always has a green light
            for i in stand_alone_walking_index:
                phase[i] = lightv2.LIGHT_STATE_GREEN
            phases.append(
                {
                    "states": phase,
                }
            )
        # Part of the sidewalk that overlaps the roadway does not overlap with the straight road, so the light is always green, and a phase is added.
        lights_dict = defaultdict(set)
        for phase in phases:
            for i, l in enumerate(phase["states"]):
                lights_dict[i].add(l)
        if not all(lightv2.LIGHT_STATE_GREEN in ls for ls in lights_dict.values()):
            states = phases[-1]["states"]
            reversed_states = [
                lightv2.LIGHT_STATE_UNSPECIFIED for _ in range(len(junc_lane_ids))
            ]
            for i, s in enumerate(states):
                # right turn lanes without overlap, the light on the sidewalk that does not intersect with the roadway are always green.
                if (
                    is_no_overlap_drive_right(junc_lane_ids[i])
                    or i in stand_alone_walking_index
                ):
                    reversed_states[i] = lightv2.LIGHT_STATE_GREEN
                    continue
                if s == lightv2.LIGHT_STATE_RED:
                    reversed_states[i] = lightv2.LIGHT_STATE_GREEN
                elif s == lightv2.LIGHT_STATE_GREEN:
                    reversed_states[i] = lightv2.LIGHT_STATE_RED
            phases.append(
                {
                    "states": reversed_states,
                }
            )
        j["phases"] = phases


def generate_traffic_light(
    lanes: dict,
    roads: dict,
    juncs: dict,
    green_time: float,
    yellow_time: float,
    min_direction_group: int,
):
    # Generating available phases for MP
    logging.info("Generating available phases")
    _gen_available_phases(lanes, juncs, min_direction_group)
    # Generating fixed program traffic-lights
    logging.info("Generating fixed program")
    _gen_fixed_program(juncs, lanes, green_time, yellow_time)


def convert_traffic_light(
    lanes: dict,
    roads: dict,
    juncs: dict,
    id_mapping: dict,
    green_time: float,
    yellow_time: float,
    min_direction_group: int,
    traffic_light_path: Optional[str] = None,
):
    # Generating available phases for MP
    logging.info("Generating available phases")
    _gen_available_phases(lanes, juncs, min_direction_group)
    # Generating fixed program traffic-lights or convert SUMO traffic-lights
    logging.info("Generating fixed program")
    _convert_fixed_program(
        lanes, roads, juncs, green_time, yellow_time, id_mapping, traffic_light_path
    )
