"""
OD generate constants
"""

import pycityproto.city.map.v2.map_pb2 as mapv2
import pycityproto.city.person.v1.person_pb2 as personv1
import pycityproto.city.trip.v2.trip_pb2 as tripv2

DIS_CAR = 1000
DIS_BIKE = 500
V_CAR = 20 / 3.6
V_BIKE = 10 / 3.6
HUMAN_MODE_STATS = {
    "HWH": 18.79,
    "HWH+": 20.03,
    "HW+WH": 13.76,
    "HWHWH": 1.09,
    "HSH": 3.1,
    "HSH+": 3.36,
    "HOH": 10.91,
    "HOH+": 11.45,
    "HWHWH+": 1.43,
}
BUS = tripv2.TRIP_MODE_BUS_WALK
CAR = tripv2.TRIP_MODE_DRIVE_ONLY
BIKE = tripv2.TRIP_MODE_BIKE_WALK
WALK = tripv2.TRIP_MODE_WALK_ONLY
LANE_TYPE_DRIVING = mapv2.LANE_TYPE_DRIVING
ALL_TRIP_MODES = [
    BUS,
    BUS,
    CAR,
    CAR,
    WALK,
]
