"""
OD generate constants
"""

import pycityproto.city.geo.v2.geo_pb2 as geov2
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
    "HSOSH": 0.00,
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
PT_START_ID = 1_0000_0000
PRIMARY_SCHOOL, JUNIOR_HIGH_SCHOOL, HIGH_SCHOOL, COLLEGE, BACHELOR, MASTER, DOCTOR = (
    personv1.EDUCATION_PRIMARY_SCHOOL,
    personv1.EDUCATION_JUNIOR_HIGH_SCHOOL,
    personv1.EDUCATION_HIGH_SCHOOL,
    personv1.EDUCATION_COLLEGE,
    personv1.EDUCATION_BACHELOR,
    personv1.EDUCATION_MASTER,
    personv1.EDUCATION_DOCTOR,
)
EDUCATION_LEVELS = [
    PRIMARY_SCHOOL,
    JUNIOR_HIGH_SCHOOL,
    HIGH_SCHOOL,
    COLLEGE,
    BACHELOR,
    MASTER,
    DOCTOR,
]
# probabilities
EDUCATION_STATS = [1 / len(EDUCATION_LEVELS) for _ in range(len(EDUCATION_LEVELS))]
CONSUMPTION_LEVELS = [
    personv1.CONSUMPTION_LOW,
    personv1.CONSUMPTION_RELATIVELY_LOW,
    personv1.CONSUMPTION_MEDIUM,
    personv1.CONSUMPTION_RELATIVELY_HIGH,
    personv1.CONSUMPTION_HIGH,
]
# probabilities
CONSUMPTION_STATS = [
    1 / len(CONSUMPTION_LEVELS) for _ in range(len(CONSUMPTION_LEVELS))
]
GENDERS = [
    personv1.GENDER_FEMALE,
    personv1.GENDER_MALE,
]
# probabilities
GENDER_STATS = [1 / len(GENDERS) for _ in range(len(GENDERS))]
AGES = [i for i in range(8, 75)]
# probabilities
AGE_STATS = [1 / len(AGES) for _ in range(len(AGES))]
# work catg
WORK_CATGS = {"business", "industrial", "administrative"}
# education catg
EDUCATION_CATGS = {"education"}
# home catg
HOME_CATGS = {"residential"}
PT_DRIVER_ATTRIBUTES = {
    "BUS": {
        "length": 15,
        "width": 2,
        "max_speed": 41.666666666666664,
        "max_acceleration": 3,
        "max_braking_acceleration": -10,
        "usual_acceleration": 2,
        "usual_braking_acceleration": -4.5,
    },
    "SUBWAY": {
        "length": 25,
        "width": 2,
        "max_speed": 41.666666666666664,
        "max_acceleration": 3,
        "max_braking_acceleration": -10,
        "usual_acceleration": 2,
        "usual_braking_acceleration": -4.5,
    },
    "UNSPECIFIED": {
        "length": 5,
        "width": 2,
        "max_speed": 41.666666666666664,
        "max_acceleration": 3,
        "max_braking_acceleration": -10,
        "usual_acceleration": 2,
        "usual_braking_acceleration": -4.5,
    },
}
