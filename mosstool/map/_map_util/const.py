"""
MOSS Map Tools constants
"""

import numpy as np

EPS = 1e-4

LANE_START_ID = 0_0000_0000
ROAD_START_ID = 2_0000_0000
JUNC_START_ID = 3_0000_0000
AOI_START_ID = 5_0000_0000
POI_START_ID = 7_0000_0000
PUBLIC_LINE_START_ID = 9_0000_0000

CLUSTER_PENALTY = 0.01  # Clustering penalty term
UP_THRESHOLD = 0.01  # Select the one with the largest number of clusters within the threshold range
HAS_WALK_LANES_HIGHWAY = [
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "primary_link",
    "secondary_link",
    "tertiary_link",
    "residential",
    "service",
]
SUMO_WAY_LEVELS = [
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
    "residential",
    "service",
    "footway",
]  # TO be converted SUMO road levels

MAX_JUNC_LANE_LENGTH = 500
MIN_WALK_CONNECTED_COMPONENT = 5  # Minimum length of connected component of sidewalk
DEFAULT_ROAD_SPLIT_LENGTH = 20  # Default road shortening length
DEFAULT_JUNCTION_WALK_LANE_WIDTH = 4.0  # Default sidewalk width
D_DIS_GATE = 35
W_DIS_GATE = 30
STOP_DIS_GATE = 30
STOP_HUGE_GATE = 50
D_HUGE_GATE = 505
W_HUGE_GATE = 500
LENGTH_PER_DOOR = 150
MAX_DOOR_NUM = 8
AOI_GATE_OFFSET = 10
SQRT2 = 2**0.5
COVER_GATE = 0.8  # aoi whose area is covered beyond this portion is covered
SMALL_GATE = 400  # aoi whose area less than this is small
AOI_MERGE_GRID = 3000
DIS_GATE = 18  # match poi to nearest aoi within distance gate
DOUBLE_DIS_GATE = 2 * DIS_GATE
SCALE = 1.3  # small aois are scaled up to see whether they can be connected
THIS_IS_COVERED_POI = 0
THIS_IS_PROJECTED_POI = 1
THIS_IS_ISOLATE_POI = 2
UPSAMPLE_FACTOR = 4
ISOLATED_POI_CATG = set()
DEFAULT_LANE_NUM = 1
MAIN_WAY_ANGLE_THRESHOLD = np.pi / 12
AROUND_WAY_ANGLE_THRESHOLD = np.pi / 6
SAME_DIREC_THRESHOLD = np.pi / 18
CURVATURE_THRESHOLDS = {
    "AROUND": 0.028512,  # When taking this value, the calculated speed limit threshold is exactly 30km/h
    "LEFT": 0.028512,
    "RIGHT": 0.028512,
}
MAX_WAY_DIS_DIFFERENCE = 8
STATION_CAPACITY = {
    "BUS": 35,
    "SUBWAY": 850,
}
DEFAULT_SCHEDULES = {
    "BUS": [t for t in range(int(5.5 * 3600), int(22.5 * 3600), int(5 * 60))],
    "SUBWAY": [t for t in range(int(5 * 3600), int(23 * 3600), int(7 * 60))],
}
DEFAULT_MAX_SPEED = {
    "SUBWAY": 100 / 3.6,
    "WALK": 30 / 3.6,
    "BUS": 10 / 3.6,
}
DEFAULT_STATION_LENGTH = {
    "BUS": 10,
    "SUBWAY": 500,
}
DEFAULT_STATION_DURATION = {
    "BUS": 100,
    "SUBWAY": 180,
}
MAX_BATCH_SIZE = 15_0000
MIN_HAS_WALK_LANE_LENGTH = 5
LARGE_LANE_NUM_THRESHOLD = 4
SMALL_LANE_NUM_THRESHOLD = 2
DEFAULT_TURN_NUM = {
    "AUXILIARY_SMALL_LEFT": 1,
    "AUXILIARY_SMALL_RIGHT": 1,
    "AUXILIARY_LARGE_LEFT": 1,
    "AUXILIARY_LARGE_RIGHT": 1,
    "AUXILIARY_AROUND": 1,
    "MAIN_AROUND": 1,
    "MAIN_SMALL_LEFT": 1,
    "MAIN_LARGE_LEFT": 2,
    "MAIN_SMALL_RIGHT": 1,
    "MAIN_LARGE_RIGHT": 2,
}
