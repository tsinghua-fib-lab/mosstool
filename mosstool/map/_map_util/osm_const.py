"""
OSM constants
"""

WAY_FILTER = [
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
]
MAX_SPEEDS = {
    "motorway": 120,
    "trunk": 90,
    "primary": 60,
    "secondary": 50,
    "tertiary": 40,
    "unclassified": 40,
    "residential": 30,
    "motorway_link": 30,
    "trunk_link": 30,
    "primary_link": 30,
    "secondary_link": 30,
    "tertiary_link": 30,
    "service": 20,
    "junction": 30,
    "walk": 30,
    "default": 60,
}
OVERPASS_FILTER = '["highway"]["area"!~"yes"]["access"!~"private"]["highway"!~"abandoned|construction|planned|platform|proposed|raceway"]["service"!~"private"]'
DEFAULT_LANES = {
    "motorway": 4,
    "trunk": 4,
    "primary": 4,
    "secondary": 3,
    "tertiary": 2,
    "unclassified": 2,
    "residential": 2,
    "motorway_link": 3,
    "trunk_link": 3,
    "primary_link": 3,
    "secondary_link": 2,
    "tertiary_link": 1,
    "service": 1,
    "default": 2,
}
TURN_CONFIG = {
    1: "ALSR",
    2: "ALS|SR",
    3: "AL|S|R",
    4: "AL|S|S|R",
    5: "AL|L|S|S|R",
}
