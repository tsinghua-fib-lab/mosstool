import logging

import geojson

from mosstool.type import Map
from mosstool.util.map_splitter import split_map

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
with open("data/geojson/split_box.geojson", "r") as f:
    bbox = geojson.load(f)
with open("data/map/m.pb", "rb") as f:
    pb = Map()
    pb.ParseFromString(f.read())
split_map(
    geo_data=bbox,
    map=pb,
    output_path="data/temp",
)
