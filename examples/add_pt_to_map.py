import json
import logging
import pickle

from mosstool.map.builder import Builder
from mosstool.type import Map
from mosstool.util.format_converter import dict2pb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
# input
PT_PATH = "./data/geojson/wuhan_pt.json"
# map build from `./examples/build_map.py`
ORIG_MAP_PATH = f"./data/temp/map.pb"
# output
MAP_PB_PATH = "./data/temp/srt.raw_pt_map.pb"
MAP_PKL_PATH = "./data/temp/srt.raw_pt_map.pkl"
with open(PT_PATH, "r") as f:
    pt = json.load(f)
# map build from `./map_generation/build_map.py`
with open(ORIG_MAP_PATH, "rb") as f:
    net = Map()
    net.ParseFromString(f.read())
builder = Builder(
    net=net,
    public_transport=pt,
    proj_str=net.header.projection,
    gen_sidewalk_speed_limit=50 / 3.6,
    aoi_mode="append",
    road_expand_mode="M",
)
m_dict = builder.build("test")

pickle.dump(m_dict, open(MAP_PKL_PATH, "wb"))
with open(MAP_PB_PATH, "wb") as f:
    pb = dict2pb(m_dict, Map())
    f.write(pb.SerializeToString())
