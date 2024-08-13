import json
import logging

from mosstool.map.public_transport.public_transport_post import \
    public_transport_process
from mosstool.util.format_converter import dict2pb, pb2dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
from mosstool.map.builder import Builder
from mosstool.type import Map, Persons

with open("data/temp/wuhan_pt.json", "r") as f:
    pt = json.load(f)
# map build from `./examples/build_map.py`
with open(f"data/temp/map.pb", "rb") as f:
    net = Map()
    net.ParseFromString(f.read())
builder = Builder(
    net=net,
    public_transport=pt,
    proj_str="+proj=tmerc +lat_0=30.491 +lon_0=114.504",
    gen_sidewalk_speed_limit=50 / 3.6,
    aoi_mode="append",
    road_expand_mode="M",
)
m_dict = builder.build("test")
# pre-route
# run: ./routing -map data/temp/map.pb
new_m = public_transport_process(m_dict, "http://localhost:52101")
pb = dict2pb(new_m, Map())
with open("data/temp/map_with_pt.pb", "wb") as f:
    f.write(pb.SerializeToString())
