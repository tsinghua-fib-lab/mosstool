import logging

from mosstool.map.sumo.map import MapConverter
from mosstool.type import Map
from mosstool.util.format_converter import dict2pb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
s2m = MapConverter(
    net_path="data/sumo/shenzhen.net.xml",
    traffic_light_path="data/sumo/trafficlight.xml",
    poly_path="data/sumo/poly.xml",
)
m = s2m.convert_map()
pb = dict2pb(m, Map())
with open("data/temp/sumo_map.pb", "wb") as f:
    f.write(pb.SerializeToString())
