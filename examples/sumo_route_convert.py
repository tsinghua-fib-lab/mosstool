import logging

from mosstool.map.sumo.map import MapConverter
from mosstool.trip.sumo.route import RouteConverter
from mosstool.type import Map, Persons
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
id2uid = s2m.get_sumo_id_mappings()
map_pb = dict2pb(m, Map())
s2r = RouteConverter(
    converted_map=map_pb,
    sumo_id_mappings=id2uid,
    route_path="./data/sumo/trips.xml",
    additional_path="./data/sumo/additional.xml",
)
r = s2r.convert_route()
pb = dict2pb(r, Persons())
with open("data/temp/sumo_persons.pb", "wb") as f:
    f.write(pb.SerializeToString())
