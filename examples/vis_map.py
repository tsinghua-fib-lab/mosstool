import os

from pymongo import MongoClient

from mosstool.map.vis import VisMap
from mosstool.type import Map
from mosstool.util.format_converter import coll2pb

client = MongoClient(os.environ["MONGO_URI"])
coll = client[os.environ["MAP_DB"]][os.environ["MAP_COLL"]]
pb = Map()
pb = coll2pb(coll, pb)
m = VisMap(pb)
print(m.lane_shapely_xys[0])
deck = m.visualize()
deck.to_html("map.html")
