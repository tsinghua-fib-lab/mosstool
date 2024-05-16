import os

from pycityproto.city.geo.v2.geo_pb2 import LanePosition, Position
from pycityproto.city.map.v2.map_pb2 import Map
from pymongo import MongoClient

from ..format_converter import dict2pb, json2pb, pb2dict, pb2json, coll2pb, pb2coll


def test_pb2json2pb():
    pb = Position(lane_position=LanePosition(lane_id=1, s=2))
    json = pb2json(pb)
    print(json)
    pb2 = json2pb(json, Position())
    assert pb == pb2


def test_pb2dict2pb():
    pb = Position(lane_position=LanePosition(lane_id=1, s=2))
    d = pb2dict(pb)
    print(d)
    assert d == {"lane_position": {"lane_id": 1, "s": 2}}
    pb2 = dict2pb(d, Position())
    assert pb == pb2


def test_coll2pb2coll():
    client = MongoClient(os.environ["MONGO_URI"])
    coll = client[os.environ["MAP_DB"]][os.environ["MAP_COLL"]]
    pb = Map()
    pb = coll2pb(coll, pb)
    print(pb)
    assert pb.header.name == "qinglonghu"
    assert len(pb.roads) > 0
    assert len(pb.junctions) > 0
    assert len(pb.aois) > 0
    assert len(pb.lanes) > 0
    assert len(pb.pois) >= 0

    coll = client[os.environ["TEST_DB"]][os.environ["TEST_COLL"]]
    coll.drop()
    pb2coll(pb, coll)
