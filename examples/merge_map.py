import logging

from mosstool.type import Map
from mosstool.map.builder import Builder
from mosstool.util.map_merger import merge_map
from mosstool.util.format_converter import dict2pb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

REBUILD_MAP = True
maps = []
for i in [0, 1]:
    with open(f"data/temp/test_{i}.pb", "rb") as f:
        pb = Map()
        pb.ParseFromString(f.read())
        maps.append(pb)
merged_pb = merge_map(
    partial_maps=maps,
    output_path="data/temp/merged_m.pb",
)
if REBUILD_MAP:
    builder = Builder(
    net=merged_pb,
    proj_str=merged_pb.header.projection,
    gen_sidewalk_speed_limit=50 / 3.6,
    aoi_mode="append", # keep AOIs in merged_pb
    road_expand_mode="M",
)
    rebuild_m = builder.build(merged_pb.header.name)
    rebuild_pb = dict2pb(rebuild_m, Map())
    with open("data/temp/rebuild_merged_m.pb", "wb") as f:
        f.write(rebuild_pb.SerializeToString())

