from mosstool.type import Map
from mosstool.util.map_merger import merge_map

maps = []
for i in [0, 1]:
    with open(f"data/temp/test_{i}.pb", "rb") as f:
        pb = Map()
        pb.ParseFromString(f.read())
        maps.append(pb)
merge_map(
    partial_maps=maps,
    output_path="data/temp/merged_m.pb",
)
