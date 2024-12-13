import logging
import math
import os
from collections import Counter
from collections.abc import Callable
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from mosstool.map.builder import Builder
from mosstool.map.osm import RoadNet
from mosstool.type import Map
from mosstool.util.format_converter import dict2pb


# The following will outline the brief steps for importing from CAD to xlsx files, using AutoCAD 2025 - 简体中文 (Simplified Chinese) as an example.
# 1. Navigate to and click on "插入-链接" and "提取-提取数据". In the "界面-“数据提取-开始", click "创建新数据提取" then proceed to next.
# 2. After clicking "next", a prompt appears to save the data extraction as a "*.dxe" file. Name it and click "save".
# 3. After saving, the interface "界面-“数据提取-定义数据源" opens. Click on "数据源-图形/图纸集" and check "包括当前图形". Then click "next".
# 4. In the "数据提取-选取对象" interface, uncheck "显示所有对象类型", click "仅显示块", then click "next".
# 5. In the "数据提取-选择特性" interface, keep default selections and click "next".
# 6. In the "数据提取-优化数据" interface, keep default settings and click "next".
# 7. In the "数据提取-选择输出" interface, check "将数据提取处理表插入图形" and "将数据输入至外部文件". Click "...", choose ".xlsx" type in "另存为" dialog, input save path and filename, click "save", then click "next" in "选择输出".
# 8. In the "数据提取-表格样式" interface, keep default settings and click "next".
# 9. In the "数据提取-完成" interface, click finish to complete the attribute extraction and get the output Excel file.
def cad2osm(
    cad_path: str,
    node_start_id: int = 0,
    way_start_id: int = 0,
    merge_gate: float = 0.0,
    x_transform_func: Optional[Callable[[float], float]] = None,
    y_transform_func: Optional[Callable[[float], float]] = None,
) -> list[dict]:
    df = pd.read_excel(cad_path)
    df = df.fillna(0)
    if x_transform_func is None:
        x_transform_func = lambda x: x
    if y_transform_func is None:
        y_transform_func = lambda y: y
    node_id, way_id = node_start_id, way_start_id
    osm_data: list[dict] = []
    ANGLE_STEP = 1
    _edge_node_ids: set[int] = set()
    for i, now_row in df.iterrows():
        if now_row["名称"] == "直线":
            _edge_node_ids.add(node_id)
            n = {
                "type": "node",
                "id": node_id,
                "x": float(now_row["起点 X"]),
                "y": float(now_row["起点 Y"]),
            }
            osm_data.append(n)
            node_id += 1

            _edge_node_ids.add(node_id)
            n = {
                "type": "node",
                "id": node_id,
                "x": float(now_row["端点 X"]),
                "y": float(now_row["端点 Y"]),
            }
            osm_data.append(n)
            node_id += 1

            w = {
                "type": "way",
                "id": way_id,
                "nodes": [node_id - 2, node_id - 1],
                "tags": {
                    "highway": "tertiary"
                },  # here we set the same tag for all ways
                "others": now_row,
            }
            way_id += 1

            osm_data.append(w)
        elif now_row["名称"] == "圆弧":
            c_x = float(now_row["中心 X"])
            c_y = float(now_row["中心 Y"])
            start_angle = float(now_row["起点角度"])
            r = float(now_row["半径"])
            agl = 0
            nodes_indexes = []
            total_angle = float(now_row["总角度"])
            for agl in np.linspace(
                0, total_angle, max(int(total_angle / ANGLE_STEP), 3)
            ):
                n = {
                    "type": "node",
                    "id": node_id,
                    "x": c_x + r * math.cos(math.pi * (agl + start_angle) / 180),
                    "y": c_y + r * math.sin(math.pi * (agl + start_angle) / 180),
                }
                osm_data.append(n)
                nodes_indexes.append(n["id"])
                node_id += 1
            _edge_node_ids.add(nodes_indexes[0])
            _edge_node_ids.add(nodes_indexes[-1])
            w = {
                "type": "way",
                "id": way_id,
                "nodes": nodes_indexes,
                "tags": {
                    "highway": "tertiary"
                },  # here we set the same tag for all ways
                "others": now_row,
            }
            osm_data.append(w)
            way_id += 1
    for i in osm_data:
        for _key, _transform_func in zip(
            ["x", "y"], [x_transform_func, y_transform_func]
        ):
            if _key in i:
                i[_key] = _transform_func(i[_key])
    to_merge_xys_dict = {
        i["id"]: (i["x"], i["y"]) for i in osm_data if "x" in i and "y" in i
    }
    tree_id_to_node_id = {idx: i for idx, i in enumerate(to_merge_xys_dict.keys())}
    tree = KDTree([v for v in to_merge_xys_dict.values()])  # type:ignore
    merged = set()
    father_id_dict = {
        i: i for i in range(node_id)
    }  # Initialize the parent node id of each node
    for _node_idx, xy in to_merge_xys_dict.items():
        if _node_idx in merged:
            continue
        a = [
            tree_id_to_node_id[i] for i in tree.query_ball_point(xy, merge_gate) # type:ignore
        ]  
        if len(a) == 1:
            unique_node_idx = a.pop()
            father_id_dict[unique_node_idx] = _node_idx
            merged.add(unique_node_idx)
        else:
            visited_nids = {_node_idx}
            while len(a) > 0:
                b = []
                for i in a:
                    if i in visited_nids:
                        continue
                    _xy = to_merge_xys_dict[i]
                    b.extend(
                        [
                            tree_id_to_node_id[j]
                            for j in tree.query_ball_point(_xy, merge_gate) # type:ignore
                        ]  
                    )
                    visited_nids.add(i)
                    father_id_dict[i] = _node_idx
                a, b = b, []
            merged |= visited_nids
    for i in range(node_id):
        while father_id_dict[i] != father_id_dict[father_id_dict[i]]:
            father_id_dict[i] = father_id_dict[father_id_dict[i]]
    to_delete_ids = set()
    for i in osm_data:
        if i["type"] == "way":
            i["nodes"] = [father_id_dict[n] for n in i["nodes"]]
            if not len(set(i["nodes"])) >= 2:
                to_delete_ids.add(("way", i["id"]))
        elif i["type"] == "node" and father_id_dict[i["id"]] != i["id"]:
            to_delete_ids.add(("node", i["id"]))
    return [i for i in osm_data if (i["type"], i["id"]) not in to_delete_ids]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
PROJ_STR = "+proj=tmerc +lat_0=33.9 +lon_0=116.4"
os.makedirs("cache", exist_ok=True)
rn = RoadNet(
    proj_str=PROJ_STR,
)
CAD_PATH = "./data/cad/cad.xlsx"
osm_data = cad2osm(
    CAD_PATH,
    x_transform_func=lambda x: x - 229036.4782002,
    y_transform_func=lambda y: y - 214014.32078879,
)
import pickle

pickle.dump(osm_data, open("./cache/osm_data.pkl", "wb"))
print(Counter(i["type"] for i in osm_data))
path = "cache/topo_from_cad.geojson"
net = rn.create_road_net(path, osm_data_cache=osm_data)

builder = Builder(
    net=net,
    gen_sidewalk_speed_limit=50 / 3.6,
    road_expand_mode="M",
    proj_str=PROJ_STR,
)
m = builder.build("test")
pb = dict2pb(m, Map())
with open("data/temp/cad_map.pb", "wb") as f:
    f.write(pb.SerializeToString())
