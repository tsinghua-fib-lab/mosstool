from itertools import product
import os
from typing import Dict, Tuple, Union, cast
import pandas as pd

from pyproj import Proj
import shapely
import warnings

from ...type import Map, LaneType, LanePosition

__all__ = ["Convertor"]


class Convertor:
    """
    convert mosstool map to GMNS map
    """

    def __init__(self, m: Map):
        self.m = m
        self.projector = Proj(self.m.header.projection)
        self._roads = {road.id: road for road in self.m.roads}
        self._road2nodes: Dict[int, Tuple[str, str]] = {}
        """road id -> (from node, to node)"""
        self._lanes = {lane.id: lane for lane in self.m.lanes}
        self._lane_shapelys: Dict[int, shapely.geometry.LineString] = {}
        for lane in self.m.lanes:
            lnglats = self.projector(
                [node.x for node in lane.center_line.nodes],
                [node.y for node in lane.center_line.nodes],
                inverse=True,
            )
            self._lane_shapelys[lane.id] = shapely.geometry.LineString(
                list(zip(*lnglats))
            )
        self._driving_lane_ids = {
            lane.id for lane in self.m.lanes if lane.type == LaneType.LANE_TYPE_DRIVING
        }
        self._aois = {aoi.id: aoi for aoi in self.m.aois}
        self._aoi_shapelys: Dict[
            int, Union[shapely.geometry.Point, shapely.geometry.Polygon]
        ] = {}
        for aoi in self.m.aois:
            lnglats = self.projector(
                [node.x for node in aoi.positions],
                [node.y for node in aoi.positions],
                inverse=True,
            )
            if aoi.HasField("area"):
                self._aoi_shapelys[aoi.id] = shapely.geometry.Polygon(
                    list(zip(*lnglats))
                )
            else:
                self._aoi_shapelys[aoi.id] = shapely.geometry.Point(
                    lnglats[0][0], lnglats[1][0]
                )
        # used for checking route result after static traffic assignment
        self._connected_road_pairs: Dict[Tuple[int, int], int] = (
            {}
        )  # {(start_road_id, end_road_id): junction_id}
        self._not_perfect_junction_ids = set()

    def _to_nodes(self):
        """
        convert junctions to GMNS nodes

        Return:
        - pd.DataFrame: GMNS nodes
        """
        data = []
        for j in self.m.junctions:
            # 1. check road connectivity
            group_set = {(g.in_road_id, g.out_road_id) for g in j.driving_lane_groups}
            self._connected_road_pairs.update(
                {(g.in_road_id, g.out_road_id): j.id for g in j.driving_lane_groups}
            )
            in_road_ids = set()
            out_road_ids = set()
            for lane_id in j.lane_ids:
                lane = self._lanes[lane_id]
                if lane.type != LaneType.LANE_TYPE_DRIVING:
                    continue
                pre_lane_id = (
                    lane.predecessors[0].id if len(lane.predecessors) > 0 else None
                )
                suc_lane_id = (
                    lane.successors[0].id if len(lane.successors) > 0 else None
                )
                if pre_lane_id is not None:
                    pre_lane = self._lanes[pre_lane_id]
                    in_road_ids.add(pre_lane.parent_id)
                if suc_lane_id is not None:
                    suc_lane = self._lanes[suc_lane_id]
                    out_road_ids.add(suc_lane.parent_id)
            # consider all the pairs of in_roads and out_roads, there should be a driving group containing the pair
            for in_road_id, out_road_id in product(in_road_ids, out_road_ids):
                if (in_road_id, out_road_id) not in group_set:
                    # warnings.warn(
                    #     f"junction {j.id} has no driving group for in_road {in_road_id} and out_road {out_road_id}"
                    # )
                    self._not_perfect_junction_ids.add(j.id)
            # 2. compute the junction center
            multiline = shapely.geometry.MultiLineString(
                [self._lane_shapelys[lane_id] for lane_id in j.lane_ids]
            )
            lngs, lats = multiline.centroid.xy
            lng, lat = lngs[0], lats[0]
            # 3. save to df
            id = str(j.id)
            data.append(
                {
                    "name": id,
                    "node_id": id,
                    "zone_id": id,
                    "x_coord": lng,
                    "y_coord": lat,
                    "geometry": f"POINT({lng} {lat})",
                }
            )
        for id, a in self._aois.items():
            lngs, lats = self._aoi_shapelys[id].centroid.xy
            lng, lat = lngs[0], lats[0]
            data.append(
                {
                    "name": f"{id}-start",
                    "node_id": f"{id}-start",
                    "zone_id": f"{id}-start",
                    "x_coord": lng,
                    "y_coord": lat,
                    "geometry": f"POINT({lng} {lat})",
                }
            )
            data.append(
                {
                    "name": f"{id}-end",
                    "node_id": f"{id}-end",
                    "zone_id": f"{id}-end",
                    "x_coord": lng,
                    "y_coord": lat,
                    "geometry": f"POINT({lng} {lat})",
                }
            )
        data.sort(key=lambda x: x["name"])
        df = pd.DataFrame(data)
        if len(self._not_perfect_junction_ids) > 0:
            warnings.warn(
                f"junctions {self._not_perfect_junction_ids} have no driving group for some in_roads and out_roads, which will cause some traffic assignment results to be invalid"
            )
        return df

    def _to_lines(self):
        """
        convert lanes to GMNS lines

        Return:
        - pd.DataFrame: GMNS lines
        """
        # links name,link_id,from_node_id,to_node_id,facility_type,dir_flag,length,lanes,capacity,free_speed,link_type,cost,VDF_fftt1,VDF_cap1,VDF_alpha1,VDF_beta1,VDF_PHF1,VDF_gamma1,VDF_mu1
        data = []
        # add virtual nodes for road without predecessors or successors
        virtual_nodes = []
        for road in self.m.roads:
            dls = [i for i in road.lane_ids if i in self._driving_lane_ids]
            lane = self._lanes[dls[0]]
            from_lane_id = None
            for l in dls:
                l0 = self._lanes[l]
                if len(l0.predecessors) > 0:
                    from_lane_id = l0.predecessors[0].id
                    break
            to_lane_id = None
            for l in dls:
                l0 = self._lanes[l]
                if len(l0.successors) > 0:
                    to_lane_id = l0.successors[0].id
                    break

            if from_lane_id is None:
                lng, lat = self._lane_shapelys[dls[0]].coords[0]
                from_node = f"{road.id}-start"
                virtual_nodes.append(
                    {
                        "name": from_node,
                        "node_id": from_node,
                        "zone_id": from_node,
                        "x_coord": lng,
                        "y_coord": lat,
                        "geometry": f"POINT({lng} {lat})",
                    }
                )
            else:
                from_node = str(self._lanes[from_lane_id].parent_id)

            if to_lane_id is None:
                lng, lat = self._lane_shapelys[dls[0]].coords[-1]
                to_node = f"{road.id}-end"
                virtual_nodes.append(
                    {
                        "name": to_node,
                        "node_id": to_node,
                        "zone_id": to_node,
                        "x_coord": lng,
                        "y_coord": lat,
                        "geometry": f"POINT({lng} {lat})",
                    }
                )
            else:
                to_node = str(self._lanes[to_lane_id].parent_id)

            length = lane.length / 1609.34  ## convert to miles
            geometry = "LINESTRING("
            for xy in lane.center_line.nodes:
                lon, lat = self.projector(xy.x, xy.y, inverse=True)
                geometry += f"{lon} {lat},"
            geometry = geometry[:-1] + ")"

            n_lane = len(dls)
            speed = lane.max_speed * 3.6 / 1.60934  ## convert to miles/hour

            capacity = 30 * min(lane.max_speed * 3.6, 70)
            data.append(
                {
                    "name": str(road.id),
                    "link_id": str(road.id),
                    "from_node_id": from_node,
                    "to_node_id": to_node,
                    "length": length,
                    "lanes": n_lane,
                    "free_speed": speed,
                    "geometry": geometry,
                    "capacity": capacity,
                    "link_type": 1,  # 1: one direction, 2: two way
                }
            )
            self._road2nodes[road.id] = (from_node, to_node)
        # link aoi-start to the driving positions to the successor junction
        # link aoi-end to the driving positions to the predecessor junction
        for id, a in self._aois.items():
            lngs, lats = self._aoi_shapelys[id].centroid.xy
            lng, lat = lngs[0], lats[0]
            for pos in a.driving_positions:
                pos = cast(LanePosition, pos)
                lane = self._lanes[pos.lane_id]
                lane_shapely = self._lane_shapelys[pos.lane_id]
                speed = 30 / 1.60934  # slow speed
                s = pos.s
                if len(lane.successors) > 0:
                    # link aoi-start
                    suc_lane = self._lanes[lane.successors[0].id]
                    from_node = f"{id}-start"
                    to_node = str(suc_lane.parent_id)
                    length = max(lane.length - s, 5) / 1609.34
                    lane_end = lane_shapely.coords[-1]
                    data.append(
                        {
                            "name": f"{id}-start-{lane.parent_id}-{to_node}",
                            "link_id": f"{id}-start-{lane.parent_id}-{to_node}",
                            "from_node_id": from_node,
                            "to_node_id": to_node,
                            "length": length,
                            "lanes": 1,
                            "free_speed": speed,
                            "geometry": f"LINESTRING({lng} {lat},{lane_end[0]} {lane_end[1]})",
                            "capacity": 30 * 30,
                            "link_type": 1,  # 1: one direction, 2: two way
                        }
                    )
                if len(lane.predecessors) > 0:
                    # link aoi-end
                    pre_lane = self._lanes[lane.predecessors[0].id]
                    from_node = str(pre_lane.parent_id)
                    to_node = f"{id}-end"
                    length = max(s, 5) / 1609.34
                    lane_start = lane_shapely.coords[0]
                    data.append(
                        {
                            "name": f"{id}-end-{lane.parent_id}-{from_node}",
                            "link_id": f"{id}-end-{lane.parent_id}-{from_node}",
                            "from_node_id": from_node,
                            "to_node_id": to_node,
                            "length": length,
                            "lanes": 1,
                            "free_speed": speed,
                            "geometry": f"LINESTRING({lane_start[0]} {lane_start[1]},{lng} {lat})",
                            "capacity": 30 * 30,
                            "link_type": 1,  # 1: one direction, 2: two way
                        }
                    )
        data.sort(key=lambda x: x["name"])
        virtual_nodes.sort(key=lambda x: x["name"])
        df = pd.DataFrame(data)
        virtual_nodes_df = pd.DataFrame(virtual_nodes)
        return df, virtual_nodes_df

    def save(self, work_dir: str):
        """
        save GMNS map to work_dir

        Args:
        - work_dir (str): work directory
        """
        nodes = self._to_nodes()
        lines, virtual_nodes = self._to_lines()
        nodes = pd.concat([nodes, virtual_nodes], ignore_index=True)
        nodes.to_csv(os.path.join(work_dir, "node.csv"), index=False)
        lines.to_csv(os.path.join(work_dir, "link.csv"), index=False)
