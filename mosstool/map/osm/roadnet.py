import copy
import logging
from collections import defaultdict
from itertools import combinations
from math import hypot
from typing import Dict, FrozenSet, List, Optional, Set

import networkx as nx
import pyproj
import requests
from geojson import Feature, FeatureCollection, LineString, MultiPoint, dump
from tqdm import tqdm

from .._map_util.osm_const import *
from ._motif import close_nodes, motif_H, suc_is_close_by_other_way
from ._wayutil import merge_way_nodes, parse_osm_way_tags

__all__ = ["RoadNet"]


class RoadNet:
    """
    Process OSM raw data to road and junction as geojson formats
    """

    def __init__(
        self,
        proj_str: str,
        max_longitude: float,
        min_longitude: float,
        max_latitude: float,
        min_latitude: float,
        wikipedia_name: Optional[str] = None,
        proxies: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
        - proj_str (str): projection string, e.g. 'epsg:3857'
        - max_longitude (float): max longitude
        - min_longitude (float): min longitude
        - max_latitude (float): max latitude
        - min_latitude (float): min latitude
        - wikipedia_name (str): wikipedia name of the area in OSM.
        - proxies (dict): proxies for requests, e.g. {'http': 'http://localhost:1080', 'https': 'http://localhost:1080'}
        """
        # configs
        self.proj_parameter = proj_str
        self.projector = pyproj.Proj(self.proj_parameter)
        self.bbox = (
            min_latitude,
            min_longitude,
            max_latitude,
            max_longitude,
        )
        self.proxies = proxies
        self.wikipedia_name = wikipedia_name
        self.way_filter = WAY_FILTER
        # OSM raw data
        self.ways = {}  # way_id -> way
        self.nodes = {}  # node_id -> node
        self.node_ways = defaultdict(list)  # node_id -> way_id list
        self.node_wayset = defaultdict(set)  # node_id -> way_id set

        # junctions
        self.junction_next_id = 500_0000_0000
        self.junction2nodes: Dict[int, List[int]] = {}
        self.node2junction: Dict[int, int] = {}

    @property
    def default_way_settings(self):
        default_lanes = DEFAULT_LANES
        max_speeds = {i: j / 3.6 for i, j in MAX_SPEEDS.items()}
        turn_config = TURN_CONFIG
        return {"lane": default_lanes, "max_speed": max_speeds, "turn": turn_config}

    def _download_osm(self):
        """Fetch raw data from OpenStreetMap"""
        bbox_str = ",".join(str(i) for i in self.bbox)
        query = f"[out:json][timeout:180][bbox:{bbox_str}];"
        wikipedia_name = self.wikipedia_name
        if wikipedia_name is not None:
            query += f'area[wikipedia="{wikipedia_name}"]->.searchArea;'
        area = "(area.searchArea)" if wikipedia_name is not None else ""
        query += f"(way{area}{OVERPASS_FILTER};>;);out;"
        logging.info(query)
        logging.info("Downloading map from OSM")
        data = requests.get(
            f"http://overpass-api.de/api/interpreter?data={query}",
            proxies=self.proxies,
        ).json()["elements"]
        assert all(i["type"] != "way" or "highway" in i["tags"] for i in data)
        return data

    def _way_length(self, way):
        xy = [[node["x"], node["y"]] for node in (self.nodes[i] for i in way["nodes"])]
        return sum(hypot(i[0] - j[0], i[1] - j[1]) for i, j in zip(xy, xy[1:]))

    def _update_node_ways(self):
        """
        Build node_ways(node_id -> list[way_id]) from ways
        """
        self.node_ways = defaultdict(list)
        for way_id, way in self.ways.items():
            for node_id in way["nodes"]:
                self.node_ways[node_id].append(way_id)
        self.node_wayset = {
            node_id: set(way_ids) for node_id, way_ids in self.node_ways.items()
        }

    def _assert_ways(self):
        for way_id, way in self.ways.items():
            assert way_id == way["id"]
            if len(way["nodes"]) != len(set(way["nodes"])):
                logging.warning(f"way {way_id} has duplicate nodes")

    def dump_as_geojson(self, path: str):
        geos = []
        used_nodes = set()  # start point and end point of way
        # way -> LineString
        for way in self.ways.values():
            used_nodes.add(way["nodes"][0])
            used_nodes.add(way["nodes"][-1])
            geos.append(
                Feature(
                    geometry=LineString(
                        [
                            (self.nodes[node_id]["lon"], self.nodes[node_id]["lat"])
                            for node_id in way["nodes"]
                        ]
                    ),
                    properties=way,
                )
            )
        self._update_node_ways()
        # junction -> MultiPoint
        for jid, nodes in self.junction2nodes.items():
            # no outputs if nodes are normal
            # normal means: for every node：
            # 1. It is not the starting point or end point of the way, and the corresponding number of ways is 0 or 1
            if all(
                node_id not in used_nodes
                and len(self.node_wayset.get(node_id, set())) <= 1
                for node_id in nodes
            ):
                continue
            geos.append(
                Feature(
                    geometry=MultiPoint(
                        [
                            (
                                self.nodes[node_id]["lon"],
                                self.nodes[node_id]["lat"],
                            )
                            for node_id in nodes
                        ]
                    ),
                    properties={
                        "id": jid,
                        "nodes": {
                            node_id: self.node_ways[node_id] for node_id in nodes
                        },
                    },
                )
            )
        geos = FeatureCollection(geos)
        with open(path, "w") as f:
            dump(geos, f, indent=2, ensure_ascii=False)

    def to_topo(self):
        """
        Returns the road topology in the form of a FeatureCollection, where:
        way: LineString representation, including attributes: id, lanes, highway, max_speed, name
        junction: MultiPoint representation, including attributes: id, in_ways, out_ways
        """
        geos = []
        jid_to_in_ways = defaultdict(list)
        jid_to_out_ways = defaultdict(list)
        links = {}  # (start_jid, end_jid) -> way_id, used to check whether there are multiple paths between junctions
        for way in self.ways.values():
            geos.append(
                Feature(
                    id=way["id"],
                    geometry=LineString(
                        [
                            (self.nodes[node_id]["lon"], self.nodes[node_id]["lat"])
                            for node_id in way["nodes"]
                        ]
                    ),
                    properties={
                        "id": way["id"],
                        "lanes": way["lanes"],
                        "highway": way["highway"],
                        "max_speed": way["max_speed"],
                        "name": way["name"],
                    },
                )
            )
            start_jid = self.node2junction[way["nodes"][0]]
            end_jid = self.node2junction[way["nodes"][-1]]
            jid_to_out_ways[start_jid].append(way["id"])
            jid_to_in_ways[end_jid].append(way["id"])
            link = (start_jid, end_jid)
            if link in links:
                logging.error(f"{start_jid} -> {end_jid}: {way['id']} vs {links[link]}")
            links[link] = way["id"]

        for jid, nodes in self.junction2nodes.items():
            in_ways = jid_to_in_ways[jid]
            out_ways = jid_to_out_ways[jid]
            assert len(in_ways) > 0 or len(out_ways) > 0
            geos.append(
                Feature(
                    id=jid,
                    geometry=MultiPoint(
                        [
                            (
                                self.nodes[node_id]["lon"],
                                self.nodes[node_id]["lat"],
                            )
                            for node_id in nodes
                        ]
                    ),
                    properties={
                        "id": jid,
                        "in_ways": in_ways,
                        "out_ways": out_ways,
                    },
                )
            )
            # check if there are normal junctions
            if len(in_ways) == len(out_ways) == 1:
                # If the junction at the other end of the two is the same, it means that it is a Around junction and is not considered ordinary.
                in_way = self.ways[in_ways[0]]
                out_way = self.ways[out_ways[0]]
                in_way_start_jid = self.node2junction[in_way["nodes"][0]]
                out_way_end_jid = self.node2junction[out_way["nodes"][-1]]
                if in_way_start_jid != out_way_end_jid:
                    logging.error(f"naive junction: {jid}")

        return FeatureCollection(geos)

    def _get_osm(self):
        osm_data = self._download_osm()
        # find every valuable nodes
        self.nodes = {}
        for doc in tqdm(osm_data):
            if doc["type"] != "node":
                continue
            if "lon" in doc and "lat" in doc:
                doc["x"], doc["y"] = self.projector(doc["lon"], doc["lat"])
                self.nodes[doc["id"]] = doc
        # all ways
        self.ways = {}
        for way in tqdm(osm_data):
            # Filter non-way elements
            if way["type"] != "way":
                continue
            # Filter road elements that are not within the target category
            highway = way["tags"]["highway"]
            if highway not in self.way_filter:
                continue
            info = parse_osm_way_tags(way["tags"], self.default_way_settings)
            way.update(info)
            del way["tags"]
            self.ways[way["id"]] = way
        self._assert_ways()
        logging.info(f"get {len(self.ways)} ways and {len(self.nodes)} nodes")
        self._update_node_ways()
        # Only keep nodes that have roads passing through them
        self.nodes = {
            node_id: self.nodes[node_id]
            for node_id, ways in self.node_ways.items()
            if len(ways) > 0
        }
        ## ================================================ ======= ##

        # A node through which multiple roads pass is called a joint. This part of the code splits the road at the joint.
        logging.info("split ways at joints")

        joints = {node_id for node_id, ways in self.node_ways.items() if len(ways) > 1}
        # Maximum road ID, +1 before use
        way_id = max(self.ways)
        # New road collection
        new_ways = {}
        for way in tqdm(self.ways.values()):
            last_pos = 0
            for pos, node in list(enumerate(way["nodes"]))[1:-1]:
                if node in joints:
                    new_way = copy.deepcopy(way)
                    new_way["nodes"] = way["nodes"][last_pos : pos + 1]
                    assert len(new_way["nodes"]) > 0
                    way_id += 1
                    new_way["id"] = way_id
                    new_way["original_id"] = way["id"]
                    new_ways[way_id] = new_way
                    last_pos = pos
            if last_pos != 0:
                way["nodes"] = way["nodes"][last_pos:]
            assert len(way["nodes"]) > 0
            new_ways[way["id"]] = way
        self.ways = new_ways
        self._update_node_ways()
        # Update joints
        joints = {node_id for node_id, ways in self.node_ways.items() if len(ways) > 1}
        # Only keep roads with joints at one end and delete roads that are not connected to joints at both ends (these roads are isolated)
        self.ways = {
            way_id: way
            for way_id, way in self.ways.items()
            if way["nodes"][0] in joints or way["nodes"][-1] in joints
        }
        # Ensure that all way IDs are correct
        self._assert_ways()
    # Intermediate processing operations

    def _remove_redundant_ways(self):
        """
        Some nodes are connected by multiple paths, remove these redundant paths;
        """
        # Some nodes are connected by multiple paths, remove these redundant paths.
        connects = defaultdict(set) # (start, end) -> way_id list
        for way_id, way in self.ways.items():
            start = way["nodes"][0]
            end = way["nodes"][-1]
            if way["oneway"]:
                connects[(start, end)].add(way_id)
            else:
                connects[(start, end)].add(way_id)
                connects[(end, start)].add(way_id)
        removed_ids = set()
        marked_ids = set()
        for (start, end), way_ids in connects.items():
            if len(way_ids) > 1:
                # If there is a two-way road, keep the two-way road, otherwise keep the shortest length
                way_id_list = list(way_ids)
                way_id_list.sort(
                    key=lambda wid: (
                        # Ascending order, two-way road first (not true -> false)
                        not self.ways[wid]["oneway"],
                        self._way_length(self.ways[wid]), #Ascending order, smaller length first
                    )
                )
                marked_ids.add(way_id_list[0])
                # Delete all paths except the first one
                for way_id in way_id_list[1:]:
                    removed_ids.add(way_id)
        for wid in marked_ids:
            self.ways[wid]["remove_redundant_ways"] = "same-start-end"
        for wid in removed_ids:
            del self.ways[wid]

        self._assert_ways()

    def _remove_simple_joints(self):
        """
        Some joints have only two roads connected, then merge the two roads into the same one (merge the short one into the long one)
        """
        self._update_node_ways()

        # Possible connection points
        # end way --> start way
        # One-way connection relationship
        di_g = nx.DiGraph()
        # Two-way connection relationship
        g = nx.Graph()
        for node_id, way_ids in self.node_ways.items():
            # Only nodes connected by two roads
            if len(way_ids) != 2:
                continue
            ai, bi = list(way_ids)
            if ai == bi:
                logging.warning(f"node {node_id} is isolated")
                continue # beijingbigger way 992244675 is a circle
            a, b = self.ways[ai], self.ways[bi]
            assert a["nodes"][0] == node_id or a["nodes"][-1] == node_id
            assert b["nodes"][0] == node_id or b["nodes"][-1] == node_id
            if a["oneway"] != b["oneway"]:
                # A one-way street and a two-way street are not processed
                continue
            if a["oneway"] and b["oneway"]:
                # They are all one-way streets. Let’s see what kind of connection is there.
                if a["nodes"][0] == node_id == b["nodes"][-1]:
                    # b ---> node_id ---> a
                    di_g.add_edge(bi, ai)
                elif a["nodes"][-1] == node_id == b["nodes"][0]:
                    # a ---> node_id ---> b
                    di_g.add_edge(ai, bi)
            else:
                # It’s all a two-way street
                g.add_edge(ai, bi)
        # One-way road processing: Find all source points in the graph, connect the ways from the source points to the sink points, and delete other ways except the ways corresponding to the source points.
        source_nodes = [node for node in di_g.nodes if di_g.in_degree(node) == 0]
        removed_ids = set()
        for n in source_nodes:
            way_ids = [n]
            while True:
                next_nodes = list(di_g.neighbors(n))
                if len(next_nodes) == 0:
                    break
                assert len(next_nodes) == 1, f"{n}->{next_nodes}"
                n = next_nodes[0]
                way_ids.append(n)
            # Assemble new ways and delete redundant ways
            main_way_id = way_ids[0]
            main_way = self.ways[main_way_id]
            main_way["remove_simple_joints"] = "main_di"
            for way_id in way_ids[1:]:
                way = self.ways[way_id]
                main_way["nodes"] += way["nodes"][1:]
                removed_ids.add(way_id)
        # Two-way processing, find all connected components, confirm that there are 2 points with degree 1, and the rest are points with degree 2
        # Select any point with degree 1, calculate the path to another point, and then assemble the new way
        # Find all Unicom components
        for component in list(nx.connected_components(g)):
            # Use assert to confirm that there are 2 nodes with degree 1, and the rest are nodes with degree 2
            degree_one_count = 0
            degree_two_count = 0
            for node in component:
                if g.degree(node) == 1: # type: ignore
                    degree_one_count += 1
                elif g.degree(node) == 2: # type: ignore
                    degree_two_count += 1
                else:
                    raise AssertionError("The degree of the node does not meet the requirements")
            assert degree_one_count == 2, "The number of nodes with degree 1 is incorrect"
            assert degree_two_count == len(component) - 2, "The number of nodes with degree 2 is incorrect"
            # Select a node with degree 1 and calculate the path to another node with degree 1
            degree_one_nodes = [
                node for node in component if g.degree(node) == 1 # type: ignore
            ]
            start_node = degree_one_nodes[0]
            end_node = degree_one_nodes[1]
            way_ids = nx.shortest_path(g, start_node, end_node)
            # Assemble new ways and delete redundant ways
            main_way_id = way_ids[0]
            main_way = self.ways[main_way_id]
            main_way["remove_simple_joints"] = "main"
            for way_id in way_ids[1:]:
                way = self.ways[way_id]
                try:
                    main_way["nodes"] = merge_way_nodes(main_way["nodes"], way["nodes"])
                except ValueError as e:
                    # with open("cache/graph.pkl", "wb") as f:
                    #     pickle.dump([g, component], f)
                    raise e
                removed_ids.add(way_id)
        for wid in removed_ids:
            del self.ways[wid]
        self._assert_ways()

    def _init_junctions(self):
        """
        Initialize junction so that every node connected to more than 1 way is a junction.
        """
        self._update_node_ways()

        def add_node_as_junction(node_id):
            if node_id in self.node2junction:
                return
            self.junction2nodes[self.junction_next_id] = [node_id]
            self.node2junction[node_id] = self.junction_next_id
            self.junction_next_id += 1

        for way in self.ways.values():
            other_nodes = way["nodes"][1:-1]
            # If the middle nodes correspond to multiple ways, an error will be reported.
            assert all(
                len(self.node_ways[node]) == 1 for node in other_nodes
            ), other_nodes
            # Add junction at the beginning and end
            add_node_as_junction(way["nodes"][0])
            add_node_as_junction(way["nodes"][-1])

    def _make_all_one_way(self):
        """
        In OSM, single-lane and double-lane roads are mixed and need to be unified into single-lane roads.
        Originally a two-way road, it became two one-way roads, with a U-turn added at the end of the road.
        """
        self._update_node_ways()
        # turnarounds = []
        next_way_id = max(self.ways) + 1
        next_node_id = max(self.nodes) + 1
        for way in list(self.ways.values()):
            if way["oneway"]:
                continue
            # Mark this road as being split from the original two-way road. Then the coordinates will be offset to separate the two roads.
            way["oneway_split"] = True
            new_way = copy.deepcopy(way)
            new_way["id"] = next_way_id
            next_way_id += 1
            new_way["original_id"] = way["id"]
            nodes = way["nodes"][::-1]
            # Create new ids for new new_way nodes
            new_nodes = [nodes[0]]
            for node_id in nodes[1:-1]:
                new_node = copy.deepcopy(self.nodes[node_id])
                new_node["id"] = next_node_id
                next_node_id += 1
                self.nodes[new_node["id"]] = new_node
                new_nodes.append(new_node["id"])
            new_nodes.append(nodes[-1])
            new_way["nodes"] = new_nodes
            self.ways[new_way["id"]] = new_way
    def _remove_out_of_roadnet(self):
        """
        Remove isolated roads outside the main road network, that is, only retain the largest connected component
        """
        G = nx.DiGraph()
        for way in self.ways.values():
            length = self._way_length(way)
            start_node = way["nodes"][0]
            end_node = way["nodes"][-1]
            G.add_edge(start_node, end_node, length=length)
        # Find the maximum connected component
        max_component = max(nx.weakly_connected_components(G), key=len)
        # Delete roads that are not within the maximum connected component
        for way_id, way in list(self.ways.items()):
            start_node = way["nodes"][0]
            end_node = way["nodes"][-1]
            if start_node not in max_component or end_node not in max_component:
                del self.ways[way_id]

    def _merge_junction_by_motif(self):
        """
        Identify fixed patterns in the topology and combine them into a junction
        """
        # After bidirectional road processing, it is now a directed graph.
        G = nx.DiGraph()
        for way in self.ways.values():
            length = self._way_length(way)
            start_node = way["nodes"][0]
            end_node = way["nodes"][-1]
            start_junc = self.node2junction[start_node]
            end_junc = self.node2junction[end_node]
            G.add_edge(start_junc, end_junc, length=length)
        all_motifs: Set[FrozenSet[int]] = set()
        all_motifs.update(suc_is_close_by_other_way(G))
        all_motifs.update(close_nodes(G))
        all_motifs.update(motif_H(G))
        biG = G.to_undirected(reciprocal=False)
        mergeG = nx.Graph()
        for motif in all_motifs:
            for junc in motif:
                mergeG.add_node(junc)
            for junc1, junc2 in combinations(motif, 2):
                if not biG.has_edge(junc1, junc2):
                    continue
                if biG.edges[(junc1, junc2)]["length"] > 100:
                    continue
                mergeG.add_edge(junc1, junc2)
        for c in nx.connected_components(mergeG):
            c = list(c)
            main_junc = c[0]
            for junc in c[1:]:
                # Merge junc into main_junc
                self.junction2nodes[main_junc] += self.junction2nodes[junc]
                for node in self.junction2nodes[junc]:
                    self.node2junction[node] = main_junc
                del self.junction2nodes[junc]

        # Remove all meaningless ways (that is, both ends of the way are the same junction)
        for way_id, way in list(self.ways.items()):
            start_junc = self.node2junction[way["nodes"][0]]
            end_junc = self.node2junction[way["nodes"][-1]]
            if start_junc == end_junc:
                del self.ways[way_id]

    def _clean_topo(self):
        """
        If there are multiple roads connecting two intersections, only the one with the largest number of lanes and the shortest length is retained.
        If there are only two roads at the intersection, one in and one out, then delete the intersection and join the two roads.
        """
        # Remove duplicate roads
        links = defaultdict(list) # (start_junc, end_junc) -> way_ids
        for way in self.ways.values():
            start_junc = self.node2junction[way["nodes"][0]]
            end_junc = self.node2junction[way["nodes"][-1]]
            links[(start_junc, end_junc)].append(way["id"])
        # Remove redundant paths
        for way_ids in links.values():
            # Keep the only connection between two junctions according to the rule of having the most lanes and the shortest length.
            ways = [self.ways[i] for i in way_ids]
            ways.sort(key=lambda x: (-x["lanes"], self._way_length(x)))
            if len(ways) > 1:
                ways[0]["clean_topo"] = "best-one"
                for way in ways[1:]:
                    del self.ways[way["id"]]

        # Remove trivial junctions
        G = nx.DiGraph()
        for way_id, way in self.ways.items():
            length = self._way_length(way)
            start_node = way["nodes"][0]
            end_node = way["nodes"][-1]
            start_junc = self.node2junction[start_node]
            end_junc = self.node2junction[end_node]
            G.add_edge(start_junc, end_junc, length=length, id=way_id)
        # Find nodes whose out-degree and in-degree are both 1
        candidates = set()
        for node in G.nodes:
            if G.in_degree(node) == G.out_degree(node) == 1:
                candidates.add(node)
        for c in candidates:
            pre = list(G.predecessors(c))[0]
            # If the incoming edge and outgoing edge are the successors of each other, it means that this is a U-turn node with separate two-way streets and will not be removed.
            suc = list(G.successors(c))[0]
            if c in set(G.successors(suc)):
                continue
            # Otherwise merge
            in_edge_way_id = G[pre][c]["id"]
            out_edge_way_id = G[c][suc]["id"]
            in_way = self.ways[in_edge_way_id]
            out_way = self.ways[out_edge_way_id]
            # merge way
            in_way["nodes"] += out_way["nodes"][1:]
            in_way["lanes"] = max(in_way["lanes"], out_way["lanes"])
            # Adjust G topology
            G.add_edge(
                pre,
                suc,
                length=G[pre][c]["length"] + G[c][suc]["length"],
                id=in_edge_way_id,
            )
            G.remove_edge(pre, c)
            G.remove_edge(c, suc)
            # Delete out_way
            del self.ways[out_edge_way_id]
            # Delete node
            for node in self.junction2nodes[c]:
                del self.node2junction[node]
            del self.junction2nodes[c]
    def create_road_net(self, output_path: Optional[str] = None):
        """
        Create Road net from OpenStreetMap.

        Args:
        - output_path (str): GeoJSON file output path.

        Returns:
        - roads and junctions in GeoJSON format.
        """
        # Get OSM data
        self._get_osm()
        # self.dump_as_geojson("cache/1.geojson")
        # Process

        # 1. Remove redundant ways/nodes
        self._remove_redundant_ways()
        self._remove_simple_joints()
        # self.dump_as_geojson("cache/2.geojson")
        # 2. Convert all ways to one-way streets
        self._make_all_one_way()
        # self.dump_as_geojson("cache/3.geojson")
        # 3. Remove isolated roads outside the main road network
        self._remove_out_of_roadnet()
        # self.dump_as_geojson("cache/3_1.geojson")
        self._init_junctions()
        # self.dump_as_geojson("cache/4.geojson")
        # 3. Build junction
        self._merge_junction_by_motif()
        # self.dump_as_geojson("cache/5.geojson")
        # 4. Delete redundant content
        self._clean_topo()
        # self.dump_as_geojson("cache/6.geojson")
        # 5. Save topology
        topo = self.to_topo()
        if output_path is not None:
            with open(output_path, encoding="utf-8", mode="w") as f:
                dump(topo, f, indent=2, ensure_ascii=False)
        return topo
