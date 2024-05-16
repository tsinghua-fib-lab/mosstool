import logging
from typing import Dict, Optional

import pyproj
import requests
import shapely.ops as ops
from geojson import Feature, FeatureCollection, Polygon, dump
from shapely.geometry import MultiPolygon as sMultiPolygon
from shapely.geometry import Polygon as sPolygon

__all__ = ["Building"]


class Building:
    """
    Process OSM raw data to AOI as geojson format files
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
        self.bbox = (
            min_latitude,
            min_longitude,
            max_latitude,
            max_longitude,
        )
        self.projector = pyproj.Proj(proj_str)
        self.wikipedia_name = wikipedia_name
        self.proxies = proxies
        # OSM raw data
        self._ways = []
        self._nodes = []
        self._rels = []
        self._ways_aoi = []
        self._ways_rel = []

        # generate AOIs
        self.aois: list = []

    def _query_raw_data(self):
        """
        Get raw data from OSM API
        OSM query language: https://wiki.openstreetmap.org/wiki/Overpass_API/Language_Guide
        Can be run and visualized in real time at https://overpass-turbo.eu/
        """
        logging.info("Querying osm raw data")
        bbox_str = ",".join(str(i) for i in self.bbox)
        query_header = f"[out:json][timeout:120][bbox:{bbox_str}];"
        area_wikipedia_name = self.wikipedia_name
        if area_wikipedia_name is not None:
            query_header += f'area[wikipedia="{area_wikipedia_name}"]->.searchArea;'
        osm_data = None
        for _ in range(3):  # retry 3 times
            try:
                query_body_raw = [
                    (
                        "way",
                        '[!highway][!tunnel][!boundary][!railway][!natural][!barrier][!junction][!waterway][!public_transport][landuse!~"grass|forest"][place!~"suburb|neighbourhood"][amenity!=fountain][historic!=city_gate][artwork_type!=sculpture][man_made!~"bridge|water_well"][building!~"wall|train_station|roof"]',
                    ),
                    (
                        "rel",
                        '[landuse][landuse!~"grass|forest"][type=multipolygon]',
                    ),
                    (
                        "rel",
                        "[amenity][amenity!=fountain][type=multipolygon]",
                    ),
                    (
                        "rel",
                        '[building][building!~"wall|train_station|roof"][type=multipolygon]',
                    ),
                    (
                        "rel",
                        "[leisure][leisure!=nature_reserve][type=multipolygon]",
                    ),
                ]
                query_body = ""
                for obj, args in query_body_raw:
                    area = (
                        "(area.searchArea)" if area_wikipedia_name is not None else ""
                    )
                    query_body += obj + area + args + ";"
                query_body = "(" + query_body + ");"
                query = query_header + query_body + "(._;>;);" + "out body;"
                logging.info(f"{query}")
                osm_data = requests.get(
                    "http://overpass-api.de/api/interpreter?data=" + query,
                    proxies=self.proxies,
                ).json()["elements"]
                break
            except Exception as e:
                logging.warning(f"Exception when querying OSM data {e}")
                logging.warning("No response from OSM, Please try again later!")
        if osm_data is None:
            raise Exception("No AOI response from OSM!")
        nodes = [d for d in osm_data if d["type"] == "node"]
        ways = [d for d in osm_data if d["type"] == "way"]
        rels = [
            d for d in osm_data if d["type"] == "relation"
        ]  # relation: set of ways to be a poly, way as outer or inner
        logging.info(f"nodes, ways, relations: {len(nodes)}, {len(ways)}, {len(rels)}")

        # ways_rel_id = {mem['ref'] if mem['type'] == 'way' else -1 for rel in rels for mem in rel['members']}
        # assert -1 not in ways_rel_id  # relation is made of way
        for rel in rels:
            rel["members"] = [mem for mem in rel["members"] if mem["type"] == "way"]
        ways_rel_id = {mem["ref"] for rel in rels for mem in rel["members"]}
        # Participate in the way that constitutes aoi as part of the relationship
        ways_rel = [x for x in ways if x["id"] in ways_rel_id]
        logging.info(f"ways for relation:{len(ways_rel)}")

        ways_aoi = [
            w for w in ways if "tags" in w and w["id"] not in ways_rel_id
        ]  # The way that alone constitutes aoi
        logging.info(f"ways directly for aoi:{len(ways_aoi)}")
        self._nodes = nodes
        self._rels = rels
        self._ways_aoi = ways_aoi
        self._ways_rel = ways_rel

    def _make_raw_aoi(self):
        """
        Construct aoi from original data: mainly generate polygon as the geometry of aoi
        """
        nodes = self._nodes
        rels = self._rels
        ways_aoi = self._ways_aoi
        ways_rel = self._ways_rel
        logging.info("Making raw aoi")
        nodes_dict = {
            n["id"]: [a for a in self.projector(n["lon"], n["lat"], inverse=False)]
            for n in nodes
        }
        ways_aoi_dict = {w["id"]: w for w in ways_aoi}  # nodes, tags
        ways_rel_dict = {w["id"]: w for w in ways_rel}
        rels_dict = {r["id"]: r for r in rels}  # members: [(ref, role)], tags
        aoi_id = 0

        # Process ways_aoi into aoi: each way is a ring shape, no inner shape
        invalid_way_cnt = 0
        for wid, way in ways_aoi_dict.items():
            nodes = way["nodes"]
            if nodes[0] != nodes[-1]:  # should be a ring
                invalid_way_cnt += 1
                continue
            geo = sPolygon([nodes_dict[node] for node in nodes])
            if geo.is_valid:
                self.aois.append(
                    {
                        "id": aoi_id,
                        "geo": geo,
                        "tags": [way["tags"]],
                    }
                )
                aoi_id += 1
            else:
                invalid_way_cnt += 1
                logging.warning("invalid geometry!")
        logging.info(f"invalid_way_cnt: {invalid_way_cnt}")

        # Process rels to become aoi: each rel describes several ways that together form a ring shape, and may have inner shapes. The inner shapes have not been processed yet (geo is all solid poly)
        invalid_relation_cnt = 0
        for rid, rel in rels_dict.items():
            t = {"outer": [], "inner": []}
            for x in rel["members"]:
                wid, role = x["ref"], x["role"]
                if role not in {"inner", "outer"}:
                    invalid_relation_cnt += 1
                    break
                t[role].append(wid)
            if not t["outer"]:
                continue

            if len(t["outer"]) > 1:  # The shape has more than 1 sides
                nodes_ring = []  # Ring edges
                nodes_line = []  # No loop edges
                for wid in t["outer"]:
                    nodes = ways_rel_dict[wid]["nodes"]
                    if nodes[0] == nodes[-1]:
                        nodes_ring.append(nodes)
                    else:
                        nodes_line.append(nodes)

                if (
                    nodes_ring and not nodes_line
                ):  # Multiple ring-forming edges: It may be combined into a polygon by union; it may also be separated into multiple shapes, in which case it will be split into multiple Aoi
                    polys = [
                        sPolygon([nodes_dict[node] for node in nodes])
                        for nodes in nodes_ring
                    ]
                    geo = ops.unary_union(polys)
                    if isinstance(
                        geo, sMultiPolygon
                    ):  # Union still has multiple polygons
                        polys = list(geo.geoms)
                        for poly in polys:
                            if poly.is_valid:
                                self.aois.append(
                                    {
                                        "id": aoi_id,
                                        "geo": poly,
                                        "tags": [rels_dict[rid].get("tags", {})],
                                    }
                                )
                                aoi_id += 1
                            else:
                                invalid_relation_cnt += 1
                                logging.warning("invalid geometry!")
                    else:
                        assert isinstance(geo, sPolygon)  # union as one polygon
                        if geo.is_valid:
                            self.aois.append(
                                {
                                    "id": aoi_id,
                                    "geo": geo,
                                    "tags": [
                                        rels_dict[rid].get("tags", {}),
                                        *(
                                            ways_rel_dict[wid].get("tags", {})
                                            for wid in t["outer"]
                                        ),
                                    ],
                                }
                            )
                            aoi_id += 1
                        else:
                            invalid_relation_cnt += 1
                            logging.warning("invalid geometry!")
                elif (
                    nodes_line and not nodes_ring
                ):  # Connect multiple non-cyclic edges to form a polygon
                    geo = list(
                        ops.polygonize(
                            [
                                [nodes_dict[node] for node in nodes]
                                for nodes in nodes_line
                            ]
                        )
                    )
                    if (
                        len(geo) == 1
                    ):  # should be connected to form a unique closed polygon
                        geo = geo[0]
                        if geo.is_valid:
                            self.aois.append(
                                {
                                    "id": aoi_id,
                                    "geo": geo,
                                    "tags": [
                                        rels_dict[rid].get("tags", {}),
                                        *(
                                            ways_rel_dict[wid].get("tags", {})
                                            for wid in t["outer"]
                                        ),
                                    ],
                                }
                            )
                            aoi_id += 1
                        else:
                            invalid_relation_cnt += 1
                            logging.warning("invalid geometry!")
                    else:
                        logging.warning(
                            f"not processed: line outer ways make more than one polygon:{len(geo)}"
                        )
                else:
                    logging.warning(
                        f"not processed: exist both ring and line outer ways:{len(nodes_ring)},{len(nodes_line)}"
                    )
            else:  # len(t['outer']) == 1: There is only 1 outer edge in the shape
                wid = t["outer"][0]
                nodes = ways_rel_dict[wid]["nodes"]
                if nodes[0] == nodes[-1]:  # should be a ring
                    geo = sPolygon([nodes_dict[node] for node in nodes])
                    if geo.is_valid:
                        self.aois.append(
                            {
                                "id": aoi_id,
                                "geo": geo,
                                "tags": [
                                    rels_dict[rid].get("tags", {}),
                                    ways_rel_dict[wid].get("tags", {}),
                                ],
                            }
                        )
                        aoi_id += 1
                    else:
                        invalid_relation_cnt += 1
                        print("invalid geometry!")
                else:
                    invalid_relation_cnt += 1

        logging.info(f"invalid_relation_cnt: {invalid_relation_cnt}")
        logging.info(f"raw aoi: {len(self.aois)}")

    def create_building(self, output_path: Optional[str] = None):
        """
        Create AOIs from OpenStreetMap.

        Args:
        - output_path (str): GeoJSON file output path.

        Returns:
        - AOIs in GeoJSON format.
        """
        self._query_raw_data()
        self._make_raw_aoi()
        geos = []
        for aoi in self.aois:
            geos.append(
                Feature(
                    geometry=Polygon(
                        [
                            [
                                list(self.projector(c[0], c[1], inverse=True))
                                for c in aoi["geo"].exterior.coords
                            ]
                        ]
                    ),
                    properties={
                        "id": aoi["id"],
                        "osm_tags": aoi["tags"],
                        "type": "",
                    },
                )
            )
        geos = FeatureCollection(geos)
        if output_path is not None:
            with open(output_path, encoding="utf-8", mode="w") as f:
                dump(geos, f, indent=2, ensure_ascii=False)
        return geos
