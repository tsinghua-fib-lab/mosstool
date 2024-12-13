import logging
from typing import Optional

import requests
from geojson import Feature, FeatureCollection, Point, dump

from .._map_util.format_checker import osm_format_checker

__all__ = ["PointOfInterest"]


class PointOfInterest:
    """
    Process OSM raw data to POI as geojson format files
    """

    def __init__(
        self,
        max_longitude: Optional[float] = None,
        min_longitude: Optional[float] = None,
        max_latitude: Optional[float] = None,
        min_latitude: Optional[float] = None,
        wikipedia_name: Optional[str] = None,
        proxies: Optional[dict[str, str]] = None,
    ):
        """
        Args:
        - max_longitude (Optional[float]): max longitude
        - min_longitude (Optional[float]): min longitude
        - max_latitude (Optional[float]): max latitude
        - min_latitude (Optional[float]): min latitude
        - wikipedia_name (Optional[str]): wikipedia name of the area in OSM.
        - proxies (Optional[dict[str, str]]): proxies for requests, e.g. {'http': 'http://localhost:1080', 'https': 'http://localhost:1080'}
        """
        self.bbox = (
            min_latitude,
            min_longitude,
            max_latitude,
            max_longitude,
        )
        self.wikipedia_name = wikipedia_name
        self.proxies = proxies
        # OSM raw data
        self._nodes = []

        # generate POIs
        self.pois: list = []

    def _query_raw_data(self, osm_data_cache: Optional[list[dict]] = None):
        """
        Get raw data from OSM API
        OSM query language: https://wiki.openstreetmap.org/wiki/Overpass_API/Language_Guide
        Can be run and visualized in real time at https://overpass-turbo.eu/
        """
        if osm_data_cache is None:
            logging.info("Querying osm raw data")
            assert all(
                i is not None for i in self.bbox
            ), f"longitude and latitude are required without cache file!"
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
                            "node",
                            "",
                        ),
                    ]
                    query_body = ""
                    for obj, args in query_body_raw:
                        area = (
                            "(area.searchArea)"
                            if area_wikipedia_name is not None
                            else ""
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
                raise Exception("No POI response from OSM!")
        else:
            osm_data = osm_data_cache
        nodes = [d for d in osm_data if d["type"] == "node"]
        logging.info(f"node: {len(nodes)}")
        self._nodes = nodes

    def _make_raw_poi(self):
        """
        Construct POI from original OSM data.
        """
        _raw_pois = []
        for d in self._nodes:
            d_tags = d.get("tags", {})
            p_name = d_tags.get("name", "")
            # name
            d["name"] = p_name
            # catg
            if "landuse" in d_tags:
                value = d_tags["landuse"]
                # Exclude invalid fields
                if not "yes" in value:
                    p_catg = value
                    d["catg"] = p_catg
                    _raw_pois.append(d)
                    continue
            if "leisure" in d_tags:
                value = d_tags["leisure"]
                if not "yes" in value:
                    p_catg = "leisure|" + value
                    d["catg"] = p_catg
                    _raw_pois.append(d)
                    continue
            if "amenity" in d_tags:
                value = d_tags["amenity"]
                if not "yes" in value:
                    p_catg = "amenity|" + value
                    d["catg"] = p_catg
                    _raw_pois.append(d)
                    continue
            if "building" in d_tags:
                value = d_tags["building"]
                if not "yes" in value:
                    p_catg = "building|" + value
                    d["catg"] = p_catg
                    _raw_pois.append(d)
                    continue
        logging.info(f"raw poi: {len(_raw_pois)}")
        self.pois = _raw_pois

    def create_pois(
        self,
        output_path: Optional[str] = None,
        osm_data_cache: Optional[list[dict]] = None,
        osm_cache_check: bool = False,
    ):
        """
        Create POIs from OpenStreetMap.

        Args:
        - osm_data_cache (Optional[list[dict]]): OSM data cache.
        - output_path (str): GeoJSON file output path.
        - osm_cache_check (bool): check the format of input OSM data cache.

        Returns:
        - POIs in GeoJSON format.
        """
        osm_format_checker(osm_cache_check, osm_data_cache, {"node": ["lon", "lat"]})
        self._query_raw_data(osm_data_cache)
        self._make_raw_poi()
        geos = []
        for poi_id, poi in enumerate(self.pois):
            geos.append(
                Feature(
                    geometry=Point([poi["lon"], poi["lat"]]),
                    properties={
                        "id": poi_id,
                        "osm_tags": poi["tags"],
                        "name": poi["name"],
                        "catg": poi["catg"],
                    },
                )
            )
        geos = FeatureCollection(geos)
        if output_path is not None:
            with open(output_path, encoding="utf-8", mode="w") as f:
                dump(geos, f, indent=2, ensure_ascii=False)
        return geos
