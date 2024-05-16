from copy import deepcopy
from typing import Dict, Tuple, Union

import numpy as np
import geojson
import pydeck as pdk
import pyproj
from ...type import Map, Position, LaneType
from shapely.geometry import LineString, Point, Polygon

from ...util.color import hex_to_rgba
from ...util.format_converter import pb2dict

__all__ = ["VisMap"]


class VisMap:
    """
    Preprocessing map data for visualization.
    """

    def __init__(self, m: Map):
        """
        Args:
        - m: Map
        """
        self.m = m
        self.projector = pyproj.Proj(m.header.projection)
        self.center = self._get_center()
        self.lane_features, self.lane_shapely_xys = self._build_lanes()
        self.road_features, self.road_shapely_xys = self._build_roads()
        self.aoi_features, self.aoi_shapely_xys = self._build_aois()
        self.poi_features, self.poi_shapely_xys = self._build_pois()

    @property
    def feature_collection(self):
        """
        Return:
        - geojson.FeatureCollection: all features in the map
        """
        features = (
            list(self.lane_features.values())
            + list(self.road_features.values())
            + list(self.aoi_features.values())
            + list(self.poi_features.values())
        )
        return geojson.FeatureCollection(features)

    def _get_center(self):
        header = self.m.header
        min_x, max_x = header.west, header.east
        min_y, max_y = header.south, header.north
        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        lng, lat = self.projector(center_x, center_y, inverse=True)
        return lng, lat

    def _build_lanes(self):
        features: Dict[int, dict] = {}
        shapely_xys: Dict[int, LineString] = {}
        for lane in self.m.lanes:
            id = lane.id
            # xys = [[node.x, node.y] for node in lane.center_line.nodes]
            # shapely_xys[id] = LineString(xys)
            # xs, ys = zip(*xys)
            try:
                xyzs = [[node.x, node.y, node.z] for node in lane.center_line.nodes]
            except AttributeError:
                xyzs = [[node.x, node.y, 0] for node in lane.center_line.nodes]
            shapely_xys[id] = LineString(xyzs)
            xs, ys, zs = zip(*xyzs)
            lnglats = list(zip(*self.projector(xs, ys, inverse=True)))
            lnglatzs = list(zip(*np.array(lnglats).T, zs))
            d = pb2dict(lane)
            try:
                del d["center_line"]
                del d["left_border_line"]
                del d["right_border_line"]
            except KeyError:
                pass
            feature = {
                "type": "Feature",
                "id": id,
                # "geometry": {"type": "LineString", "coordinates": lnglats},
                "geometry": {"type": "LineString", "coordinates": lnglatzs},
                "properties": d,
            }
            features[id] = feature
        return features, shapely_xys

    def _build_roads(self):
        features: Dict[int, dict] = {}
        shapely_xys: Dict[int, LineString] = {}
        for road in self.m.roads:
            id = road.id
            driving_lane_ids = []
            walking_lane_ids = []
            for lane_id in road.lane_ids:
                if lane_id not in self.lane_features:
                    raise ValueError(f"bad map: lane {lane_id} in road {id} not found")
                lane = self.lane_features[lane_id]
                if lane["properties"]["type"] == LaneType.LANE_TYPE_DRIVING:
                    driving_lane_ids.append(lane_id)
                elif lane["properties"]["type"] == LaneType.LANE_TYPE_WALKING:
                    walking_lane_ids.append(lane_id)
            main_lane_ids = driving_lane_ids if len(driving_lane_ids) > 0 else walking_lane_ids
            if len(main_lane_ids) == 0:
                raise ValueError(f"bad map: road {id} has no driving or walking lanes")
            center_lane_id = main_lane_ids[len(main_lane_ids) // 2]
            shapely_xys[id] = self.lane_shapely_xys[center_lane_id]
            feature = deepcopy(self.lane_features[center_lane_id])
            feature["id"] = id
            feature["properties"] = pb2dict(road)
            features[id] = feature
        return features, shapely_xys

    def _build_aois(self):
        features: Dict[int, dict] = {}
        shapely_xys: Dict[int, Union[Point, Polygon]] = {}
        for aoi in self.m.aois:
            id = aoi.id
            xys = [[node.x, node.y] for node in aoi.positions]
            xs, ys = zip(*xys)
            lnglats = list(zip(*self.projector(xs, ys, inverse=True)))
            d = pb2dict(aoi)
            try:
                del d["positions"]
            except KeyError:
                pass
            if len(lnglats) == 1:
                shapely_xys[id] = Point(xys[0])
                feature = {
                    "type": "Feature",
                    "id": id,
                    "geometry": {"type": "Point", "coordinates": lnglats[0]},
                    "properties": d,
                }
            else:
                shapely_xys[id] = Polygon(xys)
                feature = {
                    "type": "Feature",
                    "id": id,
                    "geometry": {"type": "Polygon", "coordinates": [lnglats]},
                    "properties": d,
                }
            features[id] = feature
        return features, shapely_xys

    def _build_pois(self):
        features: Dict[int, dict] = {}
        shapely_xys: Dict[int, Point] = {}
        for poi in self.m.pois:
            id = poi.id
            x, y = poi.position.x, poi.position.y
            shapely_xys[id] = Point(x, y)
            lnglat = list(self.projector(x, y, inverse=True))
            d = pb2dict(poi)
            try:
                del d["position"]
            except KeyError:
                pass
            feature = {
                "type": "Feature",
                "id": id,
                "geometry": {"type": "Point", "coordinates": lnglat},
                "properties": d,
            }
            features[id] = feature
        return features, shapely_xys

    def position2lnglat(self, position: Position) -> Tuple[float, float]:
        """
        Convert position to (lng, lat).

        Args:
        - position: Position

        Returns:
        - Tuple[float, float]: (lng, lat)
        """

        if position.HasField("aoi_position"):
            geo = self.aoi_shapely_xys[position.aoi_position.aoi_id]
        elif position.HasField("lane_position"):
            lane = self.lane_shapely_xys[position.lane_position.lane_id]
            geo = lane.interpolate(position.lane_position.s)
        else:
            raise ValueError(f"Unknown position type: {position}")
        x, y = geo.centroid.coords[0]
        return self.projector(x, y, inverse=True)

    def id2lnglats(self, lane_id_or_road_id: int) -> np.ndarray:
        """
        Convert lane_id or road_id to lnglats np.ndarray.

        Args:
        - lane_id_or_road_id: int, lane_id or road_id

        Returns:
        - np.ndarray: lnglats np.ndarray[[lng0, lat0], [lng1, lat1], ...]
        """

        line = self.lane_shapely_xys.get(
            lane_id_or_road_id, self.road_shapely_xys.get(lane_id_or_road_id)
        )
        if line is None:
            raise ValueError(f"lane_id or road_id {lane_id_or_road_id} not found")
        xys = np.array(line.coords, dtype=np.float64)
        return np.array(list(self.projector(*xys.T, inverse=True)), dtype=np.float64).T

    def visualize(self):
        """
        Visualize the map using pydeck.
        TODO: custom color for different types of lanes, AOIs and POIs.

        Args:
        - m: The map to be visualized.

        Returns:
        - The pydeck Deck.
        """
        features = []
        for lane in self.lane_features.values():
            color = "#29A2FF" if lane["properties"]["type"] == 1 else "#FFBE1A"
            color_rgba = hex_to_rgba(color, 0.5 * 255)
            feature = deepcopy(lane)
            feature["properties"] = {
                "color": color_rgba,
                "tooltip": f"type: {lane['properties']['type']}, turn: {lane['properties']['turn']}, parent_id: {lane['properties']['parent_id']}",
            }
            features.append(feature)
        for aoi in self.aoi_features.values():
            color = "#FFD4CE"
            color_rgba = hex_to_rgba(color, 0.5 * 255)
            feature = deepcopy(aoi)
            feature["properties"] = {
                "color": color_rgba,
                "tooltip": f"type: {aoi['properties'].get('urban_land_use', 'unknown')}",
            }
            features.append(feature)
        for poi in self.poi_features.values():
            color = "#F7624D"
            color_rgba = hex_to_rgba(color, 0.5 * 255)
            feature = deepcopy(poi)
            feature["properties"] = {
                "color": color_rgba,
                "tooltip": f"name: {poi['properties']['name']}, category: {poi['properties']['category']}",
            }
            features.append(feature)
        layer = pdk.Layer(
            "GeoJsonLayer",
            features,
            stroked=True,
            filled=True,
            pickable=True,
            line_cap_rounded=True,
            get_fill_color="properties.color",
            get_line_color="properties.color",
            get_line_width=1,
            line_width_min_pixels=1,
            tooltip=True,
        )
        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(
                latitude=self.center[1],
                longitude=self.center[0],
                zoom=10,
                pitch=0,
            ),
            tooltip={
                "html": "ID: {id}<br>{properties.tooltip}",
                "style": {"color": "white", "word-wrap": "break-word"},
            },  # type: ignore
        )
        return deck
