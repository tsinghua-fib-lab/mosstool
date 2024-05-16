from collections import defaultdict
from typing import List, cast

import numpy as np
import pydeck as pdk
from pycityproto.city.geo.v2.geo_pb2 import Position
from pycityproto.city.trip.v2.trip_pb2 import Schedule
from pycityproto.city.routing.v2.routing_pb2 import Journey, MovingDirection
from pycityproto.city.person.v1.person_pb2 import Person

from ...map.vis.map import VisMap

__all__ = ["VisTrip"]


class VisTrip:
    """
    Preprocessing person (trip) data for visualization.
    """

    def __init__(self, vis_map: VisMap, persons: List[Person]):
        self.m = vis_map
        self.persons = persons

    def visualize_home(self):
        """
        Visualize people's homes using pydeck.
        TODO: colormap, test for AOI

        Returns:
        - The pydeck Deck.
        """
        # For those whose home is in the AOI, perform clustering and draw the GeoJSON of the AOI. The color indicates the quantity and the tooltip indicates the ID list.
        # For home on Lane, draw scatter points, tooltip indicates ID
        aoi_persons = defaultdict(list)
        features = []
        for p in self.persons:
            home = cast(Position, p.home)
            if home.HasField("aoi_position"):
                aoi_persons[home.aoi_position.aoi_id].append(p.id)
            elif home.HasField("lane_position"):
                lng, lat = self.m.position2lnglat(home)
                feature = {
                    "type": "Feature",
                    "id": p.id,
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lng, lat],
                    },
                    "properties": {
                        "id": p.id,
                        "tooltip": f"position: (lane={home.lane_position.lane_id},s={home.lane_position.s:.2f})",
                    },
                }
                features.append(feature)
            else:
                raise ValueError(f"Unknown home position type for person(id={p.id})")
        # Fill in the GeoJSON of the AOI
        for aoi_id, pids in aoi_persons.items():
            aoi = self.m.aoi_features[aoi_id]
            aoi["properties"] = {
                # "color": [255, 0, 0, 255],
                "tooltip": f"persons: {pids}",
            }
            features.append(aoi)

        layer = pdk.Layer(
            "GeoJsonLayer",
            features,
            stroked=True,
            filled=True,
            pickable=True,
            line_cap_rounded=True,
            get_fill_color=[255, 0, 0, 255],
            get_line_color=[255, 0, 0, 255],
            get_line_width=1,
            line_width_min_pixels=1,
            tooltip=True,
        )
        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(
                latitude=self.m.center[1],
                longitude=self.m.center[0],
                zoom=10,
                pitch=0,
            ),
            tooltip={
                "html": "ID: {id}<br>{properties.tooltip}",
                "style": {"color": "white", "word-wrap": "break-word"},
            },  # type: ignore
        )
        return deck

    def visualize_od(self):
        """
        Visualize people's all start-end links using pydeck ArcLayer.
        TODO: colormap

        Returns:
        - The pydeck Deck.
        """
        arcs = []  # {"source": [lng, lat], "target": [lng, lat]}
        for p in self.persons:
            positions = [p.home]
            for schedule in p.schedules:
                schedule = cast(Schedule, schedule)
                for _ in range(schedule.loop_count):
                    for trip in schedule.trips:
                        positions.append(trip.end)
            arc = []
            for i in range(len(positions) - 1):
                lng1, lat1 = self.m.position2lnglat(positions[i])
                lng2, lat2 = self.m.position2lnglat(positions[i + 1])
                arc.append({"source": [lng1, lat1], "target": [lng2, lat2]})
            arcs += arc
        layer = pdk.Layer(
            "ArcLayer",
            arcs,
            get_source_position="source",
            get_target_position="target",
            get_width=5,
            pickable=True,
            get_source_color=[255, 0, 0, 127],
            get_target_color=[0, 255, 0, 127],
            auto_highlight=True,
        )
        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(
                latitude=self.m.center[1],
                longitude=self.m.center[0],
                zoom=10,
                pitch=0,
            ),
        )
        return deck

    def visualize_route(self):
        """
        Visualize people's all routes using pydeck GeoJsonLayer.
        TODO: colormap
        TODO: junction lane
        
        Returns:
        - The pydeck Deck.
        """
        features = []
        for p in self.persons:
            for si, schedule in enumerate(p.schedules):
                schedule = cast(Schedule, schedule)
                for ti, trip in enumerate(schedule.trips):
                    for ji, journey in enumerate(trip.routes):
                        journey = cast(Journey, journey)
                        # only driving and walking are supported
                        if journey.HasField("driving"):
                            driving = journey.driving
                            lnglats = np.concatenate(
                                [
                                    self.m.id2lnglats(road_id)
                                    for road_id in driving.road_ids
                                ]
                            )
                            feature = {
                                "type": "Feature",
                                "geometry": {
                                    "type": "LineString",
                                    "coordinates": lnglats.tolist(),
                                },
                                "properties": {
                                    "id": p.id,
                                    "tooltip": f"schedule: {si}, trip: {ti}, journey: drive-{ji}",
                                },
                            }
                            features.append(feature)
                        elif journey.HasField("walking"):
                            walking = journey.walking
                            lnglats = []
                            for seg in walking.route:
                                one = self.m.id2lnglats(seg.lane_id)
                                if seg.moving_direction == MovingDirection.MOVING_DIRECTION_BACKWARD:
                                    one = one[::-1]
                                lnglats.append(one)
                            lnglats = np.concatenate(lnglats)
                            feature = {
                                "type": "Feature",
                                "geometry": {
                                    "type": "LineString",
                                    "coordinates": lnglats.tolist(),
                                },
                                "properties": {
                                    "id": p.id,
                                    "tooltip": f"schedule: {si}, trip: {ti}, journey: walk-{ji}",
                                },
                            }
                            features.append(feature)
        layer = pdk.Layer(
            "GeoJsonLayer",
            features,
            stroked=True,
            filled=True,
            pickable=True,
            line_cap_rounded=True,
            get_fill_color=[255, 0, 0, 127],
            get_line_color=[255, 0, 0, 127],
            get_line_width=1,
            line_width_min_pixels=1,
            tooltip=True,
        )
        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(
                latitude=self.m.center[1],
                longitude=self.m.center[0],
                zoom=10,
                pitch=0,
            ),
            tooltip={
                "html": "ID: {id}<br>{properties.tooltip}",
                "style": {"color": "white", "word-wrap": "break-word"},
            },  # type: ignore
        )
        return deck
