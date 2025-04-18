"""
Frequently Used Protobuf Type
"""

# TODO: pycityproto version checking

from pycityproto.city.geo.v2.geo_pb2 import (AoiPosition, LanePosition,
                                             LongLatPosition, Position)
from pycityproto.city.map.v2.map_pb2 import (Aoi, AoiType, Lane, LaneTurn,
                                             LaneType, Map)
from pycityproto.city.person.v2.person_pb2 import (Consumption, Education,
                                                   Gender, Person,PersonType,
                                                   PersonProfile, Persons)
from pycityproto.city.routing.v2.routing_pb2 import (DrivingJourneyBody,
                                                     Journey, JourneyType)
from pycityproto.city.routing.v2.routing_service_pb2 import (GetRouteRequest,
                                                             GetRouteResponse)
from pycityproto.city.trip.v2.trip_pb2 import Schedule, Trip, TripMode

__all__ = [
    "Map",
    "AoiType",
    "LaneType",
    "LaneTurn",
    "Person",
    "PersonType",
    "Persons",
    "TripMode",
    "Position",
    "AoiPosition",
    "LanePosition",
    "LongLatPosition",
    "GetRouteRequest",
    "GetRouteResponse",
    "Lane",
    "Aoi",
    "Schedule",
    "Trip",
    "Journey",
    "JourneyType",
    "DrivingJourneyBody",
]
