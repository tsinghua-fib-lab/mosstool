import logging
from typing import cast

from pycityproto.city.person.v1.person_pb2 import Person
from pycityproto.city.routing.v2.routing_pb2 import RouteType
from pycityproto.city.routing.v2.routing_service_pb2 import GetRouteRequest
from pycityproto.city.trip.v2.trip_pb2 import Schedule, TripMode

from .client import RoutingClient

__all__ = ["pre_route"]

_TYPE_MAP = {
    TripMode.TRIP_MODE_DRIVE_ONLY: RouteType.ROUTE_TYPE_DRIVING,
    TripMode.TRIP_MODE_BIKE_WALK: RouteType.ROUTE_TYPE_WALKING,
    TripMode.TRIP_MODE_BUS_WALK: RouteType.ROUTE_TYPE_WALKING,
    TripMode.TRIP_MODE_WALK_ONLY: RouteType.ROUTE_TYPE_WALKING,
}


async def pre_route(
    client: RoutingClient, person: Person, in_place: bool = False
) -> Person:
    """
    Fill in the route of the person's all schedules.
    The function will REMOVE all schedules that can not be routed.

    Args:
    - client (RoutingClient): routing service client
    - person (Person): person object
    - in_place (bool, optional): whether to modify the person object in place. Defaults to False.

    Returns:
    - None
    """
    if not in_place:
        p = Person()
        p.CopyFrom(person)
        person = p
    start = person.home
    departure_time = None
    all_schedules = list(person.schedules)
    person.ClearField("schedules")
    for schedule in all_schedules:
        schedule = cast(Schedule, schedule)
        if schedule.HasField("departure_time"):
            departure_time = schedule.departure_time
        if schedule.loop_count != 1:
            # Schedule is not a one-time trip, departure time is not accurate, no pre-calculation is performed
            logging.warning(
                "Schedule is not a one-time trip, departure time is not accurate, no pre-calculation is performed"
            )
            start = schedule.trips[-1].end
            continue
        good_trips = []
        for trip in schedule.trips:
            last_departure_time = departure_time
            # Cover departure time
            if trip.HasField("departure_time"):
                departure_time = trip.departure_time
            if departure_time is None:
                # No explicit departure time, no pre-calculation is performed
                logging.warning(
                    "No explicit departure time, no pre-calculation is performed"
                )
                # append directly
                good_trips.append(trip)
                # update start position
                start = trip.end
                # Set departure time invalid
                departure_time = None
                continue
            # build request
            res = await client.GetRoute(
                GetRouteRequest(
                    type=_TYPE_MAP[trip.mode],
                    start=start,
                    end=trip.end,
                    time=departure_time,
                )
            )
            if len(res.journeys) == 0:
                logging.warning("No route found")
                departure_time = last_departure_time
            else:
                # append directly
                good_trips.append(trip)
                trip.ClearField("routes")
                trip.routes.MergeFrom(res.journeys)
                # update start position
                start = trip.end
                # Set departure time invalid
                departure_time = None
        if len(good_trips) > 0:
            good_schedule = cast(Schedule, person.schedules.add())
            good_schedule.CopyFrom(schedule)
            good_schedule.ClearField("trips")
            good_schedule.trips.extend(good_trips)
    return person
