from pycityproto.city.person.v1.person_pb2 import (
    Person,
    PersonAttribute,
    VehicleAttribute,
    PedestrianAttribute,
    BikeAttribute,
)

__all__ = [
    "DEFAULT_PERSON",
]

DEFAULT_PERSON = Person(
    attribute=PersonAttribute(
        length=5,
        width=2,
        max_speed=150 / 3.6,
        max_acceleration=3,
        max_braking_acceleration=-10,
        usual_acceleration=2,
        usual_braking_acceleration=-4.5,
    ),
    vehicle_attribute=VehicleAttribute(lane_change_length=10, min_gap=1),
    pedestrian_attribute=PedestrianAttribute(speed=1.34),
    bike_attribute=BikeAttribute(speed=5),
)
"""
Default person template with attribute and vehicle_attribute.
"""
