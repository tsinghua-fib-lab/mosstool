from pycityproto.city.person.v1.person_pb2 import \
    BikeAttribute as v1BikeAttribute
from pycityproto.city.person.v1.person_pb2 import \
    PedestrianAttribute as v1PedestrianAttribute
from pycityproto.city.person.v1.person_pb2 import Person as v1Person
from pycityproto.city.person.v1.person_pb2 import \
    PersonAttribute as v1PersonAttribute
from pycityproto.city.person.v1.person_pb2 import \
    VehicleAttribute as v1VehicleAttribute
from pycityproto.city.person.v2.person_pb2 import (BikeAttribute,
                                                   EmissionAttribute,
                                                   PedestrianAttribute, Person as v2Person,
                                                   PersonAttribute,
                                                   VehicleAttribute,
                                                   VehicleEngineEfficiency,
                                                   VehicleEngineType)

__all__ = [
    "V1_DEFAULT_PERSON",
    "V2_DEFAULT_PERSON",
]

V1_DEFAULT_PERSON = v1Person(
    attribute=v1PersonAttribute(
        length=5,
        width=2,
        max_speed=150 / 3.6,
        max_acceleration=3,
        max_braking_acceleration=-10,
        usual_acceleration=2,
        usual_braking_acceleration=-4.5,
    ),
    vehicle_attribute=v1VehicleAttribute(
        lane_change_length=10,
        min_gap=1,
    ),
    pedestrian_attribute=v1PedestrianAttribute(speed=1.34),
    bike_attribute=v1BikeAttribute(speed=5),
)
V2_DEFAULT_PERSON = v2Person(
    attribute=PersonAttribute(),
    vehicle_attribute=VehicleAttribute(
        length=5,
        width=2,
        max_speed=150 / 3.6,
        max_acceleration=3,
        max_braking_acceleration=-10,
        usual_acceleration=2,
        usual_braking_acceleration=-4.5,
        headway=1.5,
        lane_max_speed_recognition_deviation=1.0,
        lane_change_length=10,
        min_gap=1,
        emission_attribute=EmissionAttribute(
            weight=2100,
            type=VehicleEngineType.VEHICLE_ENGINE_TYPE_FUEL,
            coefficient_drag=0.251,
            lambda_s=0.29,
            frontal_area=2.52,
            fuel_efficiency=VehicleEngineEfficiency(
                energy_conversion_efficiency=0.27 * 0.049
            ),
        ),
    ),
    pedestrian_attribute=PedestrianAttribute(speed=1.34),
    bike_attribute=BikeAttribute(speed=5),
)
"""
Default person template with attribute and vehicle_attribute.
"""
