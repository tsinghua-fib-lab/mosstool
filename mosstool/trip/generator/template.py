from pycityproto.city.person.v2.person_pb2 import (BikeAttribute,
                                                   EmissionAttribute,
                                                   PedestrianAttribute, Person,
                                                   PersonAttribute,
                                                   VehicleAttribute,
                                                   VehicleEngineEfficiency,
                                                   VehicleEngineType)

__all__ = [
    "DEFAULT_PERSON",
]
DEFAULT_PERSON = Person(
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
