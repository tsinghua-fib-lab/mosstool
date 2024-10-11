from typing import List, Literal, Optional, Union

import numpy as np
from pycityproto.city.person.v2.person_pb2 import (BikeAttribute,
                                                   EmissionAttribute,
                                                   PedestrianAttribute, Person,
                                                   PersonAttribute,
                                                   VehicleAttribute,
                                                   VehicleEngineEfficiency,
                                                   VehicleEngineType)

from ._util.const import (CALIBRATED_BRAKE_ACC_PDF,
                          CALIBRATED_BRAKE_ACC_VALUES, CALIBRATED_HEADWAY_PDF,
                          CALIBRATED_HEADWAY_VALUES, CALIBRATED_MAX_ACC_PDF,
                          CALIBRATED_MAX_ACC_VALUES)

__all__ = [
    "default_vehicle_template_generator",
    "default_bus_template_generator",
    "ProbabilisticTemplateGenerator",
]


def default_vehicle_template_generator() -> Person:
    return Person(
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
                    energy_conversion_efficiency=0.27 * 0.049,
                    c_ef=66.98,
                ),
            ),
            model="normal",
        ),
        pedestrian_attribute=PedestrianAttribute(speed=1.34, model="normal"),
        bike_attribute=BikeAttribute(speed=5, model="normal"),
    )


def default_bus_template_generator() -> Person:
    return Person(
        attribute=PersonAttribute(),
        vehicle_attribute=VehicleAttribute(
            length=15,
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
                weight=18000,
                type=VehicleEngineType.VEHICLE_ENGINE_TYPE_FUEL,
                coefficient_drag=0.251,
                lambda_s=0.29,
                frontal_area=8.67,
                fuel_efficiency=VehicleEngineEfficiency(
                    energy_conversion_efficiency=0.27 * 0.049,
                    c_ef=70.27,
                ),
            ),
            model="normal",
        ),
        pedestrian_attribute=PedestrianAttribute(speed=1.34, model="normal"),
        bike_attribute=BikeAttribute(speed=5, model="normal"),
    )


class ProbabilisticTemplateGenerator:
    def __init__(
        self,
        max_speed_values: Optional[List[float]] = None,
        max_speed_probabilities: Optional[List[float]] = None,
        max_acceleration_values: Optional[List[float]] = None,
        max_acceleration_probabilities: Optional[List[float]] = None,
        max_braking_acceleration_values: Optional[List[float]] = None,
        max_braking_acceleration_probabilities: Optional[List[float]] = None,
        usual_braking_acceleration_values: Optional[List[float]] = None,
        usual_braking_acceleration_probabilities: Optional[List[float]] = None,
        headway_values: Optional[List[float]] = None,
        headway_probabilities: Optional[List[float]] = None,
        min_gap_values: Optional[List[float]] = None,
        min_gap_probabilities: Optional[List[float]] = None,
        seed: int = 0,
        template: Optional[Person] = None,
    ):
        """
        Args:
        - max_speed_values (Optional[List[float]]): A list of possible maximum speeds.
        - max_speed_probabilities (Optional[List[float]]): Probabilities corresponding to max_speed_values.
        - max_acceleration_values (Optional[List[float]]): A list of possible maximum accelerations.
        - max_acceleration_probabilities (Optional[List[float]]): Probabilities corresponding to max_acceleration_values.
        - max_braking_acceleration_values (Optional[List[float]]): A list of possible maximum braking accelerations.
        - max_braking_acceleration_probabilities (Optional[List[float]]): Probabilities corresponding to max_braking_acceleration_values.
        - usual_braking_acceleration_values (Optional[List[float]]): A list of usual braking accelerations.
        - usual_braking_acceleration_probabilities (Optional[List[float]]): Probabilities corresponding to usual_braking_acceleration_values.
        - headway_values (Optional[List[float]]): A list of safe time headways.
        - headway_probabilities (Optional[List[float]]): Probabilities corresponding to headway_values.
        - min_gap_values (Optional[List[float]]): A list of minimum gaps.
        - min_gap_probabilities (Optional[List[float]]): Probabilities corresponding to min_gap_values.
        - seed (int): Seed value for the random number generator.
        - template (Optional[Person]): The template function of generated person object.
        """
        self.template_p = template
        if self.template_p is None:
            self.template_p = default_vehicle_template_generator()
        self.template_p.ClearField("schedules")
        # max speed
        self.max_speed_values = max_speed_values
        self.max_speed_probabilities = max_speed_probabilities
        if max_speed_probabilities is not None and max_speed_values is not None:
            _sum = sum(max_speed_probabilities)
            self.max_speed_probabilities = np.array(
                [d / _sum for d in max_speed_probabilities]
            )
            assert len(max_speed_probabilities) == len(
                max_speed_values
            ), f"Inconsistent length between max speed values and probabilities"
        else:
            self.max_speed_values = None
            self.max_speed_probabilities = None
        # max acceleration
        self.max_acceleration_values = max_acceleration_values
        self.max_acceleration_probabilities = max_acceleration_probabilities
        if (
            max_acceleration_probabilities is not None
            and max_acceleration_values is not None
        ):
            _sum = sum(max_acceleration_probabilities)
            self.max_acceleration_probabilities = np.array(
                [d / _sum for d in max_acceleration_probabilities]
            )
            assert len(max_acceleration_probabilities) == len(
                max_acceleration_values
            ), f"Inconsistent length between max acceleration values and probabilities"
        else:
            self.max_acceleration_values = None
            self.max_acceleration_probabilities = None
        # max braking acceleration
        self.max_braking_acceleration_values = max_braking_acceleration_values
        self.max_braking_acceleration_probabilities = (
            max_braking_acceleration_probabilities
        )
        if (
            max_braking_acceleration_probabilities is not None
            and max_braking_acceleration_values is not None
        ):
            _sum = sum(max_braking_acceleration_probabilities)
            self.max_braking_acceleration_probabilities = np.array(
                [d / _sum for d in max_braking_acceleration_probabilities]
            )
            assert len(max_braking_acceleration_probabilities) == len(
                max_braking_acceleration_values
            ), f"Inconsistent length between max braking acceleration values and probabilities"
        else:
            self.max_braking_acceleration_values = None
            self.max_braking_acceleration_probabilities = None
        # usual braking acceleration
        self.usual_braking_acceleration_values = usual_braking_acceleration_values
        self.usual_braking_acceleration_probabilities = (
            usual_braking_acceleration_probabilities
        )
        if (
            usual_braking_acceleration_probabilities is not None
            and usual_braking_acceleration_values is not None
        ):
            _sum = sum(usual_braking_acceleration_probabilities)
            self.usual_braking_acceleration_probabilities = np.array(
                [d / _sum for d in usual_braking_acceleration_probabilities]
            )
            assert len(usual_braking_acceleration_probabilities) == len(
                usual_braking_acceleration_values
            ), f"Inconsistent length between usual braking acceleration values and probabilities"
        else:
            self.usual_braking_acceleration_values = None
            self.usual_braking_acceleration_probabilities = None
        # safe time headway
        self.headway_values = headway_values
        self.headway_probabilities = headway_probabilities
        if headway_probabilities is not None and headway_values is not None:
            _sum = sum(headway_probabilities)
            self.headway_probabilities = np.array(
                [d / _sum for d in headway_probabilities]
            )
            assert len(headway_probabilities) == len(
                headway_values
            ), f"Inconsistent length between headway values and probabilities"
        else:
            self.headway_values = None
            self.headway_probabilities = None
        # min gap
        self.min_gap_values = min_gap_values
        self.min_gap_probabilities = min_gap_probabilities
        if min_gap_probabilities is not None and min_gap_values is not None:
            _sum = sum(min_gap_probabilities)
            self.min_gap_probabilities = np.array(
                [d / _sum for d in min_gap_probabilities]
            )
            assert len(min_gap_probabilities) == len(
                min_gap_values
            ), f"Inconsistent length between minGap values and probabilities"
        else:
            self.min_gap_values = None
            self.min_gap_probabilities = None
        # radom engine
        self.rng = np.random.default_rng(seed)

    def template_generator(
        self,
    ) -> Person:
        rng = self.rng
        p = Person()
        assert self.template_p is not None
        p.CopyFrom(self.template_p)
        # max speed
        if (
            self.max_speed_probabilities is not None
            and self.max_speed_values is not None
        ):
            p.vehicle_attribute.max_speed = rng.choice(
                self.max_speed_values, p=self.max_speed_probabilities
            )
        # max acceleration
        if (
            self.max_acceleration_probabilities is not None
            and self.max_acceleration_values is not None
        ):
            p.vehicle_attribute.max_acceleration = rng.choice(
                self.max_acceleration_values, p=self.max_acceleration_probabilities
            )
        # max braking acceleration
        if (
            self.max_braking_acceleration_probabilities is not None
            and self.max_braking_acceleration_values is not None
        ):
            p.vehicle_attribute.max_braking_acceleration = rng.choice(
                self.max_braking_acceleration_values,
                p=self.max_braking_acceleration_probabilities,
            )
        # usual braking acceleration
        if (
            self.usual_braking_acceleration_probabilities is not None
            and self.usual_braking_acceleration_values is not None
        ):
            p.vehicle_attribute.usual_braking_acceleration = rng.choice(
                self.usual_braking_acceleration_values,
                p=self.usual_braking_acceleration_probabilities,
            )
        # safe time headway
        if self.headway_probabilities is not None and self.headway_values is not None:
            p.vehicle_attribute.headway = rng.choice(
                self.headway_values, p=self.headway_probabilities
            )
        # min gap
        if self.min_gap_probabilities is not None and self.min_gap_values is not None:
            p.vehicle_attribute.min_gap = rng.choice(
                self.min_gap_values, p=self.min_gap_probabilities
            )
        return p


class GaussianTemplateGenerator:
    def __init__(
        self,
        max_speed_mean: Optional[float] = None,
        max_speed_std: Optional[float] = None,
        max_acceleration_mean: Optional[float] = None,
        max_acceleration_std: Optional[float] = None,
        max_braking_acceleration_mean: Optional[float] = None,
        max_braking_acceleration_std: Optional[float] = None,
        usual_braking_acceleration_mean: Optional[float] = None,
        usual_braking_acceleration_std: Optional[float] = None,
        headway_mean: Optional[float] = None,
        headway_std: Optional[float] = None,
        min_gap_mean: Optional[float] = None,
        min_gap_std: Optional[float] = None,
        seed: int = 0,
        template: Optional[Person] = None,
    ) -> None:
        """
        Args
        - max_speed_mean (Optional[float]): Mean of the Gaussian distribution for maximum speeds.
        - max_speed_std (Optional[float]): Standard deviation corresponding to max_speed_mean.
        - max_acceleration_mean (Optional[float]): Mean of the Gaussian distribution for maximum accelerations.
        - max_acceleration_std (Optional[float]): Standard deviation corresponding to max_acceleration_mean.
        - max_braking_acceleration_mean (Optional[float]): Mean of the Gaussian distribution for maximum braking accelerations.
        - max_braking_acceleration_std (Optional[float]): Standard deviation corresponding to max_braking_acceleration_mean.
        - usual_braking_acceleration_mean (Optional[float]): Mean of the Gaussian distribution for usual braking accelerations.
        - usual_braking_acceleration_std (Optional[float]): Standard deviation corresponding to usual_braking_acceleration_mean.
        - headway_mean (Optional[float]): Mean of the Gaussian distribution for safe time headways.
        - headway_std (Optional[float]): Standard deviation corresponding to headway_mean.
        - min_gap_mean (Optional[float]): Mean of the Gaussian distribution for minimum gaps.
        - min_gap_std (Optional[float]): Standard deviation corresponding to min_gap_mean.
        - seed (int): Seed value for the random number generator.
        - template (Optional[Person]): The template function of generated person object.
        """
        self.template_p = template
        if self.template_p is None:
            self.template_p = default_vehicle_template_generator()
        self.template_p.ClearField("schedules")
        # max speed
        self.max_speed_mean = max_speed_mean
        self.max_speed_std = max_speed_std
        # max acceleration
        self.max_acceleration_mean = max_acceleration_mean
        self.max_acceleration_std = max_acceleration_std
        # max braking acceleration
        self.max_braking_acceleration_mean = max_braking_acceleration_mean
        self.max_braking_acceleration_std = max_braking_acceleration_std
        # usual braking acceleration
        self.usual_braking_acceleration_mean = usual_braking_acceleration_mean
        self.usual_braking_acceleration_std = usual_braking_acceleration_std
        # safe time headway
        self.headway_mean = headway_mean
        self.headway_std = headway_std
        # min gap
        self.min_gap_mean = min_gap_mean
        self.min_gap_std = min_gap_std
        # radom engine
        self.rng = np.random.default_rng(seed)

    def template_generator(
        self,
    ) -> Person:
        rng = self.rng
        p = Person()
        assert self.template_p is not None
        p.CopyFrom(self.template_p)
        # max speed
        if self.max_speed_mean is not None and self.max_speed_std is not None:
            p.vehicle_attribute.max_speed = np.abs(
                np.sqrt(self.max_speed_std) * rng.normal() + self.max_speed_mean
            )
        # max acceleration
        if (
            self.max_acceleration_mean is not None
            and self.max_acceleration_std is not None
        ):
            p.vehicle_attribute.max_acceleration = np.abs(
                np.sqrt(self.max_acceleration_std) * rng.normal()
                + self.max_acceleration_mean
            )
        # max braking acceleration
        if (
            self.max_braking_acceleration_mean is not None
            and self.max_braking_acceleration_std is not None
        ):
            p.vehicle_attribute.max_braking_acceleration = -np.abs(
                np.sqrt(self.max_braking_acceleration_std) * rng.normal()
                + self.max_braking_acceleration_std
            )
        # usual braking acceleration
        if (
            self.usual_braking_acceleration_mean is not None
            and self.usual_braking_acceleration_std is not None
        ):
            p.vehicle_attribute.usual_braking_acceleration = -np.abs(
                np.sqrt(self.usual_braking_acceleration_std) * rng.normal()
                + self.usual_braking_acceleration_mean
            )
        # safe time headway
        if self.headway_mean is not None and self.headway_std is not None:
            p.vehicle_attribute.headway = np.abs(
                np.sqrt(self.headway_std) * rng.normal() + self.headway_mean
            )
        # min gap
        if self.min_gap_mean is not None and self.min_gap_std is not None:
            p.vehicle_attribute.min_gap = np.abs(
                np.sqrt(self.min_gap_std) * rng.normal() + self.min_gap_mean
            )
        return p


class UniformTemplateGenerator:
    def __init__(
        self,
        max_speed_min: Optional[float] = None,
        max_speed_max: Optional[float] = None,
        max_acceleration_min: Optional[float] = None,
        max_acceleration_max: Optional[float] = None,
        max_braking_acceleration_min: Optional[float] = None,
        max_braking_acceleration_max: Optional[float] = None,
        usual_braking_acceleration_min: Optional[float] = None,
        usual_braking_acceleration_max: Optional[float] = None,
        headway_min: Optional[float] = None,
        headway_max: Optional[float] = None,
        min_gap_min: Optional[float] = None,
        min_gap_max: Optional[float] = None,
        seed: int = 0,
        template: Optional[Person] = None,
    ) -> None:
        """
        Args
        - max_speed_mean (Optional[float]): Lower bound of the Uniform distribution for maximum speeds.
        - max_speed_std (Optional[float]): Higher bound of the Uniform distribution for maximum speeds.
        - max_acceleration_mean (Optional[float]): Lower bound of the Uniform distribution for maximum accelerations.
        - max_acceleration_std (Optional[float]): Higher bound of the Uniform distribution for maximum accelerations.
        - max_braking_acceleration_mean (Optional[float]): Lower bound of the Uniform distribution for maximum braking accelerations.
        - max_braking_acceleration_std (Optional[float]): Higher bound of the Uniform distribution for maximum braking accelerations.
        - usual_braking_acceleration_mean (Optional[float]): Lower bound of the Uniform distribution for usual braking accelerations.
        - usual_braking_acceleration_std (Optional[float]): Higher bound of the Uniform distribution for usual braking accelerations.
        - headway_mean (Optional[float]): Lower bound of the Uniform distribution for safe time headways.
        - headway_std (Optional[float]): Higher bound of the Uniform distribution for safe time headways.
        - min_gap_mean (Optional[float]): Lower bound of the Uniform distribution for minimum gaps.
        - min_gap_std (Optional[float]): Higher bound of the Uniform distribution for minimum gaps.
        - seed (int): Seed value for the random number generator.
        - template (Optional[Person]): The template function of generated person object.
        """
        self.template_p = template
        if self.template_p is None:
            self.template_p = default_vehicle_template_generator()
        self.template_p.ClearField("schedules")
        # max speed
        self.max_speed_min = max_speed_min
        self.max_speed_max = max_speed_max
        # max acceleration
        self.max_acceleration_min = max_acceleration_min
        self.max_acceleration_max = max_acceleration_max
        # max braking acceleration
        self.max_braking_acceleration_min = max_braking_acceleration_min
        self.max_braking_acceleration_max = max_braking_acceleration_max
        # usual braking acceleration
        self.usual_braking_acceleration_min = usual_braking_acceleration_min
        self.usual_braking_acceleration_max = usual_braking_acceleration_max
        # safe time headway
        self.headway_min = headway_min
        self.headway_max = headway_max
        # min gap
        self.min_gap_min = min_gap_min
        self.min_gap_max = min_gap_max
        # radom engine
        self.rng = np.random.default_rng(seed)

    def template_generator(
        self,
    ) -> Person:
        rng = self.rng
        p = Person()
        assert self.template_p is not None
        p.CopyFrom(self.template_p)
        # max speed
        if self.max_speed_min is not None and self.max_speed_max is not None:
            p.vehicle_attribute.max_speed = np.abs(
                rng.uniform(self.max_speed_min, self.max_speed_max)
            )
        # max acceleration
        if (
            self.max_acceleration_min is not None
            and self.max_acceleration_max is not None
        ):
            p.vehicle_attribute.max_acceleration = np.abs(
                rng.uniform(self.max_acceleration_min, self.max_acceleration_max)
            )
        # max braking acceleration
        if (
            self.max_braking_acceleration_min is not None
            and self.max_braking_acceleration_max is not None
        ):
            p.vehicle_attribute.max_braking_acceleration = -np.abs(
                rng.uniform(
                    self.max_braking_acceleration_min, self.max_braking_acceleration_max
                )
            )
        # usual braking acceleration
        if (
            self.usual_braking_acceleration_min is not None
            and self.usual_braking_acceleration_max is not None
        ):
            p.vehicle_attribute.usual_braking_acceleration = -np.abs(
                rng.uniform(
                    self.usual_braking_acceleration_min,
                    self.usual_braking_acceleration_max,
                )
            )
        # safe time headway
        if self.headway_min is not None and self.headway_max is not None:
            p.vehicle_attribute.headway = np.abs(
                rng.uniform(self.headway_min, self.headway_max)
            )
        # min gap
        if self.min_gap_min is not None and self.min_gap_max is not None:
            p.vehicle_attribute.min_gap = np.abs(
                rng.uniform(self.min_gap_min, self.min_gap_max)
            )
        return p


class CalibratedTemplateGenerator:
    def __init__(self, seed: int = 0, template: Optional[Person] = None) -> None:
        """
        Args
        - seed (int): Seed value for the random number generator.
        - template (Optional[Person]): The template function of generated person object.
        """
        self.template_p = template
        if self.template_p is None:
            self.template_p = default_vehicle_template_generator()
        self.template_p.ClearField("schedules")
        # radom engine
        self.rng = np.random.default_rng(seed)

    def template_generator(
        self,
    ) -> Person:
        rng = self.rng
        p = Person()
        assert self.template_p is not None
        p.CopyFrom(self.template_p)
        # max acceleration
        p.vehicle_attribute.max_acceleration = rng.choice(
            CALIBRATED_MAX_ACC_VALUES, p=CALIBRATED_MAX_ACC_PDF
        )
        # usual braking acceleration
        p.vehicle_attribute.usual_braking_acceleration = rng.choice(
            CALIBRATED_BRAKE_ACC_VALUES, p=CALIBRATED_BRAKE_ACC_PDF
        )
        # safe time headway
        p.vehicle_attribute.headway = rng.choice(
            CALIBRATED_HEADWAY_VALUES, p=CALIBRATED_HEADWAY_PDF
        )
        return p
