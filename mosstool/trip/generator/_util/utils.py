from ....type import TripMode

__all__ = [
    "is_walking",
]


def is_walking(trip_mode: TripMode) -> bool:
    """
    Determine if it is walking mode

    Args:
    - trip_mode (TripMode): mode.

    Returns:
    - bool: Whether it is walking mode.
    """
    return trip_mode in (
        TripMode.TRIP_MODE_BIKE_WALK,
        TripMode.TRIP_MODE_WALK_ONLY,
        TripMode.TRIP_MODE_BUS_WALK,
    )
