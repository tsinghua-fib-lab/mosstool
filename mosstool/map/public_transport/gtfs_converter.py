import os
from typing import Any

import numpy as np
import pandas as pd


def convert_gtfs_to_format(gtfs_dir: str) -> dict[str, Any]:
    """
    Convert GTFS txt files to the target format.

    Args:
        gtfs_dir: Directory containing GTFS txt files (routes.txt, stops.txt, trips.txt, stop_times.txt, shapes.txt)

    Returns:
        Dict containing the converted data in the target format
    """
    # check required files
    required_files = [
        "routes.txt",
        "stops.txt",
        "trips.txt",
        "stop_times.txt",
        "shapes.txt",
    ]
    for file in required_files:
        if not os.path.exists(os.path.join(gtfs_dir, file)):
            raise FileNotFoundError(f"File {file} not found in {gtfs_dir}")
    # Read GTFS files
    routes_df = pd.read_csv(os.path.join(gtfs_dir, "routes.txt"))
    stops_df = pd.read_csv(os.path.join(gtfs_dir, "stops.txt"))
    trips_df = pd.read_csv(os.path.join(gtfs_dir, "trips.txt"))
    stop_times_df = pd.read_csv(os.path.join(gtfs_dir, "stop_times.txt"))
    shapes_df = pd.read_csv(os.path.join(gtfs_dir, "shapes.txt"))

    # Initialize result dictionary
    GTFS_format_data = {}

    # Process each route
    for _, route in routes_df.iterrows():
        route_id = str(route["route_id"])

        # Get trips for this route
        route_trips = trips_df[trips_df["route_id"] == route["route_id"]]

        # Get shapes for this route
        route_shapes = shapes_df[shapes_df["shape_id"].isin(route_trips["shape_id"])]

        # Sort shape points by sequence
        route_shapes = route_shapes.sort_values("shape_pt_sequence")

        # Get coordinates for this route
        coordinates = []
        if not route_shapes.empty:
            # Group by shape_id to handle multiple shapes
            for shape_id, shape_group in route_shapes.groupby("shape_id"):
                shape_coords = [
                    [row["shape_pt_lon"], row["shape_pt_lat"]]
                    for _, row in shape_group.iterrows()
                ]
                coordinates.append(shape_coords)

        # Get stops for this route
        route_stops = []
        for _, trip in route_trips.iterrows():
            trip_stops = stop_times_df[stop_times_df["trip_id"] == trip["trip_id"]]
            trip_stops = trip_stops.sort_values("stop_sequence")

            for _, stop_time in trip_stops.iterrows():
                stop = stops_df[stops_df["stop_id"] == stop_time["stop_id"]].iloc[0]
                stop_info = {
                    "stop": {
                        "stop_name": stop["stop_name"],
                        "id": str(stop["stop_id"]),
                        "geometry": {
                            "coordinates": [stop["stop_lon"], stop["stop_lat"]]
                        },
                    }
                }
                if stop_info not in route_stops:
                    route_stops.append(stop_info)

        # Get schedules for this route
        schedules = []
        for _, trip in route_trips.iterrows():
            trip_stops = stop_times_df[stop_times_df["trip_id"] == trip["trip_id"]]
            trip_stops = trip_stops.sort_values("stop_sequence")

            # Convert departure times to seconds since midnight
            departure_times = []
            for _, stop_time in trip_stops.iterrows():
                # GTFS times are in HH:MM:SS format, convert to seconds
                time_parts = stop_time["departure_time"].split(":")
                seconds = (
                    int(time_parts[0]) * 3600
                    + int(time_parts[1]) * 60
                    + int(time_parts[2])
                )
                departure_times.append(seconds)

            schedules.append(departure_times)
        if len(schedules) > 0:
            schedules = schedules[0]
        else:
            schedules = []

        # Determine route type
        route_type = 1 if route["route_type"] == 1 else 3  # 1 for subway, 3 for bus

        # Create route object
        route_obj = {
            "route_type": route_type,
            "route_long_name": route["route_long_name"],
            "route_short_name": route["route_short_name"],
            "geometry": {"coordinates": coordinates},
            "schedules": schedules,
            "route_stops": route_stops,
        }

        # Add to result
        GTFS_format_data[route_id] = {
            "routes": [route_obj],
            "schedules": schedules,
        }

    return GTFS_format_data
