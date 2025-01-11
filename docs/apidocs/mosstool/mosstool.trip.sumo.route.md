# {py:mod}`mosstool.trip.sumo.route`

```{py:module} mosstool.trip.sumo.route
```

```{autodoc2-docstring} mosstool.trip.sumo.route
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RouteConverter <mosstool.trip.sumo.route.RouteConverter>`
  - ```{autodoc2-docstring} mosstool.trip.sumo.route.RouteConverter
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <mosstool.trip.sumo.route.__all__>`
  - ```{autodoc2-docstring} mosstool.trip.sumo.route.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: mosstool.trip.sumo.route.__all__
:value: >
   ['RouteConverter']

```{autodoc2-docstring} mosstool.trip.sumo.route.__all__
```

````

`````{py:class} RouteConverter(converted_map: pycityproto.city.map.v2.map_pb2.Map, sumo_id_mappings: dict, route_path: str, additional_path: typing.Optional[str] = None, seed: typing.Optional[int] = 0)
:canonical: mosstool.trip.sumo.route.RouteConverter

```{autodoc2-docstring} mosstool.trip.sumo.route.RouteConverter
```

```{rubric} Initialization
```

```{autodoc2-docstring} mosstool.trip.sumo.route.RouteConverter.__init__
```

````{py:method} _convert_time(time_str: str) -> numpy.float64
:canonical: mosstool.trip.sumo.route.RouteConverter._convert_time

```{autodoc2-docstring} mosstool.trip.sumo.route.RouteConverter._convert_time
```

````

````{py:method} _convert_route_trips(edges: list, repeat: int, cycle_time: numpy.float64, rid2stop: dict)
:canonical: mosstool.trip.sumo.route.RouteConverter._convert_route_trips

```{autodoc2-docstring} mosstool.trip.sumo.route.RouteConverter._convert_route_trips
```

````

````{py:method} _process_route_trips(t: xml.dom.minidom.Element, route_trips: list, trip_id: int, pre_veh_end: dict, TRIP_MODE: int, ROAD_LANE_TYPE: typing.Union[typing.Literal[walking_lane_ids], typing.Literal[driving_lane_ids]], SPEED: float, departure: numpy.float64, trip_type: typing.Union[typing.Literal[trip], typing.Literal[flow], typing.Literal[vehicle]] = 'flow')
:canonical: mosstool.trip.sumo.route.RouteConverter._process_route_trips

```{autodoc2-docstring} mosstool.trip.sumo.route.RouteConverter._process_route_trips
```

````

````{py:method} _convert_trips_with_route(t: xml.dom.minidom.Element, departure_times: list[numpy.float64], TRIP_MODE: int, ROAD_LANE_TYPE: typing.Union[typing.Literal[walking_lane_ids], typing.Literal[driving_lane_ids]], SPEED: float, trip_id: int, trip_type: typing.Union[typing.Literal[trip], typing.Literal[flow], typing.Literal[vehicle]] = 'flow')
:canonical: mosstool.trip.sumo.route.RouteConverter._convert_trips_with_route

```{autodoc2-docstring} mosstool.trip.sumo.route.RouteConverter._convert_trips_with_route
```

````

````{py:method} _convert_flows_with_from_to(f: xml.dom.minidom.Element, departure_times: list[numpy.float64], flow_id: int, ROAD_LANE_TYPE: typing.Union[typing.Literal[walking_lane_ids], typing.Literal[driving_lane_ids]], TRIP_MODE: int)
:canonical: mosstool.trip.sumo.route.RouteConverter._convert_flows_with_from_to

```{autodoc2-docstring} mosstool.trip.sumo.route.RouteConverter._convert_flows_with_from_to
```

````

````{py:method} _convert_stops(all_stops: list, trip_id: int, trip_type: typing.Union[typing.Literal[trip], typing.Literal[flow], typing.Literal[vehicle]] = 'flow')
:canonical: mosstool.trip.sumo.route.RouteConverter._convert_stops

```{autodoc2-docstring} mosstool.trip.sumo.route.RouteConverter._convert_stops
```

````

````{py:method} _get_trip_position(t: xml.dom.minidom.Element, trip_id: int, road: dict, road_id: int, ROAD_LANE_TYPE: typing.Union[typing.Literal[walking_lane_ids], typing.Literal[driving_lane_ids]], trip_type: typing.Union[typing.Literal[trip], typing.Literal[flow], typing.Literal[vehicle]], attribute: typing.Union[typing.Literal[departLane], typing.Literal[arrivalLane]])
:canonical: mosstool.trip.sumo.route.RouteConverter._get_trip_position

```{autodoc2-docstring} mosstool.trip.sumo.route.RouteConverter._get_trip_position
```

````

````{py:method} _process_agent_type()
:canonical: mosstool.trip.sumo.route.RouteConverter._process_agent_type

```{autodoc2-docstring} mosstool.trip.sumo.route.RouteConverter._process_agent_type
```

````

````{py:method} _route_trips_to_person(route_trips: list, t: xml.dom.minidom.Element, trip_id: int, ROAD_LANE_TYPE: typing.Union[typing.Literal[walking_lane_ids], typing.Literal[driving_lane_ids]], trip_type: typing.Union[typing.Literal[trip], typing.Literal[flow], typing.Literal[vehicle]], TRIP_MODE: int, SPEED: float, departure: numpy.float64)
:canonical: mosstool.trip.sumo.route.RouteConverter._route_trips_to_person

```{autodoc2-docstring} mosstool.trip.sumo.route.RouteConverter._route_trips_to_person
```

````

````{py:method} convert_route()
:canonical: mosstool.trip.sumo.route.RouteConverter.convert_route

```{autodoc2-docstring} mosstool.trip.sumo.route.RouteConverter.convert_route
```

````

`````
