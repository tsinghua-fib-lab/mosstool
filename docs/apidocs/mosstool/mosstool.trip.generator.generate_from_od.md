# {py:mod}`mosstool.trip.generator.generate_from_od`

```{py:module} mosstool.trip.generator.generate_from_od
```

```{autodoc2-docstring} mosstool.trip.generator.generate_from_od
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TripGenerator <mosstool.trip.generator.generate_from_od.TripGenerator>`
  - ```{autodoc2-docstring} mosstool.trip.generator.generate_from_od.TripGenerator
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`geo_coords <mosstool.trip.generator.generate_from_od.geo_coords>`
  - ```{autodoc2-docstring} mosstool.trip.generator.generate_from_od.geo_coords
    :summary:
    ```
* - {py:obj}`_get_mode <mosstool.trip.generator.generate_from_od._get_mode>`
  - ```{autodoc2-docstring} mosstool.trip.generator.generate_from_od._get_mode
    :summary:
    ```
* - {py:obj}`_get_mode_with_distribution <mosstool.trip.generator.generate_from_od._get_mode_with_distribution>`
  - ```{autodoc2-docstring} mosstool.trip.generator.generate_from_od._get_mode_with_distribution
    :summary:
    ```
* - {py:obj}`_match_aoi_unit <mosstool.trip.generator.generate_from_od._match_aoi_unit>`
  - ```{autodoc2-docstring} mosstool.trip.generator.generate_from_od._match_aoi_unit
    :summary:
    ```
* - {py:obj}`_generate_unit <mosstool.trip.generator.generate_from_od._generate_unit>`
  - ```{autodoc2-docstring} mosstool.trip.generator.generate_from_od._generate_unit
    :summary:
    ```
* - {py:obj}`_process_agent_unit <mosstool.trip.generator.generate_from_od._process_agent_unit>`
  - ```{autodoc2-docstring} mosstool.trip.generator.generate_from_od._process_agent_unit
    :summary:
    ```
* - {py:obj}`_fill_sch_unit <mosstool.trip.generator.generate_from_od._fill_sch_unit>`
  - ```{autodoc2-docstring} mosstool.trip.generator.generate_from_od._fill_sch_unit
    :summary:
    ```
* - {py:obj}`_fill_person_schedule_unit <mosstool.trip.generator.generate_from_od._fill_person_schedule_unit>`
  - ```{autodoc2-docstring} mosstool.trip.generator.generate_from_od._fill_person_schedule_unit
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <mosstool.trip.generator.generate_from_od.__all__>`
  - ```{autodoc2-docstring} mosstool.trip.generator.generate_from_od.__all__
    :summary:
    ```
````

### API

````{py:function} geo_coords(geo)
:canonical: mosstool.trip.generator.generate_from_od.geo_coords

```{autodoc2-docstring} mosstool.trip.generator.generate_from_od.geo_coords
```
````

````{py:function} _get_mode(p1, p2)
:canonical: mosstool.trip.generator.generate_from_od._get_mode

```{autodoc2-docstring} mosstool.trip.generator.generate_from_od._get_mode
```
````

````{py:function} _get_mode_with_distribution(partial_args: tuple[list[str], tuple[float, float, float, float, float, float, float, float, float, float, float]], p1: tuple[float, float], p2: tuple[float, float], profile: dict, seed: int = 0)
:canonical: mosstool.trip.generator.generate_from_od._get_mode_with_distribution

```{autodoc2-docstring} mosstool.trip.generator.generate_from_od._get_mode_with_distribution
```
````

````{py:function} _match_aoi_unit(partial_args: tuple[list[typing.Any]], aoi)
:canonical: mosstool.trip.generator.generate_from_od._match_aoi_unit

```{autodoc2-docstring} mosstool.trip.generator.generate_from_od._match_aoi_unit
```
````

````{py:function} _generate_unit(partial_args: tuple[mosstool.trip.generator._util.const.np.ndarray, mosstool.trip.generator._util.const.np.ndarray, mosstool.trip.generator._util.const.np.ndarray, mosstool.trip.generator._util.const.np.ndarray, dict[int, list[dict[str, typing.Any]]], dict[int, dict[str, typing.Any]], dict[str, list[int]], int, typing.Any, list[float], int], get_mode_partial_args: tuple[list[str], tuple[float, float, float, float, float, float, float, float, float, float, float]], a_home_region: int, a_profile: dict[str, typing.Any], modes: list[typing.Union[typing.Literal[mosstool.trip.generator.generate_from_od._generate_unit.H], typing.Literal[mosstool.trip.generator.generate_from_od._generate_unit.W], typing.Literal[mosstool.trip.generator.generate_from_od._generate_unit.E], typing.Literal[mosstool.trip.generator.generate_from_od._generate_unit.O], typing.Literal[+]]], p_mode: list[float], seed: int)
:canonical: mosstool.trip.generator.generate_from_od._generate_unit

```{autodoc2-docstring} mosstool.trip.generator.generate_from_od._generate_unit
```
````

````{py:function} _process_agent_unit(partial_args_tuple: tuple[tuple[mosstool.trip.generator._util.const.np.ndarray, mosstool.trip.generator._util.const.np.ndarray, mosstool.trip.generator._util.const.np.ndarray, mosstool.trip.generator._util.const.np.ndarray, dict[int, list[dict[str, typing.Any]]], dict[int, dict[str, typing.Any]], dict[str, list[int]], int, typing.Any, list[float], int], tuple[list[str], tuple[float, float, float, float, float, float, float, float, float, float, float]]], arg: tuple[int, int, int, dict[str, typing.Any], list[typing.Union[typing.Literal[H], typing.Literal[W], typing.Literal[E], typing.Literal[O], typing.Literal[+]]], list[float]])
:canonical: mosstool.trip.generator.generate_from_od._process_agent_unit

```{autodoc2-docstring} mosstool.trip.generator.generate_from_od._process_agent_unit
```
````

````{py:function} _fill_sch_unit(partial_args: tuple[mosstool.trip.generator._util.const.np.ndarray, dict[int, list[dict[str, typing.Any]]], dict[int, dict[str, typing.Any]], dict[str, list[int]], int, typing.Any, list[float], int], get_mode_partial_args: tuple[list[str], tuple[float, float, float, float, float, float, float, float, float, float, float]], p_home: int, p_home_region: int, p_work: int, p_work_region: int, p_profile: dict[str, typing.Any], modes: list[typing.Union[typing.Literal[H], typing.Literal[W], typing.Literal[E], typing.Literal[mosstool.trip.generator.generate_from_od._fill_sch_unit.O], typing.Literal[+]]], p_mode: list[float], seed: int)
:canonical: mosstool.trip.generator.generate_from_od._fill_sch_unit

```{autodoc2-docstring} mosstool.trip.generator.generate_from_od._fill_sch_unit
```
````

````{py:function} _fill_person_schedule_unit(partial_args_tuple: tuple[tuple[mosstool.trip.generator._util.const.np.ndarray, dict[int, list[dict[str, typing.Any]]], dict[int, dict[str, typing.Any]], dict[str, list[int]], int, typing.Any, list[float], int], tuple[list[str], tuple[float, float, float, float, float, float, float, float, float, float, float]]], arg: tuple[int, int, int, int, int, dict[str, typing.Any], list[typing.Union[typing.Literal[H], typing.Literal[W], typing.Literal[E], typing.Literal[O], typing.Literal[+]]], list[float], int])
:canonical: mosstool.trip.generator.generate_from_od._fill_person_schedule_unit

```{autodoc2-docstring} mosstool.trip.generator.generate_from_od._fill_person_schedule_unit
```
````

````{py:data} __all__
:canonical: mosstool.trip.generator.generate_from_od.__all__
:value: >
   ['TripGenerator']

```{autodoc2-docstring} mosstool.trip.generator.generate_from_od.__all__
```

````

`````{py:class} TripGenerator(m: mosstool.type.Map, pop_tif_path: typing.Optional[str] = None, activity_distributions: typing.Optional[dict] = None, driving_speed: float = 30 / 3.6, parking_fee: float = 20.0, driving_penalty: float = 0.0, subway_speed: float = 35 / 3.6, subway_penalty: float = 600.0, subway_expense: float = 10.0, bus_speed: float = 15 / 3.6, bus_penalty: float = 600.0, bus_expense: float = 5.0, bike_speed: float = 10 / 3.6, bike_penalty: float = 0.0, template_func: collections.abc.Callable[[], mosstool.type.Person] = default_person_template_generator, add_pop: bool = False, multiprocessing_chunk_size: int = 500, workers: int = cpu_count())
:canonical: mosstool.trip.generator.generate_from_od.TripGenerator

```{autodoc2-docstring} mosstool.trip.generator.generate_from_od.TripGenerator
```

```{rubric} Initialization
```

```{autodoc2-docstring} mosstool.trip.generator.generate_from_od.TripGenerator.__init__
```

````{py:method} _read_aois()
:canonical: mosstool.trip.generator.generate_from_od.TripGenerator._read_aois

```{autodoc2-docstring} mosstool.trip.generator.generate_from_od.TripGenerator._read_aois
```

````

````{py:method} _read_regions()
:canonical: mosstool.trip.generator.generate_from_od.TripGenerator._read_regions

```{autodoc2-docstring} mosstool.trip.generator.generate_from_od.TripGenerator._read_regions
```

````

````{py:method} _read_od_matrix()
:canonical: mosstool.trip.generator.generate_from_od.TripGenerator._read_od_matrix

```{autodoc2-docstring} mosstool.trip.generator.generate_from_od.TripGenerator._read_od_matrix
```

````

````{py:method} _match_aoi2region()
:canonical: mosstool.trip.generator.generate_from_od.TripGenerator._match_aoi2region

```{autodoc2-docstring} mosstool.trip.generator.generate_from_od.TripGenerator._match_aoi2region
```

````

````{py:method} _generate_mobi(agent_num: int = 10000, area_pops: typing.Optional[list] = None, person_profiles: typing.Optional[list] = None, seed: int = 0, max_chunk_size: int = 500)
:canonical: mosstool.trip.generator.generate_from_od.TripGenerator._generate_mobi

```{autodoc2-docstring} mosstool.trip.generator.generate_from_od.TripGenerator._generate_mobi
```

````

````{py:method} generate_persons(od_matrix: mosstool.trip.generator._util.const.np.ndarray, areas: geopandas.geodataframe.GeoDataFrame, available_trip_modes: list[str] = ['drive', 'walk', 'bus', 'subway', 'taxi'], departure_time_curve: typing.Optional[list[float]] = None, area_pops: typing.Optional[list] = None, person_profiles: typing.Optional[list[dict]] = None, seed: int = 0, agent_num: typing.Optional[int] = None) -> list[mosstool.type.Person]
:canonical: mosstool.trip.generator.generate_from_od.TripGenerator.generate_persons

```{autodoc2-docstring} mosstool.trip.generator.generate_from_od.TripGenerator.generate_persons
```

````

````{py:method} _get_driving_pos_dict() -> dict[tuple[int, int], mosstool.type.LanePosition]
:canonical: mosstool.trip.generator.generate_from_od.TripGenerator._get_driving_pos_dict

```{autodoc2-docstring} mosstool.trip.generator.generate_from_od.TripGenerator._get_driving_pos_dict
```

````

````{py:method} generate_public_transport_drivers(template_func: typing.Optional[collections.abc.Callable[[], mosstool.type.Person]] = None, stop_duration_time: float = 30.0, seed: int = 0) -> list[mosstool.type.Person]
:canonical: mosstool.trip.generator.generate_from_od.TripGenerator.generate_public_transport_drivers

```{autodoc2-docstring} mosstool.trip.generator.generate_from_od.TripGenerator.generate_public_transport_drivers
```

````

````{py:method} _generate_schedules(input_persons: list[mosstool.type.Person], seed: int)
:canonical: mosstool.trip.generator.generate_from_od.TripGenerator._generate_schedules

```{autodoc2-docstring} mosstool.trip.generator.generate_from_od.TripGenerator._generate_schedules
```

````

````{py:method} fill_person_schedules(input_persons: list[mosstool.type.Person], od_matrix: mosstool.trip.generator._util.const.np.ndarray, areas: geopandas.geodataframe.GeoDataFrame, available_trip_modes: list[str] = ['drive', 'walk', 'bus', 'subway', 'taxi'], departure_time_curve: typing.Optional[list[float]] = None, seed: int = 0) -> list[mosstool.type.Person]
:canonical: mosstool.trip.generator.generate_from_od.TripGenerator.fill_person_schedules

```{autodoc2-docstring} mosstool.trip.generator.generate_from_od.TripGenerator.fill_person_schedules
```

````

````{py:method} generate_taxi_drivers(template_func: typing.Optional[collections.abc.Callable[[], mosstool.type.Person]] = None, parking_positions: typing.Optional[list[typing.Union[mosstool.type.LanePosition, mosstool.type.AoiPosition]]] = None, agent_num: typing.Optional[int] = None, seed: int = 0) -> list[mosstool.type.Person]
:canonical: mosstool.trip.generator.generate_from_od.TripGenerator.generate_taxi_drivers

```{autodoc2-docstring} mosstool.trip.generator.generate_from_od.TripGenerator.generate_taxi_drivers
```

````

`````
