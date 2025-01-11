# {py:mod}`mosstool.trip.generator.random`

```{py:module} mosstool.trip.generator.random
```

```{autodoc2-docstring} mosstool.trip.generator.random
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PositionMode <mosstool.trip.generator.random.PositionMode>`
  -
* - {py:obj}`RandomGenerator <mosstool.trip.generator.random.RandomGenerator>`
  - ```{autodoc2-docstring} mosstool.trip.generator.random.RandomGenerator
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <mosstool.trip.generator.random.__all__>`
  - ```{autodoc2-docstring} mosstool.trip.generator.random.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: mosstool.trip.generator.random.__all__
:value: >
   ['PositionMode', 'RandomGenerator']

```{autodoc2-docstring} mosstool.trip.generator.random.__all__
```

````

`````{py:class} PositionMode
:canonical: mosstool.trip.generator.random.PositionMode

Bases: {py:obj}`enum.Enum`

````{py:attribute} AOI
:canonical: mosstool.trip.generator.random.PositionMode.AOI
:value: >
   0

```{autodoc2-docstring} mosstool.trip.generator.random.PositionMode.AOI
```

````

````{py:attribute} LANE
:canonical: mosstool.trip.generator.random.PositionMode.LANE
:value: >
   1

```{autodoc2-docstring} mosstool.trip.generator.random.PositionMode.LANE
```

````

`````

`````{py:class} RandomGenerator(m: pycityproto.city.map.v2.map_pb2.Map, position_modes: list[mosstool.trip.generator.random.PositionMode], trip_mode: pycityproto.city.trip.v2.trip_pb2.TripMode, template_func: collections.abc.Callable[[], pycityproto.city.person.v2.person_pb2.Person] = default_person_template_generator)
:canonical: mosstool.trip.generator.random.RandomGenerator

```{autodoc2-docstring} mosstool.trip.generator.random.RandomGenerator
```

```{rubric} Initialization
```

```{autodoc2-docstring} mosstool.trip.generator.random.RandomGenerator.__init__
```

````{py:method} _rand_position(candidates: typing.Union[list[pycityproto.city.map.v2.map_pb2.Aoi], list[pycityproto.city.map.v2.map_pb2.Lane]])
:canonical: mosstool.trip.generator.random.RandomGenerator._rand_position

```{autodoc2-docstring} mosstool.trip.generator.random.RandomGenerator._rand_position
```

````

````{py:method} uniform(num: int, first_departure_time_range: tuple[float, float], schedule_interval_range: tuple[float, float], seed: typing.Optional[int] = None, start_id: typing.Optional[int] = None) -> list[pycityproto.city.person.v2.person_pb2.Person]
:canonical: mosstool.trip.generator.random.RandomGenerator.uniform

```{autodoc2-docstring} mosstool.trip.generator.random.RandomGenerator.uniform
```

````

`````
