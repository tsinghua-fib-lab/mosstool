# {py:mod}`mosstool.trip.gmns.sta`

```{py:module} mosstool.trip.gmns.sta
```

```{autodoc2-docstring} mosstool.trip.gmns.sta
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`STA <mosstool.trip.gmns.sta.STA>`
  - ```{autodoc2-docstring} mosstool.trip.gmns.sta.STA
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <mosstool.trip.gmns.sta.__all__>`
  - ```{autodoc2-docstring} mosstool.trip.gmns.sta.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: mosstool.trip.gmns.sta.__all__
:value: >
   ['STA']

```{autodoc2-docstring} mosstool.trip.gmns.sta.__all__
```

````

`````{py:class} STA(map: mosstool.type.Map, work_dir: str)
:canonical: mosstool.trip.gmns.sta.STA

```{autodoc2-docstring} mosstool.trip.gmns.sta.STA
```

```{rubric} Initialization
```

```{autodoc2-docstring} mosstool.trip.gmns.sta.STA.__init__
```

````{py:method} _get_od(start: mosstool.type.Position, end: mosstool.type.Position)
:canonical: mosstool.trip.gmns.sta.STA._get_od

```{autodoc2-docstring} mosstool.trip.gmns.sta.STA._get_od
```

````

````{py:method} _check_connection(start_road_id: int, end_road_id: int)
:canonical: mosstool.trip.gmns.sta.STA._check_connection

```{autodoc2-docstring} mosstool.trip.gmns.sta.STA._check_connection
```

````

````{py:method} run(persons: mosstool.type.Persons, time_interval: int = 60, reset_routes: bool = False, column_gen_num: int = 10, column_update_num: int = 10)
:canonical: mosstool.trip.gmns.sta.STA.run

```{autodoc2-docstring} mosstool.trip.gmns.sta.STA.run
```

````

`````
