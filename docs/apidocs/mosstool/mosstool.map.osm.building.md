# {py:mod}`mosstool.map.osm.building`

```{py:module} mosstool.map.osm.building
```

```{autodoc2-docstring} mosstool.map.osm.building
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Building <mosstool.map.osm.building.Building>`
  - ```{autodoc2-docstring} mosstool.map.osm.building.Building
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <mosstool.map.osm.building.__all__>`
  - ```{autodoc2-docstring} mosstool.map.osm.building.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: mosstool.map.osm.building.__all__
:value: >
   ['Building']

```{autodoc2-docstring} mosstool.map.osm.building.__all__
```

````

`````{py:class} Building(proj_str: typing.Optional[str] = None, max_longitude: typing.Optional[float] = None, min_longitude: typing.Optional[float] = None, max_latitude: typing.Optional[float] = None, min_latitude: typing.Optional[float] = None, wikipedia_name: typing.Optional[str] = None, proxies: typing.Optional[dict[str, str]] = None)
:canonical: mosstool.map.osm.building.Building

```{autodoc2-docstring} mosstool.map.osm.building.Building
```

```{rubric} Initialization
```

```{autodoc2-docstring} mosstool.map.osm.building.Building.__init__
```

````{py:method} _query_raw_data(osm_data_cache: typing.Optional[list[dict]] = None)
:canonical: mosstool.map.osm.building.Building._query_raw_data

```{autodoc2-docstring} mosstool.map.osm.building.Building._query_raw_data
```

````

````{py:method} _make_raw_aoi()
:canonical: mosstool.map.osm.building.Building._make_raw_aoi

```{autodoc2-docstring} mosstool.map.osm.building.Building._make_raw_aoi
```

````

````{py:method} _transform_coordinate(c: tuple[float, float]) -> list[float]
:canonical: mosstool.map.osm.building.Building._transform_coordinate

```{autodoc2-docstring} mosstool.map.osm.building.Building._transform_coordinate
```

````

````{py:method} create_building(output_path: typing.Optional[str] = None, osm_data_cache: typing.Optional[list[dict]] = None, osm_cache_check: bool = False)
:canonical: mosstool.map.osm.building.Building.create_building

```{autodoc2-docstring} mosstool.map.osm.building.Building.create_building
```

````

`````
