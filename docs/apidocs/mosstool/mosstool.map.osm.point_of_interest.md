# {py:mod}`mosstool.map.osm.point_of_interest`

```{py:module} mosstool.map.osm.point_of_interest
```

```{autodoc2-docstring} mosstool.map.osm.point_of_interest
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PointOfInterest <mosstool.map.osm.point_of_interest.PointOfInterest>`
  - ```{autodoc2-docstring} mosstool.map.osm.point_of_interest.PointOfInterest
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <mosstool.map.osm.point_of_interest.__all__>`
  - ```{autodoc2-docstring} mosstool.map.osm.point_of_interest.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: mosstool.map.osm.point_of_interest.__all__
:value: >
   ['PointOfInterest']

```{autodoc2-docstring} mosstool.map.osm.point_of_interest.__all__
```

````

`````{py:class} PointOfInterest(max_longitude: typing.Optional[float] = None, min_longitude: typing.Optional[float] = None, max_latitude: typing.Optional[float] = None, min_latitude: typing.Optional[float] = None, wikipedia_name: typing.Optional[str] = None, proxies: typing.Optional[dict[str, str]] = None)
:canonical: mosstool.map.osm.point_of_interest.PointOfInterest

```{autodoc2-docstring} mosstool.map.osm.point_of_interest.PointOfInterest
```

```{rubric} Initialization
```

```{autodoc2-docstring} mosstool.map.osm.point_of_interest.PointOfInterest.__init__
```

````{py:method} _query_raw_data(osm_data_cache: typing.Optional[list[dict]] = None)
:canonical: mosstool.map.osm.point_of_interest.PointOfInterest._query_raw_data

```{autodoc2-docstring} mosstool.map.osm.point_of_interest.PointOfInterest._query_raw_data
```

````

````{py:method} _make_raw_poi()
:canonical: mosstool.map.osm.point_of_interest.PointOfInterest._make_raw_poi

```{autodoc2-docstring} mosstool.map.osm.point_of_interest.PointOfInterest._make_raw_poi
```

````

````{py:method} create_pois(output_path: typing.Optional[str] = None, osm_data_cache: typing.Optional[list[dict]] = None, osm_cache_check: bool = False)
:canonical: mosstool.map.osm.point_of_interest.PointOfInterest.create_pois

```{autodoc2-docstring} mosstool.map.osm.point_of_interest.PointOfInterest.create_pois
```

````

`````
