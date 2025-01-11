# {py:mod}`mosstool.map.public_transport.get_subway`

```{py:module} mosstool.map.public_transport.get_subway
```

```{autodoc2-docstring} mosstool.map.public_transport.get_subway
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AmapSubway <mosstool.map.public_transport.get_subway.AmapSubway>`
  - ```{autodoc2-docstring} mosstool.map.public_transport.get_subway.AmapSubway
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`geo_coords <mosstool.map.public_transport.get_subway.geo_coords>`
  - ```{autodoc2-docstring} mosstool.map.public_transport.get_subway.geo_coords
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <mosstool.map.public_transport.get_subway.__all__>`
  - ```{autodoc2-docstring} mosstool.map.public_transport.get_subway.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: mosstool.map.public_transport.get_subway.__all__
:value: >
   ['AmapSubway']

```{autodoc2-docstring} mosstool.map.public_transport.get_subway.__all__
```

````

````{py:function} geo_coords(geo)
:canonical: mosstool.map.public_transport.get_subway.geo_coords

```{autodoc2-docstring} mosstool.map.public_transport.get_subway.geo_coords
```
````

`````{py:class} AmapSubway(city_name_en_us: str, proj_str: str, amap_ak: str)
:canonical: mosstool.map.public_transport.get_subway.AmapSubway

```{autodoc2-docstring} mosstool.map.public_transport.get_subway.AmapSubway
```

```{rubric} Initialization
```

```{autodoc2-docstring} mosstool.map.public_transport.get_subway.AmapSubway.__init__
```

````{py:method} _fetch_raw_data()
:canonical: mosstool.map.public_transport.get_subway.AmapSubway._fetch_raw_data

```{autodoc2-docstring} mosstool.map.public_transport.get_subway.AmapSubway._fetch_raw_data
```

````

````{py:method} _fetch_amap_lines()
:canonical: mosstool.map.public_transport.get_subway.AmapSubway._fetch_amap_lines

```{autodoc2-docstring} mosstool.map.public_transport.get_subway.AmapSubway._fetch_amap_lines
```

````

````{py:method} _merge_stations(subway_stations)
:canonical: mosstool.map.public_transport.get_subway.AmapSubway._merge_stations

```{autodoc2-docstring} mosstool.map.public_transport.get_subway.AmapSubway._merge_stations
```

````

````{py:method} get_output_data()
:canonical: mosstool.map.public_transport.get_subway.AmapSubway.get_output_data

```{autodoc2-docstring} mosstool.map.public_transport.get_subway.AmapSubway.get_output_data
```

````

`````
