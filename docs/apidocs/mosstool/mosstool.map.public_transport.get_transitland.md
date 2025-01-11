# {py:mod}`mosstool.map.public_transport.get_transitland`

```{py:module} mosstool.map.public_transport.get_transitland
```

```{autodoc2-docstring} mosstool.map.public_transport.get_transitland
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TransitlandPublicTransport <mosstool.map.public_transport.get_transitland.TransitlandPublicTransport>`
  - ```{autodoc2-docstring} mosstool.map.public_transport.get_transitland.TransitlandPublicTransport
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`geo_coords <mosstool.map.public_transport.get_transitland.geo_coords>`
  - ```{autodoc2-docstring} mosstool.map.public_transport.get_transitland.geo_coords
    :summary:
    ```
* - {py:obj}`_get_headers <mosstool.map.public_transport.get_transitland._get_headers>`
  - ```{autodoc2-docstring} mosstool.map.public_transport.get_transitland._get_headers
    :summary:
    ```
* - {py:obj}`cut <mosstool.map.public_transport.get_transitland.cut>`
  - ```{autodoc2-docstring} mosstool.map.public_transport.get_transitland.cut
    :summary:
    ```
* - {py:obj}`_output_data_filter <mosstool.map.public_transport.get_transitland._output_data_filter>`
  - ```{autodoc2-docstring} mosstool.map.public_transport.get_transitland._output_data_filter
    :summary:
    ```
* - {py:obj}`merge_geo <mosstool.map.public_transport.get_transitland.merge_geo>`
  - ```{autodoc2-docstring} mosstool.map.public_transport.get_transitland.merge_geo
    :summary:
    ```
* - {py:obj}`get_sta_dis <mosstool.map.public_transport.get_transitland.get_sta_dis>`
  - ```{autodoc2-docstring} mosstool.map.public_transport.get_transitland.get_sta_dis
    :summary:
    ```
* - {py:obj}`gps_distance <mosstool.map.public_transport.get_transitland.gps_distance>`
  - ```{autodoc2-docstring} mosstool.map.public_transport.get_transitland.gps_distance
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <mosstool.map.public_transport.get_transitland.__all__>`
  - ```{autodoc2-docstring} mosstool.map.public_transport.get_transitland.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: mosstool.map.public_transport.get_transitland.__all__
:value: >
   ['TransitlandPublicTransport']

```{autodoc2-docstring} mosstool.map.public_transport.get_transitland.__all__
```

````

````{py:function} geo_coords(geo)
:canonical: mosstool.map.public_transport.get_transitland.geo_coords

```{autodoc2-docstring} mosstool.map.public_transport.get_transitland.geo_coords
```
````

````{py:function} _get_headers(referer_url)
:canonical: mosstool.map.public_transport.get_transitland._get_headers

```{autodoc2-docstring} mosstool.map.public_transport.get_transitland._get_headers
```
````

````{py:function} cut(line: shapely.geometry.LineString, points: list[shapely.geometry.Point], projstr: str, reverse_line: typing.Optional[shapely.geometry.LineString] = None) -> list
:canonical: mosstool.map.public_transport.get_transitland.cut

```{autodoc2-docstring} mosstool.map.public_transport.get_transitland.cut
```
````

````{py:function} _output_data_filter(output_data: dict, proj_str: str, sta_dis_gate: float)
:canonical: mosstool.map.public_transport.get_transitland._output_data_filter

```{autodoc2-docstring} mosstool.map.public_transport.get_transitland._output_data_filter
```
````

````{py:function} merge_geo(coord, proj_str, square_length=350)
:canonical: mosstool.map.public_transport.get_transitland.merge_geo

```{autodoc2-docstring} mosstool.map.public_transport.get_transitland.merge_geo
```
````

````{py:function} get_sta_dis(sta1, sta2)
:canonical: mosstool.map.public_transport.get_transitland.get_sta_dis

```{autodoc2-docstring} mosstool.map.public_transport.get_transitland.get_sta_dis
```
````

````{py:function} gps_distance(LON1: typing.Union[float, tuple[float, float]], LAT1: typing.Union[float, tuple[float, float]], LON2: typing.Optional[float] = None, LAT2: typing.Optional[float] = None)
:canonical: mosstool.map.public_transport.get_transitland.gps_distance

```{autodoc2-docstring} mosstool.map.public_transport.get_transitland.gps_distance
```
````

`````{py:class} TransitlandPublicTransport(proj_str: str, max_longitude: float, min_longitude: float, max_latitude: float, min_latitude: float, transitland_ak: typing.Optional[str] = None, proxies: typing.Optional[dict[str, str]] = None, wikipedia_name: typing.Optional[str] = None, from_osm: bool = False)
:canonical: mosstool.map.public_transport.get_transitland.TransitlandPublicTransport

```{autodoc2-docstring} mosstool.map.public_transport.get_transitland.TransitlandPublicTransport
```

```{rubric} Initialization
```

```{autodoc2-docstring} mosstool.map.public_transport.get_transitland.TransitlandPublicTransport.__init__
```

````{py:method} _query_raw_data_from_osm()
:canonical: mosstool.map.public_transport.get_transitland.TransitlandPublicTransport._query_raw_data_from_osm

```{autodoc2-docstring} mosstool.map.public_transport.get_transitland.TransitlandPublicTransport._query_raw_data_from_osm
```

````

````{py:method} _process_raw_data_from_osm()
:canonical: mosstool.map.public_transport.get_transitland.TransitlandPublicTransport._process_raw_data_from_osm

```{autodoc2-docstring} mosstool.map.public_transport.get_transitland.TransitlandPublicTransport._process_raw_data_from_osm
```

````

````{py:method} _fetch_raw_stops()
:canonical: mosstool.map.public_transport.get_transitland.TransitlandPublicTransport._fetch_raw_stops

```{autodoc2-docstring} mosstool.map.public_transport.get_transitland.TransitlandPublicTransport._fetch_raw_stops
```

````

````{py:method} _fetch_raw_lines()
:canonical: mosstool.map.public_transport.get_transitland.TransitlandPublicTransport._fetch_raw_lines

```{autodoc2-docstring} mosstool.map.public_transport.get_transitland.TransitlandPublicTransport._fetch_raw_lines
```

````

````{py:method} process_raw_data()
:canonical: mosstool.map.public_transport.get_transitland.TransitlandPublicTransport.process_raw_data

```{autodoc2-docstring} mosstool.map.public_transport.get_transitland.TransitlandPublicTransport.process_raw_data
```

````

````{py:method} merge_raw_data()
:canonical: mosstool.map.public_transport.get_transitland.TransitlandPublicTransport.merge_raw_data

```{autodoc2-docstring} mosstool.map.public_transport.get_transitland.TransitlandPublicTransport.merge_raw_data
```

````

````{py:method} get_output_data()
:canonical: mosstool.map.public_transport.get_transitland.TransitlandPublicTransport.get_output_data

```{autodoc2-docstring} mosstool.map.public_transport.get_transitland.TransitlandPublicTransport.get_output_data
```

````

`````
