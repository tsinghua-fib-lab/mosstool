# {py:mod}`mosstool.util.map_splitter`

```{py:module} mosstool.util.map_splitter
```

```{autodoc2-docstring} mosstool.util.map_splitter
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_center_point <mosstool.util.map_splitter._center_point>`
  - ```{autodoc2-docstring} mosstool.util.map_splitter._center_point
    :summary:
    ```
* - {py:obj}`_gen_header <mosstool.util.map_splitter._gen_header>`
  - ```{autodoc2-docstring} mosstool.util.map_splitter._gen_header
    :summary:
    ```
* - {py:obj}`split_map <mosstool.util.map_splitter.split_map>`
  - ```{autodoc2-docstring} mosstool.util.map_splitter.split_map
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <mosstool.util.map_splitter.__all__>`
  - ```{autodoc2-docstring} mosstool.util.map_splitter.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: mosstool.util.map_splitter.__all__
:value: >
   ['split_map']

```{autodoc2-docstring} mosstool.util.map_splitter.__all__
```

````

````{py:function} _center_point(lanes_dict: dict[int, dict], lane_ids: list[int]) -> shapely.geometry.Point
:canonical: mosstool.util.map_splitter._center_point

```{autodoc2-docstring} mosstool.util.map_splitter._center_point
```
````

````{py:function} _gen_header(map_name: str, poly_id: int, proj_str: str, lanes: list[dict]) -> dict
:canonical: mosstool.util.map_splitter._gen_header

```{autodoc2-docstring} mosstool.util.map_splitter._gen_header
```
````

````{py:function} split_map(geo_data: geojson.FeatureCollection, map: mosstool.type.Map, output_path: typing.Optional[str] = None, distance_threshold: float = 50.0) -> dict[typing.Any, mosstool.type.Map]
:canonical: mosstool.util.map_splitter.split_map

```{autodoc2-docstring} mosstool.util.map_splitter.split_map
```
````
