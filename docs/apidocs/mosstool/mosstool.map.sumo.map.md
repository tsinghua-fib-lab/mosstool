# {py:mod}`mosstool.map.sumo.map`

```{py:module} mosstool.map.sumo.map
```

```{autodoc2-docstring} mosstool.map.sumo.map
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MapConverter <mosstool.map.sumo.map.MapConverter>`
  - ```{autodoc2-docstring} mosstool.map.sumo.map.MapConverter
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <mosstool.map.sumo.map.__all__>`
  - ```{autodoc2-docstring} mosstool.map.sumo.map.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: mosstool.map.sumo.map.__all__
:value: >
   ['MapConverter']

```{autodoc2-docstring} mosstool.map.sumo.map.__all__
```

````

`````{py:class} MapConverter(net_path: str, default_lane_width: float = 3.2, green_time: float = 60.0, yellow_time: float = 5.0, poly_path: typing.Optional[str] = None, additional_path: typing.Optional[str] = None, traffic_light_path: typing.Optional[str] = None, traffic_light_min_direction_group: int = 3, merge_aoi: bool = False, enable_tqdm: bool = False, multiprocessing_chunk_size: int = 500, traffic_light_mode: typing.Union[typing.Literal[green_red], typing.Literal[green_yellow_red], typing.Literal[green_yellow_clear_red]] = 'green_yellow_clear_red', workers: int = cpu_count())
:canonical: mosstool.map.sumo.map.MapConverter

```{autodoc2-docstring} mosstool.map.sumo.map.MapConverter
```

```{rubric} Initialization
```

```{autodoc2-docstring} mosstool.map.sumo.map.MapConverter.__init__
```

````{py:method} _connect_lane(in_uid, out_uid, orig_lid, lane_turn, lane_type, junc_id, max_speed)
:canonical: mosstool.map.sumo.map.MapConverter._connect_lane

```{autodoc2-docstring} mosstool.map.sumo.map.MapConverter._connect_lane
```

````

````{py:method} _create_junctions()
:canonical: mosstool.map.sumo.map.MapConverter._create_junctions

```{autodoc2-docstring} mosstool.map.sumo.map.MapConverter._create_junctions
```

````

````{py:method} _add_lane_conn()
:canonical: mosstool.map.sumo.map.MapConverter._add_lane_conn

```{autodoc2-docstring} mosstool.map.sumo.map.MapConverter._add_lane_conn
```

````

````{py:method} _add_junc_lane_overlaps()
:canonical: mosstool.map.sumo.map.MapConverter._add_junc_lane_overlaps

```{autodoc2-docstring} mosstool.map.sumo.map.MapConverter._add_junc_lane_overlaps
```

````

````{py:method} _add_driving_lane_group()
:canonical: mosstool.map.sumo.map.MapConverter._add_driving_lane_group

```{autodoc2-docstring} mosstool.map.sumo.map.MapConverter._add_driving_lane_group
```

````

````{py:method} _add_traffic_light()
:canonical: mosstool.map.sumo.map.MapConverter._add_traffic_light

```{autodoc2-docstring} mosstool.map.sumo.map.MapConverter._add_traffic_light
```

````

````{py:method} _get_output_map()
:canonical: mosstool.map.sumo.map.MapConverter._get_output_map

```{autodoc2-docstring} mosstool.map.sumo.map.MapConverter._get_output_map
```

````

````{py:method} _add_aois_to_map()
:canonical: mosstool.map.sumo.map.MapConverter._add_aois_to_map

```{autodoc2-docstring} mosstool.map.sumo.map.MapConverter._add_aois_to_map
```

````

````{py:method} convert_map()
:canonical: mosstool.map.sumo.map.MapConverter.convert_map

```{autodoc2-docstring} mosstool.map.sumo.map.MapConverter.convert_map
```

````

````{py:method} get_sumo_id_mappings()
:canonical: mosstool.map.sumo.map.MapConverter.get_sumo_id_mappings

```{autodoc2-docstring} mosstool.map.sumo.map.MapConverter.get_sumo_id_mappings
```

````

`````
