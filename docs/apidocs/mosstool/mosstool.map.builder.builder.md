# {py:mod}`mosstool.map.builder.builder`

```{py:module} mosstool.map.builder.builder
```

```{autodoc2-docstring} mosstool.map.builder.builder
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Builder <mosstool.map.builder.builder.Builder>`
  - ```{autodoc2-docstring} mosstool.map.builder.builder.Builder
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <mosstool.map.builder.builder.__all__>`
  - ```{autodoc2-docstring} mosstool.map.builder.builder.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: mosstool.map.builder.builder.__all__
:value: >
   ['Builder']

```{autodoc2-docstring} mosstool.map.builder.builder.__all__
```

````

`````{py:class} Builder(net: typing.Union[geojson.FeatureCollection, mosstool.type.Map], proj_str: typing.Optional[str] = None, aois: typing.Optional[geojson.FeatureCollection] = None, pois: typing.Optional[geojson.FeatureCollection] = None, public_transport: typing.Optional[dict[str, list]] = None, pop_tif_path: typing.Optional[str] = None, landuse_shp_path: typing.Optional[str] = None, traffic_light_min_direction_group: int = 3, default_lane_width: float = 3.2, gen_sidewalk_speed_limit: float = 0, gen_sidewalk_length_limit: float = 5.0, expand_roads: bool = False, road_expand_mode: typing.Union[typing.Literal[L], typing.Literal[M], typing.Literal[R]] = 'R', aoi_mode: typing.Union[typing.Literal[append], typing.Literal[overwrite]] = 'overwrite', traffic_light_mode: typing.Union[typing.Literal[green_red], typing.Literal[green_yellow_red], typing.Literal[green_yellow_clear_red]] = 'green_yellow_clear_red', multiprocessing_chunk_size: int = 500, green_time: float = 30.0, yellow_time: float = 5.0, strict_mode: bool = False, merge_aoi: bool = True, aoi_matching_distance_threshold: float = 30.0, pt_station_matching_distance_threshold: float = 30.0, pt_station_matching_distance_relaxation_threshold: float = 30.0, output_lane_length_check: bool = False, enable_tqdm: bool = False, workers: int = cpu_count())
:canonical: mosstool.map.builder.builder.Builder

```{autodoc2-docstring} mosstool.map.builder.builder.Builder
```

```{rubric} Initialization
```

```{autodoc2-docstring} mosstool.map.builder.builder.Builder.__init__
```

````{py:attribute} uid_mapping
:canonical: mosstool.map.builder.builder.Builder.uid_mapping
:value: >
   None

```{autodoc2-docstring} mosstool.map.builder.builder.Builder.uid_mapping
```

````

````{py:attribute} _junction_keys
:canonical: mosstool.map.builder.builder.Builder._junction_keys
:value: >
   []

```{autodoc2-docstring} mosstool.map.builder.builder.Builder._junction_keys
```

````

````{py:attribute} map_roads
:canonical: mosstool.map.builder.builder.Builder.map_roads
:value: >
   None

```{autodoc2-docstring} mosstool.map.builder.builder.Builder.map_roads
```

````

````{py:attribute} map_junctions
:canonical: mosstool.map.builder.builder.Builder.map_junctions
:value: >
   None

```{autodoc2-docstring} mosstool.map.builder.builder.Builder.map_junctions
```

````

````{py:attribute} lane2data
:canonical: mosstool.map.builder.builder.Builder.lane2data
:value: >
   None

```{autodoc2-docstring} mosstool.map.builder.builder.Builder.lane2data
```

````

````{py:attribute} map_lanes
:canonical: mosstool.map.builder.builder.Builder.map_lanes
:value: >
   None

```{autodoc2-docstring} mosstool.map.builder.builder.Builder.map_lanes
```

````

````{py:attribute} no_left_walk
:canonical: mosstool.map.builder.builder.Builder.no_left_walk
:value: >
   'set(...)'

```{autodoc2-docstring} mosstool.map.builder.builder.Builder.no_left_walk
```

````

````{py:attribute} no_right_walk
:canonical: mosstool.map.builder.builder.Builder.no_right_walk
:value: >
   'set(...)'

```{autodoc2-docstring} mosstool.map.builder.builder.Builder.no_right_walk
```

````

````{py:method} _connect_lane_group(in_lanes: list[shapely.geometry.LineString], out_lanes: list[shapely.geometry.LineString], lane_turn: mosstool.map._map_util.const.mapv2.LaneTurn, lane_type: mosstool.map._map_util.const.mapv2.LaneType, junc_id: int, in_walk_type: typing.Union[typing.Literal[in_way], typing.Literal[out_way], typing.Literal[]] = '', out_walk_type: typing.Union[typing.Literal[in_way], typing.Literal[out_way], typing.Literal[]] = '') -> list[shapely.geometry.LineString]
:canonical: mosstool.map.builder.builder.Builder._connect_lane_group

```{autodoc2-docstring} mosstool.map.builder.builder.Builder._connect_lane_group
```

````

````{py:method} _delete_lane(lane_id: int, delete_road: bool = False) -> None
:canonical: mosstool.map.builder.builder.Builder._delete_lane

```{autodoc2-docstring} mosstool.map.builder.builder.Builder._delete_lane
```

````

````{py:method} _reset_lane_uids(orig_lane_uids: list[int], new_lane_uids: list[int]) -> None
:canonical: mosstool.map.builder.builder.Builder._reset_lane_uids

```{autodoc2-docstring} mosstool.map.builder.builder.Builder._reset_lane_uids
```

````

````{py:method} draw_junction(jid: int, save_path: str, trim_length: float = 50)
:canonical: mosstool.map.builder.builder.Builder.draw_junction

```{autodoc2-docstring} mosstool.map.builder.builder.Builder.draw_junction
```

````

````{py:method} draw_walk_junction(jid: int, save_path: str, trim_length: float = 50)
:canonical: mosstool.map.builder.builder.Builder.draw_walk_junction

```{autodoc2-docstring} mosstool.map.builder.builder.Builder.draw_walk_junction
```

````

````{py:method} _classify()
:canonical: mosstool.map.builder.builder.Builder._classify

```{autodoc2-docstring} mosstool.map.builder.builder.Builder._classify
```

````

````{py:method} _classify_main_way_ids()
:canonical: mosstool.map.builder.builder.Builder._classify_main_way_ids

```{autodoc2-docstring} mosstool.map.builder.builder.Builder._classify_main_way_ids
```

````

````{py:method} _expand_roads(wids: list[int], junc_type, junc_id: int, way_type: typing.Union[typing.Literal[main], typing.Literal[around], typing.Literal[right], typing.Literal[left], typing.Literal[]] = '')
:canonical: mosstool.map.builder.builder.Builder._expand_roads

```{autodoc2-docstring} mosstool.map.builder.builder.Builder._expand_roads
```

````

````{py:method} _expand_remain_roads()
:canonical: mosstool.map.builder.builder.Builder._expand_remain_roads

```{autodoc2-docstring} mosstool.map.builder.builder.Builder._expand_remain_roads
```

````

````{py:method} _add_sidewalk(wid, lane: shapely.geometry.LineString, other_lane: shapely.geometry.LineString, walk_type: typing.Union[typing.Literal[left], typing.Literal[right]], walk_lane_end_type: typing.Union[typing.Literal[start], typing.Literal[end]])
:canonical: mosstool.map.builder.builder.Builder._add_sidewalk

```{autodoc2-docstring} mosstool.map.builder.builder.Builder._add_sidewalk
```

````

````{py:method} _create_junction_walk_pairs(in_way_groups: tuple[list[tuple[mosstool.map._map_util.const.np.ndarray, list[int]]]], out_way_groups: tuple[list[tuple[mosstool.map._map_util.const.np.ndarray, list[int]]]], has_main_group_wids: set, junc_center: tuple[float, float])
:canonical: mosstool.map.builder.builder.Builder._create_junction_walk_pairs

```{autodoc2-docstring} mosstool.map.builder.builder.Builder._create_junction_walk_pairs
```

````

````{py:method} _create_junction_for_1_n()
:canonical: mosstool.map.builder.builder.Builder._create_junction_for_1_n

```{autodoc2-docstring} mosstool.map.builder.builder.Builder._create_junction_for_1_n
```

````

````{py:method} _create_junction_for_n_n()
:canonical: mosstool.map.builder.builder.Builder._create_junction_for_n_n

```{autodoc2-docstring} mosstool.map.builder.builder.Builder._create_junction_for_n_n
```

````

````{py:method} _create_walking_lanes()
:canonical: mosstool.map.builder.builder.Builder._create_walking_lanes

```{autodoc2-docstring} mosstool.map.builder.builder.Builder._create_walking_lanes
```

````

````{py:method} _add_junc_lane_overlaps()
:canonical: mosstool.map.builder.builder.Builder._add_junc_lane_overlaps

```{autodoc2-docstring} mosstool.map.builder.builder.Builder._add_junc_lane_overlaps
```

````

````{py:method} _add_driving_lane_group()
:canonical: mosstool.map.builder.builder.Builder._add_driving_lane_group

```{autodoc2-docstring} mosstool.map.builder.builder.Builder._add_driving_lane_group
```

````

````{py:method} _add_traffic_light()
:canonical: mosstool.map.builder.builder.Builder._add_traffic_light

```{autodoc2-docstring} mosstool.map.builder.builder.Builder._add_traffic_light
```

````

````{py:method} _add_public_transport() -> set[int]
:canonical: mosstool.map.builder.builder.Builder._add_public_transport

```{autodoc2-docstring} mosstool.map.builder.builder.Builder._add_public_transport
```

````

````{py:method} _add_reuse_aoi()
:canonical: mosstool.map.builder.builder.Builder._add_reuse_aoi

```{autodoc2-docstring} mosstool.map.builder.builder.Builder._add_reuse_aoi
```

````

````{py:method} _add_input_aoi()
:canonical: mosstool.map.builder.builder.Builder._add_input_aoi

```{autodoc2-docstring} mosstool.map.builder.builder.Builder._add_input_aoi
```

````

````{py:method} _add_all_aoi()
:canonical: mosstool.map.builder.builder.Builder._add_all_aoi

```{autodoc2-docstring} mosstool.map.builder.builder.Builder._add_all_aoi
```

````

````{py:method} write2json(topo_path: str, output_path: str)
:canonical: mosstool.map.builder.builder.Builder.write2json

```{autodoc2-docstring} mosstool.map.builder.builder.Builder.write2json
```

````

````{py:method} _post_process()
:canonical: mosstool.map.builder.builder.Builder._post_process

```{autodoc2-docstring} mosstool.map.builder.builder.Builder._post_process
```

````

````{py:method} get_output_map(name: str)
:canonical: mosstool.map.builder.builder.Builder.get_output_map

```{autodoc2-docstring} mosstool.map.builder.builder.Builder.get_output_map
```

````

````{py:method} write2db(coll: pymongo.collection.Collection, name: str)
:canonical: mosstool.map.builder.builder.Builder.write2db

```{autodoc2-docstring} mosstool.map.builder.builder.Builder.write2db
```

````

````{py:method} build(name: str)
:canonical: mosstool.map.builder.builder.Builder.build

```{autodoc2-docstring} mosstool.map.builder.builder.Builder.build
```

````

`````
