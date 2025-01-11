# {py:mod}`mosstool.map.osm.roadnet`

```{py:module} mosstool.map.osm.roadnet
```

```{autodoc2-docstring} mosstool.map.osm.roadnet
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RoadNet <mosstool.map.osm.roadnet.RoadNet>`
  - ```{autodoc2-docstring} mosstool.map.osm.roadnet.RoadNet
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <mosstool.map.osm.roadnet.__all__>`
  - ```{autodoc2-docstring} mosstool.map.osm.roadnet.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: mosstool.map.osm.roadnet.__all__
:value: >
   ['RoadNet']

```{autodoc2-docstring} mosstool.map.osm.roadnet.__all__
```

````

`````{py:class} RoadNet(proj_str: typing.Optional[str] = None, max_longitude: typing.Optional[float] = None, min_longitude: typing.Optional[float] = None, max_latitude: typing.Optional[float] = None, min_latitude: typing.Optional[float] = None, wikipedia_name: typing.Optional[str] = None, proxies: typing.Optional[dict[str, str]] = None)
:canonical: mosstool.map.osm.roadnet.RoadNet

```{autodoc2-docstring} mosstool.map.osm.roadnet.RoadNet
```

```{rubric} Initialization
```

```{autodoc2-docstring} mosstool.map.osm.roadnet.RoadNet.__init__
```

````{py:property} default_way_settings
:canonical: mosstool.map.osm.roadnet.RoadNet.default_way_settings

```{autodoc2-docstring} mosstool.map.osm.roadnet.RoadNet.default_way_settings
```

````

````{py:method} _download_osm()
:canonical: mosstool.map.osm.roadnet.RoadNet._download_osm

```{autodoc2-docstring} mosstool.map.osm.roadnet.RoadNet._download_osm
```

````

````{py:method} _way_length(way)
:canonical: mosstool.map.osm.roadnet.RoadNet._way_length

```{autodoc2-docstring} mosstool.map.osm.roadnet.RoadNet._way_length
```

````

````{py:method} _way_coords_xy(way)
:canonical: mosstool.map.osm.roadnet.RoadNet._way_coords_xy

```{autodoc2-docstring} mosstool.map.osm.roadnet.RoadNet._way_coords_xy
```

````

````{py:method} _way_coords_lonlat(way)
:canonical: mosstool.map.osm.roadnet.RoadNet._way_coords_lonlat

```{autodoc2-docstring} mosstool.map.osm.roadnet.RoadNet._way_coords_lonlat
```

````

````{py:method} _update_node_ways()
:canonical: mosstool.map.osm.roadnet.RoadNet._update_node_ways

```{autodoc2-docstring} mosstool.map.osm.roadnet.RoadNet._update_node_ways
```

````

````{py:method} _assert_ways()
:canonical: mosstool.map.osm.roadnet.RoadNet._assert_ways

```{autodoc2-docstring} mosstool.map.osm.roadnet.RoadNet._assert_ways
```

````

````{py:method} dump_as_geojson(path: str)
:canonical: mosstool.map.osm.roadnet.RoadNet.dump_as_geojson

```{autodoc2-docstring} mosstool.map.osm.roadnet.RoadNet.dump_as_geojson
```

````

````{py:method} to_topo()
:canonical: mosstool.map.osm.roadnet.RoadNet.to_topo

```{autodoc2-docstring} mosstool.map.osm.roadnet.RoadNet.to_topo
```

````

````{py:method} _get_osm(osm_data_cache: typing.Optional[list[dict]] = None)
:canonical: mosstool.map.osm.roadnet.RoadNet._get_osm

```{autodoc2-docstring} mosstool.map.osm.roadnet.RoadNet._get_osm
```

````

````{py:method} _remove_redundant_ways()
:canonical: mosstool.map.osm.roadnet.RoadNet._remove_redundant_ways

```{autodoc2-docstring} mosstool.map.osm.roadnet.RoadNet._remove_redundant_ways
```

````

````{py:method} _remove_invalid_ways()
:canonical: mosstool.map.osm.roadnet.RoadNet._remove_invalid_ways

```{autodoc2-docstring} mosstool.map.osm.roadnet.RoadNet._remove_invalid_ways
```

````

````{py:method} _remove_simple_joints()
:canonical: mosstool.map.osm.roadnet.RoadNet._remove_simple_joints

```{autodoc2-docstring} mosstool.map.osm.roadnet.RoadNet._remove_simple_joints
```

````

````{py:method} _init_junctions()
:canonical: mosstool.map.osm.roadnet.RoadNet._init_junctions

```{autodoc2-docstring} mosstool.map.osm.roadnet.RoadNet._init_junctions
```

````

````{py:method} _make_all_one_way()
:canonical: mosstool.map.osm.roadnet.RoadNet._make_all_one_way

```{autodoc2-docstring} mosstool.map.osm.roadnet.RoadNet._make_all_one_way
```

````

````{py:method} _remove_out_of_roadnet()
:canonical: mosstool.map.osm.roadnet.RoadNet._remove_out_of_roadnet

```{autodoc2-docstring} mosstool.map.osm.roadnet.RoadNet._remove_out_of_roadnet
```

````

````{py:method} _merge_junction_by_motif()
:canonical: mosstool.map.osm.roadnet.RoadNet._merge_junction_by_motif

```{autodoc2-docstring} mosstool.map.osm.roadnet.RoadNet._merge_junction_by_motif
```

````

````{py:method} _clean_topo()
:canonical: mosstool.map.osm.roadnet.RoadNet._clean_topo

```{autodoc2-docstring} mosstool.map.osm.roadnet.RoadNet._clean_topo
```

````

````{py:method} create_road_net(output_path: typing.Optional[str] = None, osm_data_cache: typing.Optional[list[dict]] = None, osm_cache_check: bool = False)
:canonical: mosstool.map.osm.roadnet.RoadNet.create_road_net

```{autodoc2-docstring} mosstool.map.osm.roadnet.RoadNet.create_road_net
```

````

`````
