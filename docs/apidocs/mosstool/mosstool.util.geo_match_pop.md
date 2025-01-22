# {py:mod}`mosstool.util.geo_match_pop`

```{py:module} mosstool.util.geo_match_pop
```

```{autodoc2-docstring} mosstool.util.geo_match_pop
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`geo_coords <mosstool.util.geo_match_pop.geo_coords>`
  - ```{autodoc2-docstring} mosstool.util.geo_match_pop.geo_coords
    :summary:
    ```
* - {py:obj}`_gps_distance <mosstool.util.geo_match_pop._gps_distance>`
  - ```{autodoc2-docstring} mosstool.util.geo_match_pop._gps_distance
    :summary:
    ```
* - {py:obj}`_get_idx_range_in_bbox <mosstool.util.geo_match_pop._get_idx_range_in_bbox>`
  - ```{autodoc2-docstring} mosstool.util.geo_match_pop._get_idx_range_in_bbox
    :summary:
    ```
* - {py:obj}`_get_pixel_info <mosstool.util.geo_match_pop._get_pixel_info>`
  - ```{autodoc2-docstring} mosstool.util.geo_match_pop._get_pixel_info
    :summary:
    ```
* - {py:obj}`_upsample_pixels_unit <mosstool.util.geo_match_pop._upsample_pixels_unit>`
  - ```{autodoc2-docstring} mosstool.util.geo_match_pop._upsample_pixels_unit
    :summary:
    ```
* - {py:obj}`_upsample_pixels_idiot_unit <mosstool.util.geo_match_pop._upsample_pixels_idiot_unit>`
  - ```{autodoc2-docstring} mosstool.util.geo_match_pop._upsample_pixels_idiot_unit
    :summary:
    ```
* - {py:obj}`_get_poly_pop_unit <mosstool.util.geo_match_pop._get_poly_pop_unit>`
  - ```{autodoc2-docstring} mosstool.util.geo_match_pop._get_poly_pop_unit
    :summary:
    ```
* - {py:obj}`_get_geo_pop <mosstool.util.geo_match_pop._get_geo_pop>`
  - ```{autodoc2-docstring} mosstool.util.geo_match_pop._get_geo_pop
    :summary:
    ```
* - {py:obj}`geo2pop <mosstool.util.geo_match_pop.geo2pop>`
  - ```{autodoc2-docstring} mosstool.util.geo_match_pop.geo2pop
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <mosstool.util.geo_match_pop.__all__>`
  - ```{autodoc2-docstring} mosstool.util.geo_match_pop.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: mosstool.util.geo_match_pop.__all__
:value: >
   ['geo2pop']

```{autodoc2-docstring} mosstool.util.geo_match_pop.__all__
```

````

````{py:function} geo_coords(geo)
:canonical: mosstool.util.geo_match_pop.geo_coords

```{autodoc2-docstring} mosstool.util.geo_match_pop.geo_coords
```
````

````{py:function} _gps_distance(LON1: typing.Union[float, tuple[float, float]], LAT1: typing.Union[float, tuple[float, float]], LON2: typing.Optional[float] = None, LAT2: typing.Optional[float] = None)
:canonical: mosstool.util.geo_match_pop._gps_distance

```{autodoc2-docstring} mosstool.util.geo_match_pop._gps_distance
```
````

````{py:function} _get_idx_range_in_bbox(min_x: float, max_x: float, min_y: float, max_y: float, xy_bound: tuple[float, float, float, float], mode: typing.Union[typing.Literal[loose], typing.Literal[tight]] = 'loose')
:canonical: mosstool.util.geo_match_pop._get_idx_range_in_bbox

```{autodoc2-docstring} mosstool.util.geo_match_pop._get_idx_range_in_bbox
```
````

````{py:function} _get_pixel_info(band, x_left: float, y_upper: float, x_step: float, y_step: float, bbox: tuple[float, float, float, float], padding: int = 20)
:canonical: mosstool.util.geo_match_pop._get_pixel_info

```{autodoc2-docstring} mosstool.util.geo_match_pop._get_pixel_info
```
````

````{py:function} _upsample_pixels_unit(partial_args: tuple[list[typing.Any]], arg: tuple[tuple[int, float, float], tuple[tuple[int, int], tuple[shapely.geometry.Point, int]]])
:canonical: mosstool.util.geo_match_pop._upsample_pixels_unit

```{autodoc2-docstring} mosstool.util.geo_match_pop._upsample_pixels_unit
```
````

````{py:function} _upsample_pixels_idiot_unit(arg)
:canonical: mosstool.util.geo_match_pop._upsample_pixels_idiot_unit

```{autodoc2-docstring} mosstool.util.geo_match_pop._upsample_pixels_idiot_unit
```
````

````{py:function} _get_poly_pop_unit(partial_args: tuple[dict[tuple[int, int], tuple[shapely.geometry.Point, int]], float, float, float, float, float, float], geo_item: tuple[int, tuple[typing.Any, tuple[float, float, float, float]]])
:canonical: mosstool.util.geo_match_pop._get_poly_pop_unit

```{autodoc2-docstring} mosstool.util.geo_match_pop._get_poly_pop_unit
```
````

````{py:function} _get_geo_pop(geos, pixel_idx2point_pop: dict[tuple[int, int], tuple[shapely.geometry.Point, int]], workers: int, x_left: float, y_upper: float, x_step: float, y_step: float, xy_gps_scale2: float, pixel_area: float, max_chunk_size: int, enable_tqdm: bool)
:canonical: mosstool.util.geo_match_pop._get_geo_pop

```{autodoc2-docstring} mosstool.util.geo_match_pop._get_geo_pop
```
````

````{py:function} geo2pop(geo_data: typing.Union[geopandas.geodataframe.GeoDataFrame, geojson.FeatureCollection], pop_tif_path: str, enable_tqdm: bool = False, upsample_factor: int = 4, pop_in_aoi_factor: float = 0.7, multiprocessing_chunk_size: int = 500) -> typing.Union[geopandas.geodataframe.GeoDataFrame, geojson.FeatureCollection]
:canonical: mosstool.util.geo_match_pop.geo2pop

```{autodoc2-docstring} mosstool.util.geo_match_pop.geo2pop
```
````
