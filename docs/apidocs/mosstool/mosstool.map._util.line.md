# {py:mod}`mosstool.map._util.line`

```{py:module} mosstool.map._util.line
```

```{autodoc2-docstring} mosstool.map._util.line
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`clip_line <mosstool.map._util.line.clip_line>`
  - ```{autodoc2-docstring} mosstool.map._util.line.clip_line
    :summary:
    ```
* - {py:obj}`has_multiple_turns <mosstool.map._util.line.has_multiple_turns>`
  - ```{autodoc2-docstring} mosstool.map._util.line.has_multiple_turns
    :summary:
    ```
* - {py:obj}`line_extend <mosstool.map._util.line.line_extend>`
  - ```{autodoc2-docstring} mosstool.map._util.line.line_extend
    :summary:
    ```
* - {py:obj}`line_max_curvature <mosstool.map._util.line.line_max_curvature>`
  - ```{autodoc2-docstring} mosstool.map._util.line.line_max_curvature
    :summary:
    ```
* - {py:obj}`connect_line_string <mosstool.map._util.line.connect_line_string>`
  - ```{autodoc2-docstring} mosstool.map._util.line.connect_line_string
    :summary:
    ```
* - {py:obj}`connect_line_string_bezier_4_t_point <mosstool.map._util.line.connect_line_string_bezier_4_t_point>`
  - ```{autodoc2-docstring} mosstool.map._util.line.connect_line_string_bezier_4_t_point
    :summary:
    ```
* - {py:obj}`connect_line_string_straight <mosstool.map._util.line.connect_line_string_straight>`
  - ```{autodoc2-docstring} mosstool.map._util.line.connect_line_string_straight
    :summary:
    ```
* - {py:obj}`offset_lane <mosstool.map._util.line.offset_lane>`
  - ```{autodoc2-docstring} mosstool.map._util.line.offset_lane
    :summary:
    ```
* - {py:obj}`align_line <mosstool.map._util.line.align_line>`
  - ```{autodoc2-docstring} mosstool.map._util.line.align_line
    :summary:
    ```
* - {py:obj}`merge_near_xy_points <mosstool.map._util.line.merge_near_xy_points>`
  - ```{autodoc2-docstring} mosstool.map._util.line.merge_near_xy_points
    :summary:
    ```
* - {py:obj}`connect_split_lines <mosstool.map._util.line.connect_split_lines>`
  - ```{autodoc2-docstring} mosstool.map._util.line.connect_split_lines
    :summary:
    ```
* - {py:obj}`merge_line_start_end <mosstool.map._util.line.merge_line_start_end>`
  - ```{autodoc2-docstring} mosstool.map._util.line.merge_line_start_end
    :summary:
    ```
* - {py:obj}`get_start_vector <mosstool.map._util.line.get_start_vector>`
  - ```{autodoc2-docstring} mosstool.map._util.line.get_start_vector
    :summary:
    ```
* - {py:obj}`get_end_vector <mosstool.map._util.line.get_end_vector>`
  - ```{autodoc2-docstring} mosstool.map._util.line.get_end_vector
    :summary:
    ```
* - {py:obj}`get_line_angle <mosstool.map._util.line.get_line_angle>`
  - ```{autodoc2-docstring} mosstool.map._util.line.get_line_angle
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <mosstool.map._util.line.__all__>`
  - ```{autodoc2-docstring} mosstool.map._util.line.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: mosstool.map._util.line.__all__
:value: >
   ['clip_line', 'line_extend', 'connect_line_string', 'line_max_curvature', 'offset_lane', 'align_line...

```{autodoc2-docstring} mosstool.map._util.line.__all__
```

````

````{py:function} clip_line(line: shapely.geometry.LineString, p1: shapely.geometry.Point, p2: shapely.geometry.Point) -> shapely.geometry.LineString
:canonical: mosstool.map._util.line.clip_line

```{autodoc2-docstring} mosstool.map._util.line.clip_line
```
````

````{py:function} has_multiple_turns(line: shapely.geometry.LineString) -> bool
:canonical: mosstool.map._util.line.has_multiple_turns

```{autodoc2-docstring} mosstool.map._util.line.has_multiple_turns
```
````

````{py:function} line_extend(line: shapely.geometry.LineString, extend_length: float)
:canonical: mosstool.map._util.line.line_extend

```{autodoc2-docstring} mosstool.map._util.line.line_extend
```
````

````{py:function} line_max_curvature(line: shapely.geometry.LineString)
:canonical: mosstool.map._util.line.line_max_curvature

```{autodoc2-docstring} mosstool.map._util.line.line_max_curvature
```
````

````{py:function} connect_line_string(line1: shapely.geometry.LineString, line2: shapely.geometry.LineString) -> shapely.geometry.LineString
:canonical: mosstool.map._util.line.connect_line_string

```{autodoc2-docstring} mosstool.map._util.line.connect_line_string
```
````

````{py:function} connect_line_string_bezier_4_t_point(strength: float, line1: shapely.geometry.LineString, line2: shapely.geometry.LineString) -> shapely.geometry.LineString
:canonical: mosstool.map._util.line.connect_line_string_bezier_4_t_point

```{autodoc2-docstring} mosstool.map._util.line.connect_line_string_bezier_4_t_point
```
````

````{py:function} connect_line_string_straight(line1: shapely.geometry.LineString, line2: shapely.geometry.LineString) -> shapely.geometry.LineString
:canonical: mosstool.map._util.line.connect_line_string_straight

```{autodoc2-docstring} mosstool.map._util.line.connect_line_string_straight
```
````

````{py:function} offset_lane(line: shapely.geometry.LineString, distance: float) -> shapely.geometry.LineString
:canonical: mosstool.map._util.line.offset_lane

```{autodoc2-docstring} mosstool.map._util.line.offset_lane
```
````

````{py:function} align_line(line1: shapely.geometry.LineString, line2: shapely.geometry.LineString) -> shapely.geometry.LineString
:canonical: mosstool.map._util.line.align_line

```{autodoc2-docstring} mosstool.map._util.line.align_line
```
````

````{py:function} merge_near_xy_points(orig_points: list[tuple[float, float]], merge_gate: float = 100) -> list[tuple[float, float]]
:canonical: mosstool.map._util.line.merge_near_xy_points

```{autodoc2-docstring} mosstool.map._util.line.merge_near_xy_points
```
````

````{py:function} connect_split_lines(lines: list[shapely.geometry.LineString], start_point: typing.Optional[shapely.geometry.Point] = None, max_line_length: float = 10000) -> list
:canonical: mosstool.map._util.line.connect_split_lines

```{autodoc2-docstring} mosstool.map._util.line.connect_split_lines
```
````

````{py:function} merge_line_start_end(line_start: shapely.geometry.LineString, line_end: shapely.geometry.LineString) -> shapely.geometry.LineString
:canonical: mosstool.map._util.line.merge_line_start_end

```{autodoc2-docstring} mosstool.map._util.line.merge_line_start_end
```
````

````{py:function} get_start_vector(line: shapely.geometry.LineString)
:canonical: mosstool.map._util.line.get_start_vector

```{autodoc2-docstring} mosstool.map._util.line.get_start_vector
```
````

````{py:function} get_end_vector(line: shapely.geometry.LineString)
:canonical: mosstool.map._util.line.get_end_vector

```{autodoc2-docstring} mosstool.map._util.line.get_end_vector
```
````

````{py:function} get_line_angle(line: shapely.geometry.LineString)
:canonical: mosstool.map._util.line.get_line_angle

```{autodoc2-docstring} mosstool.map._util.line.get_line_angle
```
````
