# {py:mod}`mosstool.map._util.bezier`

```{py:module} mosstool.map._util.bezier
```

```{autodoc2-docstring} mosstool.map._util.bezier
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Bezier <mosstool.map._util.bezier.Bezier>`
  - ```{autodoc2-docstring} mosstool.map._util.bezier.Bezier
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <mosstool.map._util.bezier.__all__>`
  - ```{autodoc2-docstring} mosstool.map._util.bezier.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: mosstool.map._util.bezier.__all__
:value: >
   ['Bezier']

```{autodoc2-docstring} mosstool.map._util.bezier.__all__
```

````

`````{py:class} Bezier
:canonical: mosstool.map._util.bezier.Bezier

```{autodoc2-docstring} mosstool.map._util.bezier.Bezier
```

````{py:method} TwoPoints(t: float, P1: numpy.ndarray, P2: numpy.ndarray)
:canonical: mosstool.map._util.bezier.Bezier.TwoPoints
:staticmethod:

```{autodoc2-docstring} mosstool.map._util.bezier.Bezier.TwoPoints
```

````

````{py:method} Points(t: float, points: list[numpy.ndarray]) -> list[numpy.ndarray]
:canonical: mosstool.map._util.bezier.Bezier.Points
:staticmethod:

```{autodoc2-docstring} mosstool.map._util.bezier.Bezier.Points
```

````

````{py:method} Point(t: float, points: list[numpy.ndarray])
:canonical: mosstool.map._util.bezier.Bezier.Point
:staticmethod:

```{autodoc2-docstring} mosstool.map._util.bezier.Bezier.Point
```

````

````{py:method} Curve(t_values: typing.Union[collections.abc.Sequence[float], numpy.ndarray], points: list[numpy.ndarray]) -> numpy.ndarray
:canonical: mosstool.map._util.bezier.Bezier.Curve
:staticmethod:

```{autodoc2-docstring} mosstool.map._util.bezier.Bezier.Curve
```

````

`````
