# {py:mod}`mosstool.trip.generator.gravity`

```{py:module} mosstool.trip.generator.gravity
```

```{autodoc2-docstring} mosstool.trip.generator.gravity
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GravityGenerator <mosstool.trip.generator.gravity.GravityGenerator>`
  - ```{autodoc2-docstring} mosstool.trip.generator.gravity.GravityGenerator
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <mosstool.trip.generator.gravity.__all__>`
  - ```{autodoc2-docstring} mosstool.trip.generator.gravity.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: mosstool.trip.generator.gravity.__all__
:value: >
   ['GravityGenerator']

```{autodoc2-docstring} mosstool.trip.generator.gravity.__all__
```

````

`````{py:class} GravityGenerator(Lambda: float, Alpha: float, Beta: float, Gamma: float)
:canonical: mosstool.trip.generator.gravity.GravityGenerator

```{autodoc2-docstring} mosstool.trip.generator.gravity.GravityGenerator
```

```{rubric} Initialization
```

```{autodoc2-docstring} mosstool.trip.generator.gravity.GravityGenerator.__init__
```

````{py:method} load_area(area: geopandas.GeoDataFrame)
:canonical: mosstool.trip.generator.gravity.GravityGenerator.load_area

```{autodoc2-docstring} mosstool.trip.generator.gravity.GravityGenerator.load_area
```

````

````{py:method} _get_one_point()
:canonical: mosstool.trip.generator.gravity.GravityGenerator._get_one_point

```{autodoc2-docstring} mosstool.trip.generator.gravity.GravityGenerator._get_one_point
```

````

````{py:method} _calculate_utm_epsg(longitude: float, latitude: float)
:canonical: mosstool.trip.generator.gravity.GravityGenerator._calculate_utm_epsg

```{autodoc2-docstring} mosstool.trip.generator.gravity.GravityGenerator._calculate_utm_epsg
```

````

````{py:method} cal_distance()
:canonical: mosstool.trip.generator.gravity.GravityGenerator.cal_distance

```{autodoc2-docstring} mosstool.trip.generator.gravity.GravityGenerator.cal_distance
```

````

````{py:method} generate(pops)
:canonical: mosstool.trip.generator.gravity.GravityGenerator.generate

```{autodoc2-docstring} mosstool.trip.generator.gravity.GravityGenerator.generate
```

````

`````
