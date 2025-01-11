# {py:mod}`mosstool.trip.generator.template`

```{py:module} mosstool.trip.generator.template
```

```{autodoc2-docstring} mosstool.trip.generator.template
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ProbabilisticTemplateGenerator <mosstool.trip.generator.template.ProbabilisticTemplateGenerator>`
  - ```{autodoc2-docstring} mosstool.trip.generator.template.ProbabilisticTemplateGenerator
    :summary:
    ```
* - {py:obj}`GaussianTemplateGenerator <mosstool.trip.generator.template.GaussianTemplateGenerator>`
  - ```{autodoc2-docstring} mosstool.trip.generator.template.GaussianTemplateGenerator
    :summary:
    ```
* - {py:obj}`UniformTemplateGenerator <mosstool.trip.generator.template.UniformTemplateGenerator>`
  - ```{autodoc2-docstring} mosstool.trip.generator.template.UniformTemplateGenerator
    :summary:
    ```
* - {py:obj}`CalibratedTemplateGenerator <mosstool.trip.generator.template.CalibratedTemplateGenerator>`
  - ```{autodoc2-docstring} mosstool.trip.generator.template.CalibratedTemplateGenerator
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`default_person_template_generator <mosstool.trip.generator.template.default_person_template_generator>`
  - ```{autodoc2-docstring} mosstool.trip.generator.template.default_person_template_generator
    :summary:
    ```
* - {py:obj}`default_bus_template_generator <mosstool.trip.generator.template.default_bus_template_generator>`
  - ```{autodoc2-docstring} mosstool.trip.generator.template.default_bus_template_generator
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <mosstool.trip.generator.template.__all__>`
  - ```{autodoc2-docstring} mosstool.trip.generator.template.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: mosstool.trip.generator.template.__all__
:value: >
   ['default_person_template_generator', 'default_bus_template_generator', 'ProbabilisticTemplateGenera...

```{autodoc2-docstring} mosstool.trip.generator.template.__all__
```

````

````{py:function} default_person_template_generator() -> pycityproto.city.person.v2.person_pb2.Person
:canonical: mosstool.trip.generator.template.default_person_template_generator

```{autodoc2-docstring} mosstool.trip.generator.template.default_person_template_generator
```
````

````{py:function} default_bus_template_generator() -> pycityproto.city.person.v2.person_pb2.Person
:canonical: mosstool.trip.generator.template.default_bus_template_generator

```{autodoc2-docstring} mosstool.trip.generator.template.default_bus_template_generator
```
````

`````{py:class} ProbabilisticTemplateGenerator(max_speed_values: typing.Optional[list[float]] = None, max_speed_probabilities: typing.Optional[list[float]] = None, max_acceleration_values: typing.Optional[list[float]] = None, max_acceleration_probabilities: typing.Optional[list[float]] = None, max_braking_acceleration_values: typing.Optional[list[float]] = None, max_braking_acceleration_probabilities: typing.Optional[list[float]] = None, usual_braking_acceleration_values: typing.Optional[list[float]] = None, usual_braking_acceleration_probabilities: typing.Optional[list[float]] = None, headway_values: typing.Optional[list[float]] = None, headway_probabilities: typing.Optional[list[float]] = None, min_gap_values: typing.Optional[list[float]] = None, min_gap_probabilities: typing.Optional[list[float]] = None, seed: int = 0, template: typing.Optional[pycityproto.city.person.v2.person_pb2.Person] = None)
:canonical: mosstool.trip.generator.template.ProbabilisticTemplateGenerator

```{autodoc2-docstring} mosstool.trip.generator.template.ProbabilisticTemplateGenerator
```

```{rubric} Initialization
```

```{autodoc2-docstring} mosstool.trip.generator.template.ProbabilisticTemplateGenerator.__init__
```

````{py:method} template_generator() -> pycityproto.city.person.v2.person_pb2.Person
:canonical: mosstool.trip.generator.template.ProbabilisticTemplateGenerator.template_generator

```{autodoc2-docstring} mosstool.trip.generator.template.ProbabilisticTemplateGenerator.template_generator
```

````

`````

`````{py:class} GaussianTemplateGenerator(max_speed_mean: typing.Optional[float] = None, max_speed_std: typing.Optional[float] = None, max_acceleration_mean: typing.Optional[float] = None, max_acceleration_std: typing.Optional[float] = None, max_braking_acceleration_mean: typing.Optional[float] = None, max_braking_acceleration_std: typing.Optional[float] = None, usual_braking_acceleration_mean: typing.Optional[float] = None, usual_braking_acceleration_std: typing.Optional[float] = None, headway_mean: typing.Optional[float] = None, headway_std: typing.Optional[float] = None, min_gap_mean: typing.Optional[float] = None, min_gap_std: typing.Optional[float] = None, seed: int = 0, template: typing.Optional[pycityproto.city.person.v2.person_pb2.Person] = None)
:canonical: mosstool.trip.generator.template.GaussianTemplateGenerator

```{autodoc2-docstring} mosstool.trip.generator.template.GaussianTemplateGenerator
```

```{rubric} Initialization
```

```{autodoc2-docstring} mosstool.trip.generator.template.GaussianTemplateGenerator.__init__
```

````{py:method} template_generator() -> pycityproto.city.person.v2.person_pb2.Person
:canonical: mosstool.trip.generator.template.GaussianTemplateGenerator.template_generator

```{autodoc2-docstring} mosstool.trip.generator.template.GaussianTemplateGenerator.template_generator
```

````

`````

`````{py:class} UniformTemplateGenerator(max_speed_min: typing.Optional[float] = None, max_speed_max: typing.Optional[float] = None, max_acceleration_min: typing.Optional[float] = None, max_acceleration_max: typing.Optional[float] = None, max_braking_acceleration_min: typing.Optional[float] = None, max_braking_acceleration_max: typing.Optional[float] = None, usual_braking_acceleration_min: typing.Optional[float] = None, usual_braking_acceleration_max: typing.Optional[float] = None, headway_min: typing.Optional[float] = None, headway_max: typing.Optional[float] = None, min_gap_min: typing.Optional[float] = None, min_gap_max: typing.Optional[float] = None, seed: int = 0, template: typing.Optional[pycityproto.city.person.v2.person_pb2.Person] = None)
:canonical: mosstool.trip.generator.template.UniformTemplateGenerator

```{autodoc2-docstring} mosstool.trip.generator.template.UniformTemplateGenerator
```

```{rubric} Initialization
```

```{autodoc2-docstring} mosstool.trip.generator.template.UniformTemplateGenerator.__init__
```

````{py:method} template_generator() -> pycityproto.city.person.v2.person_pb2.Person
:canonical: mosstool.trip.generator.template.UniformTemplateGenerator.template_generator

```{autodoc2-docstring} mosstool.trip.generator.template.UniformTemplateGenerator.template_generator
```

````

`````

`````{py:class} CalibratedTemplateGenerator(seed: int = 0, template: typing.Optional[pycityproto.city.person.v2.person_pb2.Person] = None)
:canonical: mosstool.trip.generator.template.CalibratedTemplateGenerator

```{autodoc2-docstring} mosstool.trip.generator.template.CalibratedTemplateGenerator
```

```{rubric} Initialization
```

```{autodoc2-docstring} mosstool.trip.generator.template.CalibratedTemplateGenerator.__init__
```

````{py:method} template_generator() -> pycityproto.city.person.v2.person_pb2.Person
:canonical: mosstool.trip.generator.template.CalibratedTemplateGenerator.template_generator

```{autodoc2-docstring} mosstool.trip.generator.template.CalibratedTemplateGenerator.template_generator
```

````

`````
