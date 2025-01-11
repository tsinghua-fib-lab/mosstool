# {py:mod}`mosstool.map.public_transport.get_bus`

```{py:module} mosstool.map.public_transport.get_bus
```

```{autodoc2-docstring} mosstool.map.public_transport.get_bus
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AmapBus <mosstool.map.public_transport.get_bus.AmapBus>`
  - ```{autodoc2-docstring} mosstool.map.public_transport.get_bus.AmapBus
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_get_headers <mosstool.map.public_transport.get_bus._get_headers>`
  - ```{autodoc2-docstring} mosstool.map.public_transport.get_bus._get_headers
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <mosstool.map.public_transport.get_bus.__all__>`
  - ```{autodoc2-docstring} mosstool.map.public_transport.get_bus.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: mosstool.map.public_transport.get_bus.__all__
:value: >
   ['AmapBus']

```{autodoc2-docstring} mosstool.map.public_transport.get_bus.__all__
```

````

````{py:function} _get_headers(referer_url)
:canonical: mosstool.map.public_transport.get_bus._get_headers

```{autodoc2-docstring} mosstool.map.public_transport.get_bus._get_headers
```
````

`````{py:class} AmapBus(city_name_en_us: str, city_name_zh_cn: str, bus_heads: str, amap_ak: str)
:canonical: mosstool.map.public_transport.get_bus.AmapBus

```{autodoc2-docstring} mosstool.map.public_transport.get_bus.AmapBus
```

```{rubric} Initialization
```

```{autodoc2-docstring} mosstool.map.public_transport.get_bus.AmapBus.__init__
```

````{py:method} _fetch_raw_data()
:canonical: mosstool.map.public_transport.get_bus.AmapBus._fetch_raw_data

```{autodoc2-docstring} mosstool.map.public_transport.get_bus.AmapBus._fetch_raw_data
```

````

````{py:method} _fetch_amap_positions()
:canonical: mosstool.map.public_transport.get_bus.AmapBus._fetch_amap_positions

```{autodoc2-docstring} mosstool.map.public_transport.get_bus.AmapBus._fetch_amap_positions
```

````

````{py:method} _fetch_amap_lines()
:canonical: mosstool.map.public_transport.get_bus.AmapBus._fetch_amap_lines

```{autodoc2-docstring} mosstool.map.public_transport.get_bus.AmapBus._fetch_amap_lines
```

````

````{py:method} _process_amap()
:canonical: mosstool.map.public_transport.get_bus.AmapBus._process_amap

```{autodoc2-docstring} mosstool.map.public_transport.get_bus.AmapBus._process_amap
```

````

````{py:method} get_output_data()
:canonical: mosstool.map.public_transport.get_bus.AmapBus.get_output_data

```{autodoc2-docstring} mosstool.map.public_transport.get_bus.AmapBus.get_output_data
```

````

`````
