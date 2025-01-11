# {py:mod}`mosstool.map.public_transport.public_transport_post`

```{py:module} mosstool.map.public_transport.public_transport_post
```

```{autodoc2-docstring} mosstool.map.public_transport.public_transport_post
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_fill_public_lines <mosstool.map.public_transport.public_transport_post._fill_public_lines>`
  - ```{autodoc2-docstring} mosstool.map.public_transport.public_transport_post._fill_public_lines
    :summary:
    ```
* - {py:obj}`_get_taz_cost_unit <mosstool.map.public_transport.public_transport_post._get_taz_cost_unit>`
  - ```{autodoc2-docstring} mosstool.map.public_transport.public_transport_post._get_taz_cost_unit
    :summary:
    ```
* - {py:obj}`_post_compute <mosstool.map.public_transport.public_transport_post._post_compute>`
  - ```{autodoc2-docstring} mosstool.map.public_transport.public_transport_post._post_compute
    :summary:
    ```
* - {py:obj}`public_transport_process <mosstool.map.public_transport.public_transport_post.public_transport_process>`
  - ```{autodoc2-docstring} mosstool.map.public_transport.public_transport_post.public_transport_process
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <mosstool.map.public_transport.public_transport_post.__all__>`
  - ```{autodoc2-docstring} mosstool.map.public_transport.public_transport_post.__all__
    :summary:
    ```
* - {py:obj}`ETA_FACTOR <mosstool.map.public_transport.public_transport_post.ETA_FACTOR>`
  - ```{autodoc2-docstring} mosstool.map.public_transport.public_transport_post.ETA_FACTOR
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: mosstool.map.public_transport.public_transport_post.__all__
:value: >
   ['public_transport_process']

```{autodoc2-docstring} mosstool.map.public_transport.public_transport_post.__all__
```

````

````{py:data} ETA_FACTOR
:canonical: mosstool.map.public_transport.public_transport_post.ETA_FACTOR
:value: >
   5

```{autodoc2-docstring} mosstool.map.public_transport.public_transport_post.ETA_FACTOR
```

````

````{py:function} _fill_public_lines(m: dict, server_address: str)
:canonical: mosstool.map.public_transport.public_transport_post._fill_public_lines
:async:

```{autodoc2-docstring} mosstool.map.public_transport.public_transport_post._fill_public_lines
```
````

````{py:function} _get_taz_cost_unit(arg)
:canonical: mosstool.map.public_transport.public_transport_post._get_taz_cost_unit

```{autodoc2-docstring} mosstool.map.public_transport.public_transport_post._get_taz_cost_unit
```
````

````{py:function} _post_compute(m: dict, workers: int, taz_length: float, max_chunk_size: int)
:canonical: mosstool.map.public_transport.public_transport_post._post_compute

```{autodoc2-docstring} mosstool.map.public_transport.public_transport_post._post_compute
```
````

````{py:function} public_transport_process(m: dict, server_address: str, taz_length: float = 1500, workers: int = cpu_count(), multiprocessing_chunk_size: int = 500)
:canonical: mosstool.map.public_transport.public_transport_post.public_transport_process

```{autodoc2-docstring} mosstool.map.public_transport.public_transport_post.public_transport_process
```
````
