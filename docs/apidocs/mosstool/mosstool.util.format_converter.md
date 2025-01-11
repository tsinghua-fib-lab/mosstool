# {py:mod}`mosstool.util.format_converter`

```{py:module} mosstool.util.format_converter
```

```{autodoc2-docstring} mosstool.util.format_converter
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`pb2json <mosstool.util.format_converter.pb2json>`
  - ```{autodoc2-docstring} mosstool.util.format_converter.pb2json
    :summary:
    ```
* - {py:obj}`pb2dict <mosstool.util.format_converter.pb2dict>`
  - ```{autodoc2-docstring} mosstool.util.format_converter.pb2dict
    :summary:
    ```
* - {py:obj}`pb2coll <mosstool.util.format_converter.pb2coll>`
  - ```{autodoc2-docstring} mosstool.util.format_converter.pb2coll
    :summary:
    ```
* - {py:obj}`json2pb <mosstool.util.format_converter.json2pb>`
  - ```{autodoc2-docstring} mosstool.util.format_converter.json2pb
    :summary:
    ```
* - {py:obj}`dict2pb <mosstool.util.format_converter.dict2pb>`
  - ```{autodoc2-docstring} mosstool.util.format_converter.dict2pb
    :summary:
    ```
* - {py:obj}`coll2pb <mosstool.util.format_converter.coll2pb>`
  - ```{autodoc2-docstring} mosstool.util.format_converter.coll2pb
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <mosstool.util.format_converter.__all__>`
  - ```{autodoc2-docstring} mosstool.util.format_converter.__all__
    :summary:
    ```
* - {py:obj}`T <mosstool.util.format_converter.T>`
  - ```{autodoc2-docstring} mosstool.util.format_converter.T
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: mosstool.util.format_converter.__all__
:value: >
   ['pb2json', 'pb2dict', 'pb2coll', 'json2pb', 'dict2pb', 'coll2pb']

```{autodoc2-docstring} mosstool.util.format_converter.__all__
```

````

````{py:function} pb2json(pb: google.protobuf.message.Message)
:canonical: mosstool.util.format_converter.pb2json

```{autodoc2-docstring} mosstool.util.format_converter.pb2json
```
````

````{py:function} pb2dict(pb: google.protobuf.message.Message)
:canonical: mosstool.util.format_converter.pb2dict

```{autodoc2-docstring} mosstool.util.format_converter.pb2dict
```
````

````{py:function} pb2coll(pb: google.protobuf.message.Message, coll: pymongo.collection.Collection, insert_chunk_size: int = 0, drop: bool = False)
:canonical: mosstool.util.format_converter.pb2coll

```{autodoc2-docstring} mosstool.util.format_converter.pb2coll
```
````

````{py:data} T
:canonical: mosstool.util.format_converter.T
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} mosstool.util.format_converter.T
```

````

````{py:function} json2pb(json: str, pb: mosstool.util.format_converter.T) -> mosstool.util.format_converter.T
:canonical: mosstool.util.format_converter.json2pb

```{autodoc2-docstring} mosstool.util.format_converter.json2pb
```
````

````{py:function} dict2pb(d: dict, pb: mosstool.util.format_converter.T) -> mosstool.util.format_converter.T
:canonical: mosstool.util.format_converter.dict2pb

```{autodoc2-docstring} mosstool.util.format_converter.dict2pb
```
````

````{py:function} coll2pb(coll: pymongo.collection.Collection, pb: mosstool.util.format_converter.T) -> mosstool.util.format_converter.T
:canonical: mosstool.util.format_converter.coll2pb

```{autodoc2-docstring} mosstool.util.format_converter.coll2pb
```
````
