# {py:mod}`mosstool.map.osm._motif`

```{py:module} mosstool.map.osm._motif
```

```{autodoc2-docstring} mosstool.map.osm._motif
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`close_nodes <mosstool.map.osm._motif.close_nodes>`
  - ```{autodoc2-docstring} mosstool.map.osm._motif.close_nodes
    :summary:
    ```
* - {py:obj}`suc_is_close_by_other_way <mosstool.map.osm._motif.suc_is_close_by_other_way>`
  - ```{autodoc2-docstring} mosstool.map.osm._motif.suc_is_close_by_other_way
    :summary:
    ```
* - {py:obj}`motif_H <mosstool.map.osm._motif.motif_H>`
  - ```{autodoc2-docstring} mosstool.map.osm._motif.motif_H
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <mosstool.map.osm._motif.__all__>`
  - ```{autodoc2-docstring} mosstool.map.osm._motif.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: mosstool.map.osm._motif.__all__
:value: >
   ['close_nodes', 'suc_is_close_by_other_way', 'motif_H']

```{autodoc2-docstring} mosstool.map.osm._motif.__all__
```

````

````{py:function} close_nodes(G: networkx.DiGraph) -> set[frozenset[int]]
:canonical: mosstool.map.osm._motif.close_nodes

```{autodoc2-docstring} mosstool.map.osm._motif.close_nodes
```
````

````{py:function} suc_is_close_by_other_way(G: networkx.DiGraph) -> set[frozenset[int]]
:canonical: mosstool.map.osm._motif.suc_is_close_by_other_way

```{autodoc2-docstring} mosstool.map.osm._motif.suc_is_close_by_other_way
```
````

````{py:function} motif_H(G: networkx.DiGraph) -> set[frozenset[int]]
:canonical: mosstool.map.osm._motif.motif_H

```{autodoc2-docstring} mosstool.map.osm._motif.motif_H
```
````
