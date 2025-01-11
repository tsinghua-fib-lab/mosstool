# {py:mod}`mosstool.trip.route.client`

```{py:module} mosstool.trip.route.client
```

```{autodoc2-docstring} mosstool.trip.route.client
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RoutingClient <mosstool.trip.route.client.RoutingClient>`
  - ```{autodoc2-docstring} mosstool.trip.route.client.RoutingClient
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_create_aio_channel <mosstool.trip.route.client._create_aio_channel>`
  - ```{autodoc2-docstring} mosstool.trip.route.client._create_aio_channel
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__all__ <mosstool.trip.route.client.__all__>`
  - ```{autodoc2-docstring} mosstool.trip.route.client.__all__
    :summary:
    ```
````

### API

````{py:data} __all__
:canonical: mosstool.trip.route.client.__all__
:value: >
   ['RoutingClient']

```{autodoc2-docstring} mosstool.trip.route.client.__all__
```

````

````{py:function} _create_aio_channel(server_address: str, secure: bool = False) -> grpc.aio.Channel
:canonical: mosstool.trip.route.client._create_aio_channel

```{autodoc2-docstring} mosstool.trip.route.client._create_aio_channel
```
````

`````{py:class} RoutingClient(server_address: str, secure: bool = False)
:canonical: mosstool.trip.route.client.RoutingClient

```{autodoc2-docstring} mosstool.trip.route.client.RoutingClient
```

```{rubric} Initialization
```

```{autodoc2-docstring} mosstool.trip.route.client.RoutingClient.__init__
```

````{py:method} GetRoute(req: mosstool.type.GetRouteRequest) -> mosstool.type.GetRouteResponse
:canonical: mosstool.trip.route.client.RoutingClient.GetRoute
:async:

```{autodoc2-docstring} mosstool.trip.route.client.RoutingClient.GetRoute
```

````

`````
