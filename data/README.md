# File Descriptions in Directory Data

## Overview of Directory Structure

- `data/`
  - `geojson/`: original `GeoJSON` input files
    - `net.geojson`: input road net file for `./examples/build_map.py`.
    - `split_box.geojson`: geometric blocks used for `examples/aplit_map.py` to split the map.
    - `wuhan_pt.json`: public transport lines and stations in Wuhan, China.
  - `gravitygenerator/`
    - `Beijing-shp/`
      - `beijing.shp`: geometric O-D (origin-destination) blocks in Beijing.
    - `beijing_map.pb`: contains only `AOIs` to generate `Persons` with O-D matrix.
    - `worldpop.npy`: population of each geometric block in `beijing.shp`.
  - `map/`
    - `m.pb`: example `Map` for `examples/aplit_map.py` to split.
  - `sumo/`
    - `shenzhen.net.xml`: `SUMO` map file for `examples/sumo_map_convert.py` .
    - `poly.xml`: Trained machine learning model.
    - `trips.xml`: `SUMO` trip file for `examples/sumo_route_convert.py` .
    - `trafficlight.xml`: `SUMO` traffic-light file for `examples/sumo_map_convert.py` .
    - `additional.xml`: `SUMO` additional file for `examples/sumo_route_convert.py`, providing `busStop`, `chargingStation`, `parkingArea` stop information.
  - `temp/`: temporary output files
