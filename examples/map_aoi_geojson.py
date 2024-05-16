from mosstool.map.osm import Building
# Beijing
bbox = {
    "max_lat": 40.20836867760951,
    "min_lat": 39.69203625898142,
    "min_lon": 116.12174658204533,
    "max_lon": 116.65141646506795,
}
building = Building(
    proj_str="+proj=tmerc +lat_0=39.90611 +lon_0=116.3911",
    max_latitude=bbox["max_lat"],
    min_latitude=bbox["min_lat"],
    max_longitude=bbox["max_lon"],
    min_longitude=bbox["min_lon"],
)
path = "data/temp/aois.geojson"
aois = building.create_building(output_path=path)
