from mosstool.map.osm import PointOfInterest

# Beijing
bbox = {
    "max_lat": 40.20836867760951,
    "min_lat": 39.69203625898142,
    "min_lon": 116.12174658204533,
    "max_lon": 116.65141646506795,
}
pois = PointOfInterest(
    max_latitude=bbox["max_lat"],
    min_latitude=bbox["min_lat"],
    max_longitude=bbox["max_lon"],
    min_longitude=bbox["min_lon"],
)
path = "data/temp/pois.geojson"
pois = pois.create_pois(output_path=path)
