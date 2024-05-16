from mosstool.map.vis import VisMap
from mosstool.trip.vis import VisTrip
from mosstool.type import Map, Persons
from mosstool.util.format_converter import json2pb

with open("data/temp/map.pb", "rb") as f:
    m = Map()
    m.ParseFromString(f.read())
with open("data/temp/persons.json", "r") as f:
    json = f.read()
    persons = Persons()
    persons = json2pb(json, persons)

vis_map = VisMap(m)
vis_trip = VisTrip(vis_map, list(persons.persons))
deck = vis_trip.visualize_home()
deck.to_html("data/temp/trip_home.html")
deck = vis_trip.visualize_od()
deck.to_html("data/temp/trip_od.html")
deck = vis_trip.visualize_route()
deck.to_html("data/temp/trip_route.html")
