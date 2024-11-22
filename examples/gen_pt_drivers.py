from mosstool.trip.generator import (TripGenerator,
                                     default_bus_template_generator)
from mosstool.type import Map, Persons

# map from `./examples/add_pt_to_map.py` and `./examples/add_pt_post_process.py`
with open("data/temp/srt.map_with_pt.pb", "rb") as f:
    m = Map()
    m.ParseFromString(f.read())
tg = TripGenerator(m=m)
bus_drivers = tg.generate_public_transport_drivers(
    template_func=default_bus_template_generator
)
persons_output_path = "data/temp/bus_drivers.pb"
pb = Persons(persons=bus_drivers)
with open(persons_output_path, "wb") as f:
    f.write(pb.SerializeToString())
