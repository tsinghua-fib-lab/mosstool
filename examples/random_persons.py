from mosstool.trip.generator import (PositionMode, RandomGenerator,
                                     default_person_template_generator)
from mosstool.trip.route import RoutingClient, pre_route
from mosstool.type import Map, Person, Persons, TripMode
from mosstool.util.format_converter import pb2json


async def main():
    with open("data/temp/map.pb", "rb") as f:
        m = Map()
        m.ParseFromString(f.read())
    rg = RandomGenerator(
        m,
        [PositionMode.LANE, PositionMode.LANE],
        TripMode.TRIP_MODE_DRIVE_ONLY,
        template_func=default_person_template_generator,
    )
    persons = rg.uniform(
        num=100,
        first_departure_time_range=(8 * 3600, 9 * 3600),
        schedule_interval_range=(5 * 60, 10 * 60),
        start_id=0,
    )
    print(persons)

    # pre-route
    # run: ./routing -map data/temp/map.pb
    client = RoutingClient("http://localhost:52101")
    ok_persons = []
    for p in persons:
        p = await pre_route(client, p)
        if len(p.schedules) > 0 and len(p.schedules[0].trips) > 0:
            ok_persons.append(p)
    print(ok_persons)
    print("final length: ", len(ok_persons))
    pb = Persons(persons=ok_persons)
    with open("data/temp/persons.json", "w") as f:
        f.write(pb2json(pb))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
