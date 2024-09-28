from mosstool.trip.generator import (CalibratedTemplateGenerator,
                                     GaussianTemplateGenerator, PositionMode,
                                     ProbabilisticTemplateGenerator,
                                     RandomGenerator, UniformTemplateGenerator)
from mosstool.trip.route import RoutingClient, pre_route
from mosstool.type import Map, Person, Persons, TripMode
from mosstool.util.format_converter import pb2json

pg = ProbabilisticTemplateGenerator(
    headway_values=[1.5, 2, 2.5], headway_probabilities=[1, 1, 1]
)
ug = UniformTemplateGenerator(
    max_speed_min=90,
    max_speed_max=110,
)
gg = GaussianTemplateGenerator(
    max_speed_mean=50,
    max_speed_std=10,
)
cg = CalibratedTemplateGenerator()
_generators = [pg, ug, cg, gg]
_generator_names = ["probabilistic", "uniform", "calibrated", "gaussian"]


async def main():
    with open("data/temp/map.pb", "rb") as f:
        m = Map()
        m.ParseFromString(f.read())
    for i, (_g, _g_name) in enumerate(zip(_generators, _generator_names)):
        rg = RandomGenerator(
            m,
            [PositionMode.LANE, PositionMode.LANE],
            TripMode.TRIP_MODE_DRIVE_ONLY,
            template_func=_g.template_generator,
        )
        persons = rg.uniform(
            num=100,
            first_departure_time_range=(8 * 3600, 9 * 3600),
            schedule_interval_range=(5 * 60, 10 * 60),
            start_id=100 * i,
        )

        # pre-route
        # run: ./routing -map data/temp/map.pb
        client = RoutingClient("http://localhost:52101")
        ok_persons = []
        for p in persons:
            p = await pre_route(client, p)
            if len(p.schedules) > 0 and len(p.schedules[0].trips) > 0:
                ok_persons.append(p)
        print("final length: ", len(ok_persons))
        pb = Persons(persons=ok_persons)
        with open(f"data/temp/{_g_name}_persons.json", "w") as f:
            f.write(pb2json(pb))
        with open(f"data/temp/{_g_name}_persons.pb", "wb") as f:
            f.write(pb.SerializeToString())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
