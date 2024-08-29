import geopandas as gpd

from mosstool.trip.generator import AigcGenerator
from mosstool.trip.generator.generate_from_od import TripGenerator
from mosstool.trip.route import RoutingClient, pre_route
from mosstool.type import Map, Persons

YOUR_ACCESS_TOKEN = "1"  # for World_Imagery, applied from ArcGIS (https://www.arcgis.com/home/item.html?id=10df2279f9684e4a9f6a7f08febac2a9)


async def main():
    # Initialize the generator
    aigc_generator = AigcGenerator()
    aigc_generator.set_satetoken(YOUR_ACCESS_TOKEN)
    area = gpd.read_file("data/gravitygenerator/Beijing-shp/beijing.shp")
    aigc_generator.load_area(area)

    # Generate the OD matrix
    od_matrix = aigc_generator.generate()

    # map of Beijing (with only AOI)
    with open("data/gravitygenerator/beijing_map.pb", "rb") as f:
        m = Map()
        m.ParseFromString(f.read())
    tg = TripGenerator(
        m=m,
    )
    od_persons = tg.generate_persons(
        od_matrix=od_matrix,
        departure_time_curve=[
            1,
            1,
            1,
            1,
            2,
            3,
            4,
            5,
            6,
            5,
            4,
            0.5,
            1,
            1,
            0.5,
            1,
            1,
            0.5,
            1,
            1,
            0.5,
            1,
            1,
            0.5,
        ],
        areas=area,
        agent_num=10000,
        seed=0,
    )
    pb = Persons(persons=od_persons)
    with open("data/temp/beijing_OD_person.pb", "wb") as f:
        f.write(pb.SerializeToString())

    # # The generated trip of the person is not guaranteed to be reachable in the map. Preroute is required.
    # # pre-route
    # # run: ./routing -map data/gravitygenerator/beijing_map.pb
    # client = RoutingClient("http://localhost:52101")
    # ok_persons = []
    # for p in od_persons:
    #     p = await pre_route(client, p)
    #     if len(p.schedules) > 0 and len(p.schedules[0].trips) > 0:
    #         ok_persons.append(p)
    # print(len(ok_persons))
    # pb = Persons(persons=ok_persons)
    # with open("data/temp/beijing_OD_ok_person.pb", "wb") as f:
    #     f.write(pb.SerializeToString())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
