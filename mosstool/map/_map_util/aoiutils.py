import logging
from math import ceil
from multiprocessing import Pool

import shapely.ops as ops
from shapely.affinity import scale
from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon

from .._map_util.const import *

__all__ = [
    "generate_aoi_poi",
    "generate_sumo_aoi_poi",
    "geo_coords",
]


def geo_coords(geo):
    if isinstance(geo, Polygon):
        return list(geo.exterior.coords)
    elif isinstance(geo, MultiPolygon):
        all_coords = []
        for p_geo in geo.geoms:
            all_coords.extend(geo_coords(p_geo))
        return all_coords
    else:
        return list(geo.coords)


def _fix_polygon(input_poly: Polygon):
    if input_poly.is_valid:
        return input_poly
    else:
        geo = input_poly.buffer(0)
        if isinstance(geo, Polygon) and geo.is_valid:
            polygon = geo
        elif isinstance(geo, MultiPolygon):
            candidate_poly = None
            for poly in geo.geoms:
                if poly.is_valid:
                    candidate_poly = poly
                    break
            if candidate_poly is not None:
                polygon = candidate_poly
            else:
                polygon = MultiPoint(
                    [pt for g in geo.geoms for pt in geo_coords(g)]
                ).convex_hull
        else:
            polygon = MultiPoint([pt for pt in geo_coords(input_poly)]).convex_hull
        return polygon


def _fix_aois_poly(input_aois: list) -> list:
    aois = []
    for aoi in input_aois:
        coords = aoi["coords"]
        geo = _fix_polygon(Polygon(coords))
        if geo.is_valid and geo:
            aoi["coords"] = [c for c in geo_coords(geo)]
            aois.append(aoi)
    return aois


def _connect_aoi_unit1(poly):
    """
    Find the children aoi of each poly in the merged geometry
    """
    global aois_small
    x, y = geo_coords(poly.centroid)[0]
    length = poly.length
    children = []
    added_aoi_small = []
    for aoi in aois_small:
        x2, y2 = aoi["point"]
        if SQRT2 * (abs(x - x2) + abs(y - y2)) < length:
            if aoi["geo"].intersection(poly).area > COVER_GATE * aoi["area"]:
                children.append(aoi)
                added_aoi_small.append(aoi["id"])
    if children:
        return poly, children, added_aoi_small
    else:
        return


def _connect_aoi_unit2(arg):
    """
    Remove the redundant shapes caused by amplification of the connected aoi (not exceeding the closed convex hull formed by all the vertices of the original small aoi)
    """
    poly, children = arg
    if len(children) == 1:
        return [
            children[0]
        ]  # There is only 1 child, indicating that it is not connected to other aoi and restores the original shape

    poly_inner = MultiPoint(
        [pt for aoi in children for pt in geo_coords(aoi["geo"])]
    ).convex_hull
    poly = poly.intersection(poly_inner)
    if not isinstance(poly, Polygon):
        # Because the connections between the various parts of the poly are too thin and exceed the range of the internal closed convex hull, they will be disconnected after intersection.
        logging.info("process break due to intersection with convex hull ")
        t = [scale(p, xfact=SCALE, yfact=SCALE, origin="centroid") for p in poly.geoms]
        poly = ops.unary_union(t).intersection(
            poly_inner
        )  # Amplify and fuse the disconnected parts after the intersection, and then intersect with the closed convex hull again
    if not isinstance(poly, Polygon):
        return children
    else:
        population = 0
        inner_poi = []
        inner_poi_catg = []
        osm_tags = []
        for a in children:
            external = a["external"]
            population += external["population"]
            inner_poi.extend(external["inner_poi"])
            inner_poi_catg.extend(external["inner_poi_catg"])
            osm_tags.extend(external["osm_tags"])
        return {
            "id": [a["id"] for a in children][0],
            "geo": poly,
            "external": {
                "population": population,
                "osm_tags": osm_tags,
                "inner_poi": inner_poi,
                "inner_poi_catg": inner_poi_catg,
            },
            "point": geo_coords(poly.centroid)[
                0
            ],  # For subsequent processing needs, calculate the merged geometric center
            "length": poly.length,  # perimeter
        }


def _find_aoi_parent_unit(i_aoi):
    """
    Find out when aoi is contained by other aoi
    """
    i, aoi = i_aoi
    aoi["duplicate"] = set()
    aoi["has_parent"] = False
    x, y = aoi["point"]
    geo = aoi["geo"]
    area = aoi["area"]
    for j, aoi2 in enumerate(aois_to_merge):
        if j != i:
            if (
                aoi2["area"] > area
            ):  # Reduce the amount of calculation and avoid two aoi whose overlap ratio exceeds their respective area thresholds from including each other.
                x2, y2 = aoi2[
                    "point"
                ]  # Reduce the amount of calculation and only search between adjacent aoi
                # If the large aoi contains a small aoi, the distance between two points in the two aoi cannot exceed half of the circumference of the large aoi
                # Use 1 norm for the distance and divide it by sqrt(2) to estimate the lower bound and reduce the amount of calculation.
                if (
                    SQRT2 * (abs(x - x2) + abs(y - y2)) < aoi2["length"]
                ):  # (abs(x - x2) + abs(y - y2)) / sqrt(2) < aoi2['length'] / 2
                    # if geo.covered_by(aoi2['geo']): # Using strict inclusion judgment is not robust enough, use overlap ratio judgment instead
                    if geo.intersection(aoi2["geo"]).area > COVER_GATE * area:
                        aoi["has_parent"] = True
                        break
            elif (
                aoi2["area"] == area
            ):  # There are a small number of duplicate aoi (same geo)
                if geo.intersection(aoi2["geo"]).area > COVER_GATE * area:
                    assert geo.equals(aoi2["geo"])
                    aoi["duplicate"].add(
                        j
                    )  # Don't break at this time, make sure to completely record all aoi that overlap with it.
    return aoi


def _merge_aoi(input_aois: list, merge_aoi: bool = False, workers: int = 32):
    """
    Integrate the contained small aoi into the large aoi
    """
    # Precompute geometric properties
    aois = []
    for aoi in input_aois:
        coords = aoi["coords"]
        geo = Polygon(coords)
        if not isinstance(geo, Polygon) or not geo:
            logging.warning(f"Invalid polygon {aoi['id']}")
            continue
        aoi["geo"] = geo
        aoi["point"] = geo_coords(geo.centroid)[0]  # Geometric center
        aoi["length"] = geo.length  # Perimeter
        aoi["area"] = geo.area  # area
        aois.append(aoi)
    if not merge_aoi:
        return aois
    logging.info("Merging aoi")
    # Parallelization: Find aoi that are not contained by any aoi
    global aois_to_merge
    aois_to_merge = aois
    logging.info(f"Multiprocessing for merging aoi")

    aois = [(i, a) for i, a in enumerate(aois)]
    with Pool(processes=workers) as pool:
        aois = pool.map(
            _find_aoi_parent_unit,
            aois,
            chunksize=min(ceil(len(aois) / workers), 1000),
        )

    aois_ancestor = []
    added_duplicates = set()
    for i, aoi in enumerate(aois):
        if not aoi["has_parent"]:
            dup = aoi["duplicate"]
            if not dup:
                aois_ancestor.append(aoi)
            else:  # Except for the duplicate aoi, no other aoi contains it, pick one of these duplicate aoi to become the ancestor
                for j in dup:
                    assert i in aois[j]["duplicate"]
                if i not in added_duplicates:
                    aois_ancestor.append(aoi)
                    added_duplicates = added_duplicates | dup | {i}
    return aois_ancestor


# Connect small and nearby aoi groups
def _connect_aoi(input_aois: list, merge_aoi: bool = False, workers: int = 32):
    """
    Connecting piles of small houses into a residential area
    """
    # Precompute geometric properties
    aois = []
    for aoi in input_aois:
        coords = aoi["coords"]
        geo = Polygon(coords)
        if not isinstance(geo, Polygon) or not geo:
            logging.warning(f"Invalid polygon {aoi['id']}")
            continue
        aoi["geo"] = geo
        aoi["point"] = geo_coords(geo.centroid)[0]  # Geometric center
        aoi["length"] = geo.length  # Perimeter
        aoi["area"] = geo.area  # area
        aois.append(aoi)
    if not merge_aoi:
        return aois
    logging.info("Connecting aoi")
    # Take the aoi with smaller area
    global aois_small
    aois_small = [aoi for aoi in aois if aoi["area"] < SMALL_GATE]
    aois_other = [aoi for aoi in aois if aoi["area"] >= SMALL_GATE]
    logging.info("aois_small:", len(aois_small))
    logging.info("aois_other:", len(aois_other))

    # Magnify to a certain proportion and merge intersecting graphics
    polys = [aoi["geo"] for aoi in aois_small]
    polys_scale = [scale(p, xfact=SCALE, yfact=SCALE, origin="centroid") for p in polys]
    geo_scale_connect = ops.unary_union(polys_scale)
    args = list(geo_scale_connect.geoms)  # type: ignore
    with Pool(processes=workers) as pool:
        results = pool.map(
            _connect_aoi_unit1,
            args,
            chunksize=max(min(ceil(len(args) / workers), 200), 1),
        )
    results = [x for x in results if x]
    clusters = [x[:2] for x in results]
    added_aoi_small = sum([x[-1] for x in results], [])

    # Process the aoi in the small aoi that are not covered by the enlarged graphics of any small aoi (due to ill-shape): keep as is
    added_aoi_small = set(added_aoi_small)
    aois_other += [aoi for aoi in aois_small if aoi["id"] not in added_aoi_small]
    with Pool(processes=workers) as pool:
        results = pool.map(
            _connect_aoi_unit2,
            clusters,
            chunksize=max(min(ceil(len(clusters) / workers), 200), 1),
        )
    aois_connect = [x for x in results if isinstance(x, dict)]
    aois_other += sum([x for x in results if isinstance(x, list)], [])
    return aois_other + aois_connect


def _match_poi_unit(poi):
    """
    Match the poi to the aoi that directly covers it or is the closest and less than the threshold
    """

    global aois_to_match_poi
    global iso_poi_set
    iso_poi_set = ISOLATED_POI_CATG
    x, y = poi["coords"][0]
    point = Point(x, y)
    poi["point"] = point

    for c in poi["external"]["catg"].split("|"):
        if c in iso_poi_set:
            return (poi, THIS_IS_ISOLATE_POI)

    # parents = []
    # covered = False
    parent = None
    neighbors = []
    for i, aoi in enumerate(
        aois_to_match_poi
    ):  # Only interested in aoi that may cover or project the poi, so first filter based on distance, and then calculate accurately
        x2, y2 = aoi["point"]
        if (
            SQRT2 * (abs(x - x2) + abs(y - y2)) < aoi["length"] + DOUBLE_DIS_GATE
        ):  # DIS_norm1 / sqrt(2) < perimeter / 2 + DIS_GATE
            if point.covered_by(aoi["geo"]):
                parent = i
                break
            # elif not covered:
            else:  # If not covered, try to project to adjacent aoi
                dis = point.distance(aoi["geo"])
                if dis < DIS_GATE:
                    neighbors.append((i, dis))
    if parent:
        return (poi, parent, THIS_IS_COVERED_POI)
    elif neighbors:
        return (poi, min(neighbors, key=lambda x: x[1])[0], THIS_IS_PROJECTED_POI)
    else:
        return (poi, THIS_IS_ISOLATE_POI)


def _process_stops(stops):
    def point_extend(center: Point, length: float):
        half_side = length / 2
        bottom_left = (center.x - half_side, center.y - half_side)
        top_left = (center.x - half_side, center.y + half_side)
        top_right = (center.x + half_side, center.y + half_side)
        bottom_right = (center.x + half_side, center.y - half_side)
        return Polygon([bottom_left, top_left, top_right, bottom_right, bottom_left])

    for stop in stops:
        coords = stop["coords"]
        extend_length = DEFAULT_STATION_LENGTH[stop["external"]["station_type"]]
        if len(coords) == 1:
            center = Point(coords[0])
            geo = point_extend(center, extend_length)
        else:
            geo = Polygon(coords)
        geo = _fix_polygon(geo)
        stop["geo"] = geo
        stop["point"] = geo_coords(geo.centroid)[0]  # Geometric center
        stop["length"] = geo.length  # Perimeter
        stop["area"] = geo.area  # area
    return stops


def _match_poi_to_aoi(aois, pois, workers):
    """
    poi matches aoi:
    Directly dependent items covered by existing aoi go in
    Those that are not covered, such as bus stations, become new aoi; non-bus stations are subordinate to the nearest aoi within a certain distance; those that are too far away from the existing aoi become new aoi
    """
    global aois_to_match_poi
    aois_to_match_poi = aois  # global variable!

    # Calculate whether each poi is covered. If not, calculate the projection onto the nearest aoi.
    logging.info(f"Multiprocessing for matching poi({len(pois)}) to aoi({len(aois)})")
    results = []
    for i in range(0, len(pois), MAX_BATCH_SIZE):
        pois_batch = pois[i : i + MAX_BATCH_SIZE]
        with Pool(processes=workers) as pool:
            results += pool.map(
                _match_poi_unit,
                pois_batch,
                chunksize=max(
                    min(ceil(len(pois_batch) / workers), 1000),
                    1,
                ),
            )
    pois_covered = []
    pois_projected = []
    pois_isolate = []
    pid2catg = {p["id"]: p["external"]["catg"] for p in pois}
    for x in results:
        if x[-1] == THIS_IS_COVERED_POI:
            poi, aoi_idx = x[:2]
            pois_covered.append(poi)
            aoi = aois[aoi_idx]
            aoi["external"]["inner_poi"].append(poi["id"])
            aoi["external"]["inner_poi_catg"].append(pid2catg[poi["id"]])
        elif x[-1] == THIS_IS_PROJECTED_POI:
            poi, aoi_idx = x[:2]
            pois_projected.append(poi)
            aoi = aois[aoi_idx]
            aoi["external"]["inner_poi"].append(poi["id"])
            aoi["external"]["inner_poi_catg"].append(pid2catg[poi["id"]])
        else:
            pois_isolate.append(x[0])

    return aois, pois_isolate, pois


def _post_compute_aoi_poi(aois, pois_isolate):
    # Update coordinates
    for a in aois:
        coords = geo_coords(a["geo"])
        a["coords"] = coords
    # poi becomes Aoi independently
    for p in pois_isolate:
        aois.append(
            {
                "id": p["id"],
                "coords": p["coords"],
                "external": {
                    "population": 0,
                    "osm_tags": [],
                    "inner_poi": [p["id"]],
                    "inner_poi_catg": [p["external"]["catg"]],
                },
            }
        )
    return aois


def generate_aoi_poi(input_aois, input_pois, input_stops, workers: int = 32):
    merge_aoi = False
    input_aois = _fix_aois_poly(input_aois)
    # Process covered AOI
    input_aois = _merge_aoi(input_aois, merge_aoi, workers)
    # Connect small and nearby aoi groups
    input_aois = _connect_aoi(input_aois, merge_aoi, workers)
    # Process and join poi: belong to aoi or become an independent aoi
    aois_add_poi, pois_isolate, pois_output = _match_poi_to_aoi(
        input_aois, input_pois, workers
    )
    # Convert format to output
    aois_output = _post_compute_aoi_poi(aois_add_poi, pois_isolate)
    stops_output = _process_stops(input_stops)
    return (aois_output, stops_output, pois_output)


def generate_sumo_aoi_poi(
    input_aois, input_pois, input_stops, workers: int = 32, merge_aoi: bool = False
):
    input_aois = _fix_aois_poly(input_aois)
    input_aois = _merge_aoi(input_aois, merge_aoi, workers)
    input_aois = _connect_aoi(input_aois, merge_aoi, workers)
    aois_add_poi, pois_isolate, pois_output = _match_poi_to_aoi(
        input_aois, input_pois, workers
    )
    aois_output = _post_compute_aoi_poi(aois_add_poi, pois_isolate)
    stops_output = _process_stops(input_stops)
    return (aois_output, stops_output, pois_output)
