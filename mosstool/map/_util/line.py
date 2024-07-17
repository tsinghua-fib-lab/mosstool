from functools import partial
from math import atan2
from typing import List, Optional, Tuple, cast

import numpy as np
from scipy.spatial import KDTree
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, nearest_points, snap, split, substring
from shapely.strtree import STRtree

from .angle import abs_delta_angle
from .bezier import Bezier

__all__ = [
    "clip_line",
    "line_extend",
    "connect_line_string",
    "line_max_curvature",
    "offset_lane",
    "align_line",
    "merge_near_xy_points",
    "connect_split_lines",
    "get_start_vector",
    "get_end_vector",
]


def clip_line(line: LineString, p1: Point, p2: Point) -> LineString:
    """
    Clip line between p1 and p2

    Args:
    - line (LineString): LineString
    - p1 (Point):  start point of the clipped line
    - p2 (Point):  end point of the clipped line

    Returns:
    - LineString: clipped LineString
    """

    has_z_coords = bool(any(len(c) > 2 for c in line.coords))
    line_xy = LineString([c[:2] for c in line.coords])
    p1_xy = Point(p1.x, p1.y)
    p2_xy = Point(p2.x, p2.y)
    p1_xy = nearest_points(line_xy, p1_xy)[0]
    geom = snap(line_xy, p1_xy, 1)
    parts = split(geom, p1_xy).geoms
    if len(parts) == 1:
        fp = parts[0]
    else:
        fp = parts[0] if parts[0].distance(p2_xy) < parts[1].distance(p2_xy) else parts[1]  # type: ignore
    p2_xy = nearest_points(fp, p2_xy)[0]
    geom = snap(fp, p2_xy, 0.1)
    parts = split(geom, p2_xy).geoms
    if len(parts) == 1:
        clipped_line_xy = parts[0]
    else:
        clipped_line_xy = parts[0] if parts[0].distance(p1_xy) < parts[1].distance(p1_xy) else parts[1]  # type: ignore
    clipped_line_xy = cast(LineString, clipped_line_xy)
    if not has_z_coords:
        return clipped_line_xy
    else:
        z_start = line.interpolate(line.project(p1)).z
        z_end = line.interpolate(line.project(p2)).z
        coords_z = np.linspace(z_start, z_end, len(clipped_line_xy.coords))
        coords_xy = np.array([c[:2] for c in clipped_line_xy.coords])
        return LineString(zip(*coords_xy.T, coords_z))


def has_multiple_turns(line: LineString) -> bool:
    """
    Determine whether LineString has multiple inflection points
    """
    # Get the coordinates of LineString
    coords = line.coords
    # Initialize the count of curvature change signs
    cross_products = []
    # Calculate the curvature of three adjacent points
    for i in range(1, len(coords) - 1):
        # Get three adjacent points
        p1 = coords[i - 1]
        p2 = coords[i]
        p3 = coords[i + 1]

        # Calculate vector 1
        vec1 = (p1[0] - p2[0], p1[1] - p2[1])
        # Calculate vector 2
        vec2 = (p3[0] - p2[0], p3[1] - p2[1])

        # Calculate the cross product of vector 1 and vector 2
        cross_products.append(vec1[0] * vec2[1] - vec1[1] * vec2[0])
    if all(c >= 0 for c in cross_products) or all(c <= 0 for c in cross_products):
        return False
    else:
        return True


def line_extend(line: LineString, extend_length: float):
    """
    Extend line at the beginning and end

    Args:
    - line (LineString): LineString
    - extend_length (float):  Length to extend line at both the beginning and end. unit: meter.

    Returns:
    - LineString: extended LineString
    """

    end_vec = get_end_vector(line)
    # Shapely's line length only calculates the horizontal distance
    norm_end_vec = end_vec / np.linalg.norm(end_vec[:2])
    start_vec = get_start_vector(line)
    norm_start_vec = start_vec / np.linalg.norm(start_vec[:2])
    # Starting point extension point
    start_extend_p = [
        list(np.array(line.coords[0]) + extend_length * np.array(norm_start_vec))
    ]
    # End extension point
    end_extend_p = [
        list(np.array(line.coords[-1]) + extend_length * np.array(norm_end_vec))
    ]
    return LineString(start_extend_p + line.coords[:] + end_extend_p)  # type: ignore


def line_max_curvature(line: LineString):
    """
    Calculate the maximum curvature of the line.
    """
    x = np.array([c[0] for c in line.coords])
    y = np.array([c[1] for c in line.coords])
    dydx = np.gradient(y, x)
    d2ydx2 = np.gradient(dydx, x)
    curvature = np.abs(d2ydx2) / (1 + dydx**2) ** (3 / 2)
    return max(curvature)


def connect_line_string(line1: LineString, line2: LineString) -> LineString:
    """
    Connect two line segments with a fit curve.
    """
    curve_lines: list[LineString] = []
    straight_line = connect_line_string_straight(line1, line2)
    curve_funcs = []
    p1 = np.array(line1.coords[-1][:2])
    p2 = np.array(line2.coords[0][:2])
    orig_strength = np.linalg.norm(p1 - p2) / 2
    for strength in np.linspace(0.4 * orig_strength, 0.9 * orig_strength, 8):
        curve_funcs.append(partial(connect_line_string_bezier_4_t_point, strength))
    for curve_func in curve_funcs:
        curve_line = curve_func(line1, line2)
        if has_multiple_turns(curve_line):
            continue
        else:
            curve_lines.append(curve_line)
    if not len(curve_lines) > 0:
        return straight_line
    else:
        sorted_lines = sorted(curve_lines, key=lambda l: line_max_curvature(l))
        return sorted_lines[-1]


def connect_line_string_bezier_4_t_point(
    strength: float, line1: LineString, line2: LineString
) -> LineString:
    """
    Connect two line segments with a 4 point Bezier curve.
    """
    has_z_coords = bool(any(len(c) > 2 for c in line1.coords))
    # p0 is the end point of line1
    # p3 is the start point of line2
    p0 = np.array(line1.coords[-1][:2])
    p3 = np.array(line2.coords[0][:2])
    # # The intensity is 1/2 of the distance between p0-p3
    # strength = np.linalg.norm(p0 - p3) / 2
    # p1 is the direction vector of the end point of line1
    delta1 = np.array(line1.coords[-1][:2]) - np.array(line1.coords[-2][:2])
    delta1 /= np.linalg.norm(delta1)
    p1 = p0 + delta1 * strength
    # p2 is the direction vector of the starting point of line2
    delta2 = np.array(line2.coords[1][:2]) - np.array(line2.coords[0][:2])
    delta2 /= np.linalg.norm(delta2)
    p2 = p3 - delta2 * strength
    # Generate Bezier curve
    t_values = np.linspace(0, 1, 10)

    # p0 p1 extension line intersects p2 p3
    line_p0_p1 = line_extend(LineString([p0, p1]), float(2 * strength))
    line_p2_p3 = LineString([p2, p3])
    if line_p0_p1.intersects(line_p2_p3):
        p2 = cast(Point, line_p0_p1.intersection(line_p2_p3))
        p2 = np.array(p2.coords[0])
    points = [p0, p1, p2, p3]
    curve = Bezier.Curve(t_values, points)
    line_xy = LineString(curve)
    if has_multiple_turns(line_xy):
        line_xy = LineString([p0, p3])
    if has_z_coords:
        z_start = line1.coords[-1][2]  # type: ignore
        z_end = line2.coords[0][2]  # type: ignore
        coords_z = np.linspace(z_start, z_end, len(line_xy.coords))
        coords_xy = np.array([c[:2] for c in line_xy.coords])
        return LineString(zip(*coords_xy.T, coords_z))
    else:
        return line_xy


def connect_line_string_straight(line1: LineString, line2: LineString) -> LineString:
    """
    Connect two line segments with straight line.
    """
    has_z_coords = bool(any(len(c) > 2 for c in line1.coords))
    # p0 is the end point of line1
    # p3 is the start point of line2
    p0 = np.array(line1.coords[-1][:2])
    p3 = np.array(line2.coords[0][:2])
    line_xy = LineString([p0, p3])
    if has_z_coords:
        z_start = line1.coords[-1][2]  # type: ignore
        z_end = line2.coords[0][2]  # type: ignore
        coords_z = np.linspace(z_start, z_end, len(line_xy.coords))
        coords_xy = np.array([c[:2] for c in line_xy.coords])
        return LineString(zip(*coords_xy.T, coords_z))
    else:
        return line_xy


def offset_lane(line: LineString, distance: float) -> LineString:
    """
    Offset LineString to the left/right

    Args:
    - line (LineString): LineString
    - distance (float): offset distance. Positive for left, negative for right. unit: meter.

    Returns:
    - LineString: offset LineString
    """
    has_z_coords = bool(any(len(c) > 2 for c in line.coords))
    line_xy = LineString([c[:2] for c in line.coords])
    offset_line = line_xy.offset_curve(distance)
    if offset_line:
        if not has_z_coords:
            return offset_line  # type: ignore
        else:
            coords_xy = np.array([c[:2] for c in offset_line.coords])
            z_start = line.coords[0][2]  # type: ignore
            z_end = line.coords[-1][2]  # type: ignore
            coords_z = np.linspace(z_start, z_end, len(offset_line.coords))
            return LineString(zip(*coords_xy.T, coords_z))
    else:
        # A lane that is too short will cause offset_curve to return empty. Return straight line.
        line_vec = np.array(line.coords[-1]) - np.array(line.coords[-2])
        line_angle = atan2(*line_vec[:2])
        if distance < 0:
            xy_direction = [
                np.cos(line_angle - np.pi / 2),
                np.sin(line_angle - np.pi / 2),
            ]
        else:
            xy_direction = [
                np.cos(line_angle + np.pi / 2),
                np.sin(line_angle + np.pi / 2),
            ]
        vec = np.array(xy_direction)
        offset_vec = vec / np.linalg.norm(vec) * np.abs(distance)
        if has_z_coords:
            offset_vec = np.append(offset_vec, 0)
        coords_all = [c + offset_vec for c in line.coords]
        return LineString(coords_all)


def align_line(line1: LineString, line2: LineString) -> LineString:
    """
    Align line1 along the direction of advance with line2.
    """

    line_1_xy = LineString([c[:2] for c in line1.coords])
    line_2_xy = LineString([c[:2] for c in line2.coords])
    if np.dot(get_end_vector(line_1_xy), get_end_vector(line_2_xy)) < 0:
        line_2_xy = LineString([c for c in line_2_xy.coords[::-1]])
    start_vec_2 = get_start_vector(line_2_xy)[:2]
    start_vec_2 = np.array([-start_vec_2[1], start_vec_2[0]])
    norm_start_vec_2 = start_vec_2 / np.linalg.norm(start_vec_2)
    end_vec_2 = get_end_vector(line_2_xy)[:2]
    end_vec_2 = np.array([-end_vec_2[1], end_vec_2[0]])
    norm_end_vec_2 = end_vec_2 / np.linalg.norm(end_vec_2)
    start_p_2 = [
        list(
            np.array(line_2_xy.coords[0])
            + line_2_xy.length * np.array(norm_start_vec_2)
        )
    ]
    start_vertical_line_2 = line_extend(
        LineString(line_2_xy.coords[:1] + start_p_2), line_2_xy.length  # type:ignore
    )
    start_intersections = start_vertical_line_2.intersection(line_1_xy)
    end_p_2 = [
        list(
            np.array(line_2_xy.coords[-1]) + line_2_xy.length * np.array(norm_end_vec_2)
        )
    ]
    end_vertical_line_2 = line_extend(
        LineString(line_2_xy.coords[-1:] + end_p_2), line_2_xy.length  # type:ignore
    )
    end_intersections = end_vertical_line_2.intersection(line_1_xy)
    clip_p_start = Point(line1.coords[0])
    clip_p_end = Point(line1.coords[-1])
    if isinstance(start_intersections, Point):
        clip_p_start = start_intersections
    if isinstance(end_intersections, Point):
        clip_p_end = end_intersections
    if line1.project(clip_p_start) > line1.project(clip_p_end):
        clip_p_start, clip_p_end = clip_p_end, clip_p_start
    clipped_line = clip_line(line1, clip_p_start, clip_p_end)
    clipped_line = clipped_line.simplify(0.001)
    return clipped_line  # type: ignore


def merge_near_xy_points(
    orig_points: List[Tuple[float, float]], merge_gate: float = 100
) -> List[Tuple[float, float]]:
    """
    Return merged coordinates of input points coordinates
    """
    merged_points = [tuple(c[:2]) for c in orig_points]
    orig_points = np.array(orig_points)  # type: ignore
    tree = KDTree(orig_points)
    for _, cur_point in enumerate(orig_points):
        _, indexes = tree.query(
            cur_point, k=len(orig_points), distance_upper_bound=merge_gate
        )
        indexes_set = {j for j in indexes if -1 < j < len(orig_points)}
        merge_coords = [c for i, c in enumerate(orig_points) if i in indexes_set]
        avg_coord = np.mean(merge_coords, axis=0)
        for j in indexes_set:
            merged_points[j] = avg_coord
    return merged_points  # type: ignore


def connect_split_lines(
    lines: List[LineString],
    start_point: Optional[Point] = None,
    max_line_length: float = 10_000,
) -> List:
    """
    Connect split lines (especially for `way` items in OSM) as one line.
    """
    merge_geo = linemerge(lines)
    if isinstance(merge_geo, LineString):
        return list(merge_geo.coords)

    def _remove_outlier_points(
        line: LineString,
        lines: List[LineString],
        distance_threshold: float = 20,
        max_line_length: float = 5000,
        angle_gate: float = np.pi / 2,
    ):
        coords = list(line.coords)
        line_tree = STRtree(lines)
        outlier_indexes = set()
        for idx in range(len(coords) - 1):
            pre_coord = coords[idx - 1]
            cur_coord = coords[idx]
            next_coord = coords[idx + 1]
            pre_line = LineString([pre_coord, cur_coord])
            cur_line = LineString([cur_coord, next_coord])
            delta_angle = abs_delta_angle(
                get_line_angle(pre_line), get_line_angle(cur_line)
            )
            split_points = [
                cur_line.interpolate(ratio, normalized=True)
                for ratio in np.linspace(0.1, 0.9, 10)
            ]
            has_near_points = []
            for p in split_points:
                tree_ids = line_tree.query_nearest(
                    geometry=p,
                    max_distance=distance_threshold,
                    return_distance=False,
                )
                if len(tree_ids) > 0:
                    has_near_points.append(p)
            if not len(has_near_points) / len(split_points) >= 0.5 or (
                cur_line.length > max_line_length
                and len(cur_line.coords) == 2
                and delta_angle > angle_gate
            ):
                outlier_indexes.add(idx + 1)
        if len(outlier_indexes) > 0:
            coords = coords[: list(outlier_indexes)[0]]
        return coords

    def _find_farthest_adjacent_points_and_rebase(
        coords: List, max_substring_length: float = 1500
    ):
        assert len(coords) >= 2
        pos_list = [
            (new_idx, cur_pos, next_pos, Point(cur_pos).distance(Point(next_pos)))
            for new_idx, (cur_pos, next_pos) in enumerate(
                zip(coords, coords[1:] + coords[:1]), start=1
            )
        ]
        new_index, _, _, max_distance = max(pos_list, key=lambda x: x[3])
        if max_distance > max_substring_length:
            return LineString(coords[:new_index] + coords[new_index + 1 :])
        else:
            return LineString(coords)

    lines = [l for l in lines if l.length < max_line_length]
    best_lines = []
    for reverse_flag in [True, False]:
        connected = []
        remaining = lines.copy()
        if start_point is not None:
            remaining = sorted(remaining, key=lambda x: start_point.distance(x))
        start_line = remaining.pop(0)
        if reverse_flag:
            start_line = LineString(start_line.coords[::-1])
        connected.append(start_line)
        while remaining:
            last_line = connected[-1]
            last_end_p = Point(last_line.coords[-1])
            sorted_lines = sorted(
                [(i, line) for i, line in enumerate(remaining)],
                key=lambda x: x[1].distance(last_line),
            )
            (i, line) = sorted_lines[0]
            proj_s = line.project(last_end_p, normalized=True)
            if proj_s < 1:
                connected.append(substring(line, proj_s * line.length, line.length))
            else:
                reverse_line = LineString(line.coords[::-1])
                reverse_proj_s = reverse_line.project(last_end_p, normalized=True)
                if reverse_proj_s < 1:
                    connected.append(
                        substring(
                            reverse_line,
                            reverse_proj_s * reverse_line.length,
                            reverse_line.length,
                        )
                    )
            del remaining[i]
        res_coords = [c for l in connected for c in l.coords]
        best_lines.append(_find_farthest_adjacent_points_and_rebase(res_coords))
    res = [_remove_outlier_points(l, lines) for l in best_lines]
    best_res = max(res, key=lambda x: len(x))
    return best_res


def merge_line_start_end(line_start: LineString, line_end: LineString) -> LineString:
    """
    Keep the start point of line_start and the end point of line_end unchanged, return merged result of the two lines
    """
    start_p = Point(line_start.coords[0])
    end_p = Point(line_end.coords[-1])
    line_start = clip_line(line_start, start_p, end_p)
    line_end = clip_line(line_end, start_p, end_p)
    line_start_coords = list(line_start.coords)
    end_offset = len(line_end.coords)
    for i, coord in enumerate(line_end.coords):
        if line_start.project(Point(coord)) >= line_start.length:
            end_offset = i
            break
    line_res_coords = line_start_coords + list(line_end.coords)[end_offset + 3 :]
    return LineString(list(start_p.coords) + line_res_coords[1:-1] + list(end_p.coords))


def get_start_vector(line: LineString):
    """Get the start vector of a LineString"""
    return np.array(line.coords[0]) - np.array(line.coords[1])


def get_end_vector(line: LineString):
    """Get the end vector of a LineString"""
    return np.array(line.coords[-1]) - np.array(line.coords[-2])


def get_line_angle(line: LineString):
    """Get the angle (-pi~pi) of the line"""
    v = get_start_vector(line)
    return np.arctan2(-v[1], -v[0])
