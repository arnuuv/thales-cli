"""
Handler for user-defined no-go zones and routes
"""

import json
import heapq
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from shapely.geometry import Polygon, LineString, Point, MultiPolygon
from shapely.ops import unary_union
import numpy as np


def load_user_zones(config: Dict) -> List[Polygon]:
    """
    Load user-defined no-go zones from config

    Returns:
        List of Shapely Polygon objects
    """
    user_zones = config.get("user_no_go_zones", [])
    polygons = []

    for zone in user_zones:
        coords = zone.get("coordinates", [])
        if len(coords) >= 3:
            # Close the ring if not already closed
            if coords[0] != coords[-1]:
                coords.append(coords[0])

            try:
                poly = Polygon(coords)
                if poly.is_valid:
                    polygons.append(poly)
                else:
                    # Try to fix invalid polygon
                    poly = poly.buffer(0)
                    if poly.is_valid:
                        polygons.append(poly)
                        print(
                            f"⚠️  Fixed invalid polygon for zone {zone.get('name', 'Unknown')}")
                    else:
                        print(
                            f"⚠️  Invalid polygon for zone {zone.get('name', 'Unknown')}, skipping")
            except Exception as e:
                print(f"❌ Error creating polygon: {e}")

    print(f"Loaded {len(polygons)} user-defined no-go zones")
    return polygons


def load_user_routes(config: Dict) -> List[Dict]:
    """
    Load user-defined routes from config

    Returns:
        List of route dictionaries with start, end, type
    """
    user_routes = config.get("user_routes", [])
    routes = []

    for route in user_routes:
        start = route.get("start", {})
        end = route.get("end", {})
        route_type = route.get("type", "irregular")

        if "lon" in start and "lat" in start and "lon" in end and "lat" in end:
            routes.append({
                "id": route.get("id", f"route_{len(routes)}"),
                "name": route.get("name", f"Route {len(routes) + 1}"),
                "start": (start["lon"], start["lat"]),
                "end": (end["lon"], end["lat"]),
                "type": route_type
            })

    print(f"Loaded {len(routes)} user-defined routes")
    return routes


def export_user_zones_geojson(zones: List[Polygon], output_path: Path, zone_data: List[Dict]):
    """
    Export user-defined no-go zones as GeoJSON
    """
    features = []

    for i, (poly, zone_info) in enumerate(zip(zones, zone_data)):
        feature = {
            "type": "Feature",
            "properties": {
                "id": zone_info.get("id", f"user_zone_{i}"),
                "name": zone_info.get("name", f"User No-Go Zone {i + 1}"),
                "type": "user_defined",
                "zone_type": zone_info.get("zoneType", "mpa"),
                "restriction": "no-go"
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [list(poly.exterior.coords)]
            }
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    print(f"Exported {len(features)} user zones to {output_path}")


def _extract_vertices(obstacles) -> List[Tuple[float, float]]:
    """
    Extract all exterior boundary vertices from a Polygon or MultiPolygon.
    Excludes the duplicate closing vertex from each ring.
    """
    vertices = []
    if isinstance(obstacles, Polygon):
        vertices.extend(list(obstacles.exterior.coords)[:-1])
    elif isinstance(obstacles, MultiPolygon):
        for poly in obstacles.geoms:
            vertices.extend(list(poly.exterior.coords)[:-1])
    return vertices


def _line_clear(a: Tuple[float, float], b: Tuple[float, float], obstacles, edge_buffer: float) -> bool:
    line = LineString([a, b])
    return not line.intersects(obstacles)


def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

# Add midpoints along each polygon edge as intermediate nodes


def extract_edge_midpoints(obstacles):
    midpoints = []
    if isinstance(obstacles, Polygon):
        coords = list(obstacles.exterior.coords)[:-1]
        n = len(coords)
        for i in range(n):
            a, b = coords[i], coords[(i+1) % n]
            midpoints.append(((a[0]+b[0])/2, (a[1]+b[1])/2))
    elif isinstance(obstacles, MultiPolygon):
        for poly in obstacles.geoms:
            midpoints.extend(extract_edge_midpoints(poly))
    return midpoints


def _offset_vertices_outward(obstacles, vertices, offset=0.01):
    """
    Nudge each vertex away from the nearest obstacle boundary point,
    so routes don't hug the buffer edge. Works correctly for
    concave and multi-polygon obstacles.
    """
    nudged = []
    for (vx, vy) in vertices:
        p = Point(vx, vy)
        # Find nearest point on obstacle boundary
        nearest = obstacles.exterior.interpolate(
            obstacles.exterior.project(p)
        ) if isinstance(obstacles, Polygon) else obstacles.boundary.interpolate(
            obstacles.boundary.project(p)
        )
        dx = vx - nearest.x
        dy = vy - nearest.y
        dist = (dx**2 + dy**2) ** 0.5
        if dist == 0:
            # Vertex is exactly on boundary — push away from centroid instead
            cx, cy = obstacles.centroid.x, obstacles.centroid.y
            dx = vx - cx
            dy = vy - cy
            dist = (dx**2 + dy**2) ** 0.5
        if dist == 0:
            nudged.append((vx, vy))
            continue
        nudged.append((
            vx + (dx / dist) * offset,
            vy + (dy / dist) * offset,
        ))
    return nudged

def calculate_waypoints_visibility_graph(
    start: Tuple[float, float],
    end: Tuple[float, float],
    obstacles,
    buffer: float
) -> Optional[List[Tuple[float, float]]]:
    """
    Calculate shortest path from start to end avoiding obstacles using a full
    visibility graph with Dijkstra's algorithm.

    Nodes: start, end, and all obstacle boundary vertices.
    Edges: any pair of nodes where the connecting segment doesn't intersect obstacles.
    """
    edge_buffer = buffer * 0.1

    vertices = _extract_vertices(obstacles)
    if not vertices:
        return None
    
    vertices = _offset_vertices_outward(
        obstacles, vertices, offset=buffer * 0.5)

    midpoints = extract_edge_midpoints(obstacles)
    midpoints = _offset_vertices_outward(
        obstacles, midpoints, offset=buffer * 0.5)

    # Build node list: index 0 = start, index 1 = end, rest = obstacle vertices
    nodes = [start, end] + vertices + midpoints
    n = len(nodes)

    START_IDX = 0
    END_IDX = 1

    # Precompute adjacency: only store edges that are clear
    # Use a dict of lists for the graph
    graph: Dict[int, List[Tuple[float, int]]] = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            if _line_clear(nodes[i], nodes[j], obstacles, edge_buffer):
                dist = _euclidean(nodes[i], nodes[j])
                graph[i].append((dist, j))
                graph[j].append((dist, i))

    # Dijkstra from START_IDX to END_IDX
    dist_map = {i: float('inf') for i in range(n)}
    dist_map[START_IDX] = 0.0
    prev = {i: None for i in range(n)}
    heap = [(0.0, START_IDX)]

    while heap:
        current_dist, u = heapq.heappop(heap)

        if current_dist > dist_map[u]:
            continue

        if u == END_IDX:
            break

        for edge_dist, v in graph[u]:
            new_dist = dist_map[u] + edge_dist
            if new_dist < dist_map[v]:
                dist_map[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))

    # Reconstruct path
    if dist_map[END_IDX] == float('inf'):
        print("  ⚠️  Visibility graph could not find a clear path")
        return None

    path_indices = []
    current = END_IDX
    while current is not None:
        path_indices.append(current)
        current = prev[current]
    path_indices.reverse()

    waypoints = [nodes[i] for i in path_indices]
    return waypoints


def calculate_waypoints_using_vertices(
    start: Tuple[float, float],
    end: Tuple[float, float],
    obstacles
) -> Optional[List[Tuple[float, float]]]:
    """
    Fallback: attempt to route around the bounding box of the combined obstacles.
    Tries single waypoints on all four sides, then two-corner combinations for
    concave or awkward obstacle shapes.
    """
    bounds = obstacles.bounds  # (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = bounds
    pad = 0.1  # degrees of clearance beyond bounding box

    mid_lon = (start[0] + end[0]) / 2
    mid_lat = (start[1] + end[1]) / 2

    # Four single-waypoint candidates (one per side of the bounding box)
    single_candidates = [
        (mid_lon, maxy + pad),  # north
        (mid_lon, miny - pad),  # south
        (maxx + pad, mid_lat),  # east
        (minx - pad, mid_lat),  # west
    ]

    for wp in single_candidates:
        seg1 = LineString([start, wp])
        seg2 = LineString([wp, end])
        if not seg1.intersects(obstacles) and not seg2.intersects(obstacles):
            return [start, wp, end]

    # Two-corner combinations for concave/complex obstacles
    corners = [
        (minx - pad, maxy + pad),  # NW
        (maxx + pad, maxy + pad),  # NE
        (minx - pad, miny - pad),  # SW
        (maxx + pad, miny - pad),  # SE
    ]

    for i, c1 in enumerate(corners):
        for c2 in corners[i + 1:]:
            segs = [
                LineString([start, c1]),
                LineString([c1, c2]),
                LineString([c2, end]),
            ]
            if not any(s.intersects(obstacles) for s in segs):
                return [start, c1, c2, end]

    return None


def _adaptive_buffer(obstacles, min_buf=0.005, max_buf=0.018):
    """Use smaller buffer when zones are close together to avoid trapping routes."""
    if len(obstacles) < 2:
        return max_buf

    # Find minimum distance between any two zones
    min_dist = float('inf')
    for i, z1 in enumerate(obstacles):
        for z2 in obstacles[i+1:]:
            dist = z1.distance(z2)
            if dist < min_dist:
                min_dist = dist

    # If zones are close, shrink buffer proportionally
    if min_dist < max_buf * 2:
        return max(min_buf, min_dist * 0.3)
    return max_buf


def calculate_route_avoiding_zones(
    start, end, obstacles, route_type="ferry", buffer_distance=None
):
    if route_type == "irregular":
        return [start, end], True
    if not obstacles:
        return [start, end], True
    
    adaptive = _adaptive_buffer(obstacles)
    
    if buffer_distance is None:
        buffer_distance = adaptive
    else:
        if buffer_distance > adaptive:
            print(f"  ⚠️  Explicit buffer {buffer_distance:.4f}° exceeds adaptive recommendation "
                  f"{adaptive:.4f}° — zones may be too close for this buffer size")
        else:
            print(
                f"  📏 Using explicit buffer: {buffer_distance:.4f}° ({buffer_distance * 111:.1f} km)")

    buffered_obstacles = [obs.buffer(buffer_distance) for obs in obstacles]
    combined_buffered = unary_union(buffered_obstacles)

    start = _snap_outside(start, combined_buffered)
    end = _snap_outside(end,   combined_buffered)

    direct_line = LineString([start, end])
    if not direct_line.intersects(combined_buffered):
        return [start, end], True

    print(f"  🔄 Route intersects obstacles, calculating waypoints...")
    waypoints = calculate_waypoints_visibility_graph(
        start, end, combined_buffered, buffer_distance)
    if waypoints and len(waypoints) >= 2:
        final_line = LineString(waypoints)
        if not final_line.intersects(combined_buffered):
            print(
                f"  ✅ Visibility graph found valid route with {len(waypoints)} waypoints")
            return waypoints, True
        else:
            print(f"  ⚠️  Visibility graph route still clips buffer, trying fallback...")

    waypoints = calculate_waypoints_using_vertices(
        start, end, combined_buffered)
    if waypoints:
        return waypoints, True

    print(f"  ❌ Could not find clear route, flagging as invalid")
    return [start, end], False


def _snap_outside(point, combined_buffered, step=0.005):
    """If point is inside the buffered zone, nudge it outside in 8 directions."""
    if not combined_buffered.contains(Point(point)):
        return point

    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (-1, 1), (1, -1), (-1, -1)
    ]
    for dist_multiplier in range(1, 20):
        for dx, dy in directions:
            candidate = (
                point[0] + dx * step * dist_multiplier,
                point[1] + dy * step * dist_multiplier,
            )
            if not combined_buffered.contains(Point(candidate)):
                return candidate
    return point  # give up, return original


def export_user_routes_geojson(routes: List[Dict], output_path: Path, obstacles: List[Polygon] = None):
    """
    Export user-defined routes as GeoJSON, with path planning for ferry/tss routes.
    Invalid routes (where no clear path could be found) are flagged in properties.
    """
    features = []

    for route in routes:
        start = route["start"]
        end = route["end"]
        route_type = route["type"]

        print(f"\n🚢 Processing route: {route['name']} ({route_type})")

        route_valid = True
        if obstacles and route_type in ["ferry", "tss"]:
            waypoints, route_valid = calculate_route_avoiding_zones(
                start, end, obstacles, route_type)
        else:
            waypoints = [start, end]
            if route_type != "irregular":
                print(f"  No obstacles provided, using direct line")
            else:
                print(f"  Irregular route — no obstacle avoidance")

        properties = {
            "id": route["id"],
            "name": route["name"],
            "type": route_type,
            "vessel_type": (
                "ferry" if route_type == "ferry"
                else "cargo" if route_type == "tss"
                else "unknown"
            ),
            "suspicious": route_type == "irregular",
            "waypoints": len(waypoints),
            "route_valid": route_valid,
        }

        if not route_valid:
            properties["warning"] = "Route passes through restricted zone — no clear path could be calculated"

        feature = {
            "type": "Feature",
            "properties": properties,
            "geometry": {
                "type": "LineString",
                "coordinates": waypoints
            }
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    invalid_count = sum(
        1 for f in features if not f["properties"]["route_valid"])
    print(f"\nExported {len(features)} user routes to {output_path}")
    if invalid_count:
        print(
            f"  ⚠️  {invalid_count} route(s) flagged as invalid — inspect GeoJSON warnings")
