"""
Route planning with obstacle avoidance for maritime routes
"""

import numpy as np
from typing import List, Tuple, Optional
from shapely.geometry import Point, Polygon, LineString, MultiPolygon
from shapely.ops import unary_union
import math


def plan_collision_free_route(
    start: Tuple[float, float],
    end: Tuple[float, float],
    exclusion_zones: List[Polygon],
    num_waypoints: int = 10,
    buffer_distance_km: float = 5.0
) -> List[Tuple[float, float]]:
    """
    Plan a route from start to end that avoids exclusion zones.
    
    Args:
        start: (lon, lat) start point
        end: (lon, lat) end point
        exclusion_zones: List of Shapely Polygons representing no-go zones
        num_waypoints: Desired number of waypoints
        buffer_distance_km: Safety buffer around zones in km
        
    Returns:
        List of (lon, lat) waypoints forming collision-free route
    """
    # Convert buffer to degrees (approximate at equator)
    buffer_deg = buffer_distance_km / 111.0
    
    # Buffer all exclusion zones
    buffered_zones = []
    for zone in exclusion_zones:
        try:
            buffered = zone.buffer(buffer_deg)
            if buffered.is_valid:
                buffered_zones.append(buffered)
        except Exception:
            continue
    
    # Merge overlapping zones
    if buffered_zones:
        merged_zones = unary_union(buffered_zones)
        if isinstance(merged_zones, Polygon):
            merged_zones = [merged_zones]
        elif isinstance(merged_zones, MultiPolygon):
            merged_zones = list(merged_zones.geoms)
        else:
            merged_zones = []
    else:
        merged_zones = []
    
    # Generate initial straight-line route
    initial_route = generate_straight_line_waypoints(start, end, num_waypoints)
    
    # Check for collisions
    route_line = LineString(initial_route)
    has_collision = any(route_line.intersects(zone) for zone in merged_zones)
    
    for zone in merged_zones:
        if route_line.intersects(zone):
            has_collision = True
            break
    
    # If no collision, return direct route
    if not has_collision:
        return initial_route
    
    # If collision detected, route around obstacles
    return route_around_obstacles(start, end, merged_zones, num_waypoints)


def generate_straight_line_waypoints(
    start: Tuple[float, float],
    end: Tuple[float, float],
    num_points: int
) -> List[Tuple[float, float]]:
    """Generate evenly spaced waypoints along a straight line."""
    waypoints = [start]
    
    for i in range(1, num_points):
        t = i / num_points
        lon = start[0] + (end[0] - start[0]) * t
        lat = start[1] + (end[1] - start[1]) * t
        waypoints.append((lon, lat))
    
    waypoints.append(end)
    return waypoints


def route_around_obstacles(
    start: Tuple[float, float],
    end: Tuple[float, float],
    obstacles: List[Polygon],
    num_waypoints: int
) -> List[Tuple[float, float]]:
    """
    Route around obstacles using visibility graph approach.
    
    Strategy:
    1. Find which obstacle(s) block the direct path
    2. Add waypoints around obstacle perimeters
    3. Choose shortest detour
    """
    if not obstacles:
        return generate_straight_line_waypoints(start, end, num_waypoints)
    
    # Create direct line
    direct_line = LineString([start, end])
    
    # Find intersecting obstacles
    intersecting_obstacles = []
    for obstacle in obstacles:
        if direct_line.intersects(obstacle):
            intersecting_obstacles.append(obstacle)
    
    if not intersecting_obstacles:
        return generate_straight_line_waypoints(start, end, num_waypoints)
    
    # Simple approach: route around the first major obstacle
    obstacle = intersecting_obstacles[0]
    
    # Get obstacle vertices
    if hasattr(obstacle, 'exterior'):
        vertices = list(obstacle.exterior.coords)
    else:
        # Fallback to bounding box
        bounds = obstacle.bounds
        vertices = [
            (bounds[0], bounds[1]),  # min_lon, min_lat
            (bounds[2], bounds[1]),  # max_lon, min_lat
            (bounds[2], bounds[3]),  # max_lon, max_lat
            (bounds[0], bounds[3]),  # min_lon, max_lat
        ]
    
    # Find two vertices that form the best detour
    detour_route = find_best_detour(start, end, vertices, obstacle)
    
    final_line = LineString(detour_route)
    still_colliding = any(final_line.intersects(o) for o in obstacles)

    if still_colliding:
        merged = unary_union(obstacles)
        bounds = merged.bounds
        
        mid_lon = (bounds[0] + bounds[2]) / 2
        mid_lat = (bounds[1] + bounds[3]) / 2

        corners = [
            (bounds[0] - 0.01, bounds[1] - 0.01),
            (bounds[2] + 0.01, bounds[1] - 0.01),
            (bounds[2] + 0.01, bounds[3] + 0.01),
            (bounds[0] - 0.01, bounds[3] + 0.01),
        ]

        for corner in corners:
            candidate = smooth_waypoints(
                [start, corner, end], num_points=num_waypoints)
            candidate_line = LineString(candidate)
            if not any(candidate_line.intersects(o) for o in obstacles):
                return candidate

        return detour_route
    
    return detour_route


def find_best_detour(
    start: Tuple[float, float],
    end: Tuple[float, float],
    vertices: List[Tuple[float, float]],
    obstacle: Polygon
) -> List[Tuple[float, float]]:
    """
    Find the best detour around an obstacle.
    
    Strategy: Try routing around left or right side, choose shorter path.
    """
    start_point = Point(start)
    end_point = Point(end)
    
    # Calculate which side to route (left or right)
    # Based on which vertices are closest to start and end
    
    left_path = []
    right_path = []
    
    for i, vertex in enumerate(vertices[:-1]):  # Exclude last (duplicate of first)
        v_point = Point(vertex)
        
        # Distance from vertex to start/end line
        direct_line = LineString([start, end])
        distance_to_line = v_point.distance(direct_line)
        
        # Classify as left or right based on cross product
        dx1 = end[0] - start[0]
        dy1 = end[1] - start[1]
        dx2 = vertex[0] - start[0]
        dy2 = vertex[1] - start[1]
        
        cross = dx1 * dy2 - dy1 * dx2
        
        if cross > 0:
            left_path.append(vertex)
        else:
            right_path.append(vertex)
    
    # Calculate path lengths
    def path_length(waypoints):
        total = 0
        points = [start] + waypoints + [end]
        for i in range(len(points) - 1):
            total += distance(points[i], points[i + 1])
        return total
    
    left_length = path_length(left_path) if left_path else float('inf')
    right_length = path_length(right_path) if right_path else float('inf')
    
    # Choose shorter path
    if left_length <= right_length and left_path:
        chosen_path = [start] + left_path + [end]
    elif right_path:
        chosen_path = [start] + right_path + [end]
    else:
        # Fallback: direct route
        chosen_path = [start, end]
    
    # Smooth the path
    return smooth_waypoints(chosen_path, num_points=15)


def smooth_waypoints(
    waypoints: List[Tuple[float, float]],
    num_points: int
) -> List[Tuple[float, float]]:
    """
    Smooth waypoints by interpolating between key points.
    """
    if len(waypoints) < 2:
        return waypoints
    
    # Calculate cumulative distances
    distances = [0]
    for i in range(1, len(waypoints)):
        dist = distance(waypoints[i - 1], waypoints[i])
        distances.append(distances[-1] + dist)
    
    total_distance = distances[-1]
    
    if total_distance == 0:
        return waypoints
    
    # Generate evenly spaced points
    smooth_points = []
    for i in range(num_points + 1):
        target_dist = (i / num_points) * total_distance
        
        # Find segment
        for j in range(len(distances) - 1):
            if distances[j] <= target_dist <= distances[j + 1]:
                # Interpolate within segment
                segment_start = waypoints[j]
                segment_end = waypoints[j + 1]
                segment_length = distances[j + 1] - distances[j]
                
                if segment_length > 0:
                    t = (target_dist - distances[j]) / segment_length
                    lon = segment_start[0] + t * (segment_end[0] - segment_start[0])
                    lat = segment_start[1] + t * (segment_end[1] - segment_start[1])
                    smooth_points.append((lon, lat))
                else:
                    smooth_points.append(segment_start)
                break
    
    return smooth_points


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points (approximation)."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def calculate_route_distance_km(waypoints: List[Tuple[float, float]]) -> float:
    """
    Calculate total route distance in kilometers using Haversine formula.
    """
    R = 6371  # Earth's radius in km
    total_distance = 0
    
    for i in range(len(waypoints) - 1):
        lon1, lat1 = waypoints[i]
        lon2, lat2 = waypoints[i + 1]
        
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        # Haversine formula
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        total_distance += R * c
    
    return total_distance