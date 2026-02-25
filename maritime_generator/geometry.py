"""
Geometry generation functions for maritime zones and routes
"""

import numpy as np
import random
from typing import Tuple, List, Dict
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union

from utils import (
    random_point_in_bbox, convert_km_to_deg, smooth_polygon,
    random_polygon, ensure_valid_geometry, clip_to_bbox
)


def generate_mpa_polygon(bbox: dict, radius_km: float = None, 
                        seed: int = None) -> Polygon:
    """
    Generate a Marine Protected Area (MPA) polygon.
    
    Characteristics:
    - Soft, organic shapes
    - Random radius between 5-30 km
    - Slight irregularity
    
    Args:
        bbox: Bounding box
        radius_km: Optional radius in km (default: random 5-30)
        seed: Random seed
        
    Returns:
        Shapely Polygon
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Random center point
    center = random_point_in_bbox(bbox, seed)
    center_lat = center[1]
    
    # Random radius if not specified
    if radius_km is None:
        radius_km = np.random.uniform(5.0, 30.0)
    
    # Convert to degrees
    radius_deg = convert_km_to_deg(radius_km, center_lat)
    
    # Generate organic shape with many points
    num_points = np.random.randint(12, 20)
    irregularity = np.random.uniform(0.2, 0.4)
    
    polygon = random_polygon(center, radius_deg, num_points, irregularity, seed)
    
    # Smooth for organic appearance
    smooth_dist = radius_deg * 0.15
    polygon = smooth_polygon(polygon, smooth_dist)
    
    # Clip to bbox and ensure valid
    polygon = clip_to_bbox(polygon, bbox)
    polygon = ensure_valid_geometry(polygon)
    
    return polygon


def generate_military_polygon(bbox: dict, seed: int = None) -> Polygon:
    """
    Generate a military restricted zone polygon.
    
    Characteristics:
    - Angular polygons (5-8 points)
    - Rectangular-ish or trapezoid
    - Structured/planned appearance
    
    Args:
        bbox: Bounding box
        seed: Random seed
        
    Returns:
        Shapely Polygon
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    center = random_point_in_bbox(bbox, seed)
    center_lat = center[1]
    
    # Military zones are typically 10-50 km
    radius_km = np.random.uniform(10.0, 50.0)
    radius_deg = convert_km_to_deg(radius_km, center_lat)
    
    # Angular shape with fewer points
    num_points = np.random.randint(5, 8)
    irregularity = np.random.uniform(0.05, 0.15)  # Less irregular
    
    polygon = random_polygon(center, radius_deg, num_points, irregularity, seed)
    
    # Make it more angular (less smoothing)
    smooth_dist = radius_deg * 0.05
    polygon = smooth_polygon(polygon, smooth_dist)
    
    # Clip and validate
    polygon = clip_to_bbox(polygon, bbox)
    polygon = ensure_valid_geometry(polygon)
    
    return polygon


def generate_windfarm_polygon(bbox: dict, seed: int = None) -> Polygon:
    """
    Generate a windfarm zone polygon.
    
    Characteristics:
    - Circular or boxy polygon
    - Realistic but fictional
    
    Args:
        bbox: Bounding box
        seed: Random seed
        
    Returns:
        Shapely Polygon
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    center = random_point_in_bbox(bbox, seed)
    center_lat = center[1]
    
    # Windfarms are typically 5-20 km
    radius_km = np.random.uniform(5.0, 20.0)
    radius_deg = convert_km_to_deg(radius_km, center_lat)
    
    # Choose between circular or boxy
    shape_type = random.choice(['circular', 'boxy'])
    
    if shape_type == 'circular':
        # More circular (many points, low irregularity)
        num_points = np.random.randint(16, 24)
        irregularity = np.random.uniform(0.05, 0.15)
    else:
        # Boxy (4-6 points, very low irregularity)
        num_points = np.random.randint(4, 6)
        irregularity = np.random.uniform(0.01, 0.05)
    
    polygon = random_polygon(center, radius_deg, num_points, irregularity, seed)
    
    # Light smoothing
    smooth_dist = radius_deg * 0.08
    polygon = smooth_polygon(polygon, smooth_dist)
    
    # Clip and validate
    polygon = clip_to_bbox(polygon, bbox)
    polygon = ensure_valid_geometry(polygon)
    
    return polygon


def generate_ferry_route(bbox: dict, seed: int = None) -> LineString:
    """
    Generate a ferry route.
    
    Characteristics:
    - Smooth LineString
    - Start and end near coast (within bbox)
    
    Args:
        bbox: Bounding box
        seed: Random seed
        
    Returns:
        Shapely LineString
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Start and end points (simulate coastal points)
    start = random_point_in_bbox(bbox, seed)
    
    # End point should be on opposite side (simulate crossing)
    # Add some randomness
    end_seed = seed + 1000 if seed is not None else None
    if end_seed is not None:
        np.random.seed(end_seed)
        random.seed(end_seed)
    
    end = random_point_in_bbox(bbox, end_seed)
    
    # Create smooth curve with intermediate points
    num_points = np.random.randint(8, 15)
    points = []
    
    for i in range(num_points + 1):
        t = i / num_points
        
        # Linear interpolation with slight curve
        lon = start[0] + (end[0] - start[0]) * t
        lat = start[1] + (end[1] - start[1]) * t
        
        # Add slight curve (sine wave)
        curve_amplitude = 0.02  # degrees
        curve_offset = curve_amplitude * np.sin(t * np.pi)
        
        # Perpendicular to route direction
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            perp_lon = -dy / length * curve_offset
            perp_lat = dx / length * curve_offset
        else:
            perp_lon = 0
            perp_lat = 0
        
        lon += perp_lon
        lat += perp_lat
        
        points.append((lon, lat))
    
    return LineString(points)


def generate_tss_route(bbox: dict, seed: int = None) -> List[LineString]:
    """
    Generate a Traffic Separation Scheme (TSS) route.
    
    Characteristics:
    - Two parallel lanes
    - Curved segments
    
    Args:
        bbox: Bounding box
        seed: Random seed
        
    Returns:
        List of two LineStrings (inbound and outbound lanes)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Generate base route
    base_route = generate_ferry_route(bbox, seed)
    
    # Create two parallel lanes
    lane_separation_km = 2.0  # Typical TSS separation
    center_lat = (bbox['min_lat'] + bbox['max_lat']) / 2
    separation_deg = convert_km_to_deg(lane_separation_km, center_lat) / 2
    
    # Get perpendicular direction
    coords = list(base_route.coords)
    if len(coords) < 2:
        # Fallback to simple route
        return [base_route, base_route]
    
    # Calculate perpendicular offset
    dx = coords[-1][0] - coords[0][0]
    dy = coords[-1][1] - coords[0][1]
    length = np.sqrt(dx**2 + dy**2)
    
    if length > 0:
        perp_lon = -dy / length * separation_deg
        perp_lat = dx / length * separation_deg
    else:
        perp_lon = separation_deg
        perp_lat = 0
    
    # Create two parallel routes
    inbound_coords = [(lon + perp_lon, lat + perp_lat) for lon, lat in coords]
    outbound_coords = [(lon - perp_lon, lat - perp_lat) for lon, lat in coords]
    
    inbound_route = LineString(inbound_coords)
    outbound_route = LineString(outbound_coords)
    
    return [inbound_route, outbound_route]


def generate_irregular_route(bbox: dict, num_deviations: int = 3,
                            num_stops: int = 2, seed: int = None) -> Dict:
    """
    Generate an irregular/suspicious route.
    
    Characteristics:
    - Random deviations
    - Stopping points
    - May enter restricted zones
    
    Args:
        bbox: Bounding box
        num_deviations: Number of deviation points
        num_stops: Number of stopping points
        seed: Random seed
        
    Returns:
        Dict with 'route' (LineString), 'deviations' (list), 'stops' (list)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Start with a base route
    base_route = generate_ferry_route(bbox, seed)
    coords = list(base_route.coords)
    
    # Add deviations
    deviations = []
    for i in range(num_deviations):
        # Random point along route
        t = np.random.uniform(0.2, 0.8)
        idx = int(t * (len(coords) - 1))
        
        # Create deviation
        deviation_magnitude = np.random.uniform(0.05, 0.15)  # degrees
        angle = np.random.uniform(0, 2 * np.pi)
        
        dev_lon = coords[idx][0] + deviation_magnitude * np.cos(angle)
        dev_lat = coords[idx][1] + deviation_magnitude * np.sin(angle)
        
        # Ensure within bbox
        dev_lon = np.clip(dev_lon, bbox['min_lon'], bbox['max_lon'])
        dev_lat = np.clip(dev_lat, bbox['min_lat'], bbox['max_lat'])
        
        deviations.append((dev_lon, dev_lat))
        
        # Insert into route
        coords.insert(idx + 1, (dev_lon, dev_lat))
    
    # Add stops (stationary points)
    stops = []
    for i in range(num_stops):
        t = np.random.uniform(0.1, 0.9)
        idx = int(t * (len(coords) - 1))
        stops.append(coords[idx])
    
    route = LineString(coords)
    
    return {
        'route': route,
        'deviations': deviations,
        'stops': stops
    }
