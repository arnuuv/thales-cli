"""
Utility functions for maritime data generation
"""

import numpy as np
import random
from typing import Tuple, List
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


def random_point_in_bbox(bbox: dict, seed: int = None) -> Tuple[float, float]:
    """
    Generate a random point within the bounding box.
    
    Args:
        bbox: Dict with min_lon, min_lat, max_lon, max_lat
        seed: Optional random seed
        
    Returns:
        (lon, lat) tuple
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    lon = np.random.uniform(bbox['min_lon'], bbox['max_lon'])
    lat = np.random.uniform(bbox['min_lat'], bbox['max_lat'])
    
    return (lon, lat)


def convert_km_to_deg(km: float, latitude: float) -> float:
    """
    Convert kilometers to degrees (approximate).
    
    Args:
        km: Distance in kilometers
        latitude: Latitude for longitude conversion
        
    Returns:
        Degrees (average of lat/lon conversion)
    """
    # 1 degree latitude ≈ 111 km
    lat_deg = km / 111.0
    
    # 1 degree longitude ≈ 111 km * cos(latitude)
    lon_deg = km / (111.0 * np.cos(np.radians(latitude)))
    
    # Return average for simplicity
    return (lat_deg + lon_deg) / 2.0


def smooth_polygon(polygon: Polygon, smoothness: float = 0.3) -> Polygon:
    """
    Smooth a polygon by applying a buffer operation.
    
    Args:
        polygon: Input polygon
        smoothness: Buffer distance (in degrees, typically 0.01-0.1)
        
    Returns:
        Smoothed polygon
    """
    # Buffer with negative then positive to smooth
    smoothed = polygon.buffer(-smoothness).buffer(smoothness)
    
    # Ensure valid geometry
    if not smoothed.is_valid:
        smoothed = smoothed.buffer(0)
    
    return smoothed


def random_polygon(center: Tuple[float, float], radius_deg: float, 
                   num_points: int = 8, irregularity: float = 0.3,
                   seed: int = None) -> Polygon:
    """
    Generate a random polygon around a center point.
    
    Args:
        center: (lon, lat) center point
        radius_deg: Approximate radius in degrees
        num_points: Number of vertices
        irregularity: How irregular (0.0 = circle, 1.0 = very irregular)
        seed: Optional random seed
        
    Returns:
        Shapely Polygon
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    center_lon, center_lat = center
    points = []
    
    # Generate points in a circle with irregularity
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        
        # Add irregularity
        angle += np.random.uniform(-irregularity, irregularity)
        radius = radius_deg * (1.0 + np.random.uniform(-irregularity * 0.5, irregularity * 0.5))
        
        # Account for latitude in longitude scaling
        lon_offset = radius * np.cos(angle) / np.cos(np.radians(center_lat))
        lat_offset = radius * np.sin(angle)
        
        lon = center_lon + lon_offset
        lat = center_lat + lat_offset
        
        points.append((lon, lat))
    
    # Close the polygon
    if points[0] != points[-1]:
        points.append(points[0])
    
    return Polygon(points)


def ensure_valid_geometry(geom) -> Polygon:
    """
    Ensure geometry is valid using buffer(0) trick.
    
    Args:
        geom: Shapely geometry
        
    Returns:
        Valid geometry
    """
    if not geom.is_valid:
        geom = geom.buffer(0)
    
    return geom


def clip_to_bbox(geom, bbox: dict) -> Polygon:
    """
    Clip geometry to bounding box.
    
    Args:
        geom: Shapely geometry
        bbox: Bounding box dict
        
    Returns:
        Clipped geometry
    """
    bbox_poly = Polygon([
        (bbox['min_lon'], bbox['min_lat']),
        (bbox['max_lon'], bbox['min_lat']),
        (bbox['max_lon'], bbox['max_lat']),
        (bbox['min_lon'], bbox['max_lat']),
        (bbox['min_lon'], bbox['min_lat'])
    ])
    
    clipped = geom.intersection(bbox_poly)
    
    if isinstance(clipped, Polygon):
        return ensure_valid_geometry(clipped)
    elif hasattr(clipped, 'geoms'):
        # MultiPolygon - take largest
        largest = max(clipped.geoms, key=lambda p: p.area)
        return ensure_valid_geometry(largest)
    else:
        return ensure_valid_geometry(geom)
