"""
Input/Output functions for maritime data
"""

import os
import json
from pathlib import Path
from typing import Dict, List
import geopandas as gpd
from shapely.geometry import mapping
from shapely.geometry import LineString
import geojson
import numpy as np
from PIL import Image


def create_output_folders(output_dir: str) -> Dict[str, str]:
    """
    Create output directory structure.
    
    Args:
        output_dir: Base output directory
        
    Returns:
        Dict with 'base', 'flir' paths
    """
    base_path = os.path.join(output_dir, 'maritime')
    flir_path = os.path.join(base_path, 'flir')
    
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(flir_path, exist_ok=True)
    
    return {
        'base': base_path,
        'flir': flir_path
    }


def write_geojson(gdf: gpd.GeoDataFrame, filepath: str):
    """
    Write GeoDataFrame to GeoJSON file.
    
    Args:
        gdf: GeoDataFrame
        filepath: Output file path
    """
    # Ensure CRS is WGS84
    if gdf.crs is None:
        gdf.set_crs('EPSG:4326', inplace=True)
    elif gdf.crs.to_string() != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')
    
    # Write to GeoJSON
    gdf.to_file(filepath, driver='GeoJSON')


def save_flir_frame(frame: np.ndarray, filepath: str):
    """
    Save FLIR frame as PNG.
    
    Args:
        frame: RGB image array (0-1)
        filepath: Output file path
    """
    # Convert to uint8
    frame_uint8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    
    # Create PIL Image and save
    img = Image.fromarray(frame_uint8, 'RGB')
    img.save(filepath, 'PNG')


# --- Added for GeoJSON route testing ---


def save_route_as_geojson(route, filepath, properties=None):
    """
    Save a route (list of (lon, lat) tuples) as a GeoJSON LineString FeatureCollection.
    Args:
        route: List of (lon, lat) tuples
        filepath: Output file path
        properties: Optional dict of properties for the feature
    """
    if not isinstance(route, list) or not all(isinstance(pt, (tuple, list)) and len(pt) == 2 for pt in route):
        raise ValueError("Route must be a list of (lon, lat) tuples.")
    line = LineString(route)
    feature = geojson.Feature(geometry=line, properties=properties or {})
    fc = geojson.FeatureCollection([feature])
    with open(filepath, 'w') as f:
        geojson.dump(fc, f, indent=2)
