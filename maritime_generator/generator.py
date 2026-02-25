"""
Main Maritime Generator class
"""

import os
import numpy as np
import random
from typing import Dict, List, Optional
import geopandas as gpd
from shapely.geometry import Polygon, LineString

from geometry import (
    generate_mpa_polygon, generate_military_polygon, generate_windfarm_polygon,
    generate_ferry_route, generate_tss_route, generate_irregular_route
)
from flir import generate_flir_sequence
from maritime_io import create_output_folders, write_geojson, save_flir_frame
from utils import ensure_valid_geometry


class MaritimeGenerator:
    """
    Main class for generating fictitious maritime data.
    
    Generates:
    - Restricted zones (MPA, military, windfarm)
    - Vessel routes (ferry, TSS, irregular)
    - Synthetic FLIR images
    """
    
    def __init__(self, bbox: Dict, config: Dict, output_dir: str = 'output'):
        """
        Initialize maritime generator.
        
        Args:
            bbox: Bounding box dict with min_lon, min_lat, max_lon, max_lat
            config: Configuration dict (see module docstring)
            output_dir: Base output directory
        """
        self.bbox = bbox
        self.config = config
        self.output_dir = output_dir
        
        # Set random seed
        seed = config.get('random_seed', None)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            self.seed = seed
        else:
            self.seed = None
        
        # Create output folders
        self.paths = create_output_folders(output_dir)
        
        # Storage for generated data
        self.zones_gdf = None
        self.routes_gdf = None
    
    def generate_zones(self) -> gpd.GeoDataFrame:
        """
        Generate all maritime restricted zones.
        
        Returns:
            GeoDataFrame with zone polygons
        """
        zones = []
        zone_id = 1
        
        num_zones = self.config.get('num_zones', {})
        
        # Generate MPA zones
        num_mpa = num_zones.get('mpa', 0)
        for i in range(num_mpa):
            seed = (self.seed + zone_id * 100) if self.seed is not None else None
            polygon = generate_mpa_polygon(self.bbox, seed=seed)
            
            zones.append({
                'zone_id': f'MPA_{zone_id:03d}',
                'zone_type': 'mpa',
                'restriction': 'no_entry',
                'description': f'Marine Protected Area {zone_id} - Fictitious zone for training purposes',
                'geometry': polygon
            })
            zone_id += 1
        
        # Generate military zones
        num_military = num_zones.get('military', 0)
        for i in range(num_military):
            seed = (self.seed + zone_id * 100) if self.seed is not None else None
            polygon = generate_military_polygon(self.bbox, seed=seed)
            
            zones.append({
                'zone_id': f'MIL_{zone_id:03d}',
                'zone_type': 'military',
                'restriction': 'no_entry',
                'description': f'Military Restricted Zone {zone_id} - Fictitious zone for training purposes',
                'geometry': polygon
            })
            zone_id += 1
        
        # Generate windfarm zones
        num_windfarm = num_zones.get('windfarm', 0)
        for i in range(num_windfarm):
            seed = (self.seed + zone_id * 100) if self.seed is not None else None
            polygon = generate_windfarm_polygon(self.bbox, seed=seed)
            
            zones.append({
                'zone_id': f'WIN_{zone_id:03d}',
                'zone_type': 'windfarm',
                'restriction': 'caution',
                'description': f'Windfarm Zone {zone_id} - Fictitious zone for training purposes',
                'geometry': polygon
            })
            zone_id += 1
        
        # Create GeoDataFrame
        if zones:
            self.zones_gdf = gpd.GeoDataFrame(zones, crs='EPSG:4326')
        else:
            # Empty GeoDataFrame
            self.zones_gdf = gpd.GeoDataFrame(
                columns=['zone_id', 'zone_type', 'restriction', 'description', 'geometry'],
                crs='EPSG:4326'
            )
        
        return self.zones_gdf
    
    def generate_routes(
        self, 
        num_routes: int, 
        route_types: List[str], 
        obstacles: List[Polygon] = None  # ✅ ADD THIS PARAMETER
    ) -> List[Dict]:
        """
        Generate vessel routes
        
        Args:
            num_routes: Number of routes to generate
            route_types: List of route types ('ferry', 'tss', 'irregular')
            obstacles: Optional list of Polygons to avoid
        """
        if obstacles is None:
            obstacles = []
        
        routes = []
        
        for i in range(num_routes):
            route_type = self.rng.choice(route_types)
            
            # Generate random start and end points within bounds
            start_lon = self.rng.uniform(self.min_lon, self.max_lon)
            start_lat = self.rng.uniform(self.min_lat, self.max_lat)
            end_lon = self.rng.uniform(self.min_lon, self.max_lon)
            end_lat = self.rng.uniform(self.min_lat, self.max_lat)
            
            # ✅ IMPORT THE ROUTE CALCULATION FUNCTION
            from user_data_handler import calculate_route_avoiding_zones
            
            # Calculate waypoints (avoiding obstacles for ferry/tss)
            if route_type in ["ferry", "tss"] and obstacles:
                waypoints = calculate_route_avoiding_zones(
                    (start_lon, start_lat),
                    (end_lon, end_lat),
                    obstacles,
                    route_type
                )
            else:
                waypoints = [(start_lon, start_lat), (end_lon, end_lat)]
            
            routes.append({
                "id": f"route_{i}",
                "type": route_type,
                "waypoints": waypoints
            })
        
        self.routes = routes
        return routes
    
    def generate_flir(self) -> List[str]:
        """
        Generate FLIR image sequence.
        
        Returns:
            List of file paths to generated FLIR images
        """
        num_frames = self.config.get('flir_frames', 10)
        size = (256, 256)  # Default FLIR size
        
        # Determine vessel type
        vessel_type = 'generic'
        
        seed = (self.seed + 5000) if self.seed is not None else None
        
        # Generate sequence
        frames = generate_flir_sequence(num_frames, size, vessel_type, seed=seed)
        
        # Save frames
        filepaths = []
        for i, frame in enumerate(frames):
            filename = f'frame_{i+1:04d}.png'
            filepath = os.path.join(self.paths['flir'], filename)
            save_flir_frame(frame, filepath)
            filepaths.append(filepath)
        
        return filepaths
    
    def export(self):
        """
        Export all generated data to files.
        """
        # Export zones
        if self.zones_gdf is not None and len(self.zones_gdf) > 0:
            zones_path = os.path.join(self.paths['base'], 'zones.geojson')
            write_geojson(self.zones_gdf, zones_path)
            print(f"✅ Exported {len(self.zones_gdf)} zones to {zones_path}")
        else:
            print("⚠️  No zones to export")
        
        # Export routes
        if self.routes_gdf is not None and len(self.routes_gdf) > 0:
            routes_path = os.path.join(self.paths['base'], 'routes.geojson')
            write_geojson(self.routes_gdf, routes_path)
            print(f"✅ Exported {len(self.routes_gdf)} routes to {routes_path}")
        else:
            print("⚠️  No routes to export")
    
    def run(self):
        """
        Run complete generation pipeline.
        """
        print("🌊 Starting maritime data generation...")
        
        # Generate zones
        print("📍 Generating restricted zones...")
        self.generate_zones()
        
        # Generate routes
        print("🚢 Generating vessel routes...")
        self.generate_routes()
        
        # Generate FLIR
        print("📷 Generating FLIR sequence...")
        flir_paths = self.generate_flir()
        print(f"✅ Generated {len(flir_paths)} FLIR frames")
        
        # Export
        print("💾 Exporting data...")
        self.export()
        
        print("✅ Maritime generation complete!")




