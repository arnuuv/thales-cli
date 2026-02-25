#!/usr/bin/env python3
"""
Main generator script called by Next.js API
"""

import json
import sys
from pathlib import Path
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from generator import MaritimeGenerator
from user_data_handler import (
    load_user_zones, 
    load_user_routes, 
    export_user_zones_geojson, 
    export_user_routes_geojson
)

def main():
    parser = argparse.ArgumentParser(description="Generate maritime data")
    parser.add_argument("--bbox", required=True, help="Path to bbox JSON file")
    parser.add_argument("--config", required=True, help="Path to config JSON file")
    parser.add_argument("--output", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    # Load bbox
    with open(args.bbox, 'r') as f:
        bbox = json.load(f)
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    print("🌊 Starting maritime data generation")
    print(f"📦 Bbox: {bbox}")
    print(f"⚙️  Config: {config}")
    
    output_dir = Path(args.output)
    maritime_dir = output_dir / "maritime"
    maritime_dir.mkdir(parents=True, exist_ok=True)
    
    # ✅ HANDLE USER-DEFINED DATA
    user_zones_polygons = load_user_zones(config)
    user_routes_data = load_user_routes(config)
    
    # Export user zones as GeoJSON
    if user_zones_polygons:
        user_zones_raw = config.get("user_no_go_zones", [])
        export_user_zones_geojson(
            user_zones_polygons,
            maritime_dir / "user_zones.geojson",
            user_zones_raw
        )
    
    # Export user routes as GeoJSON (with obstacle avoidance)
    if user_routes_data:
        export_user_routes_geojson(
            user_routes_data,
            maritime_dir / "user_routes.geojson",
            obstacles=user_zones_polygons
        )
    
    # ✅ EXISTING PROCEDURAL GENERATION (if configured)
    if config.get("num_routes", 0) > 0 or sum(config.get("num_zones", {}).values()) > 0:
        print("\n🎲 Generating procedural data...")
        
        generator = MaritimeGenerator(
            min_lon=bbox["min_lon"],
            min_lat=bbox["min_lat"],
            max_lon=bbox["max_lon"],
            max_lat=bbox["max_lat"],
            random_seed=config.get("random_seed", 42)
        )
        
        # Generate procedural zones
        zones = generator.generate_zones(
            num_mpa=config["num_zones"]["mpa"],
            num_military=config["num_zones"]["military"],
            num_windfarm=config["num_zones"]["windfarm"]
        )
        
        # Combine user zones with procedural zones for route planning
        all_obstacles = user_zones_polygons + [z["polygon"] for z in zones]
        
        # Generate procedural routes (avoiding all obstacles)
        routes = generator.generate_routes(
            num_routes=config["num_routes"],
            route_types=config["route_type"],
            obstacles=all_obstacles
        )
        
        # Export procedural data
        generator.export_zones(maritime_dir / "procedural_zones.geojson")
        generator.export_routes(maritime_dir / "procedural_routes.geojson")
        
        # Generate FLIR images if requested
        if config.get("flir_frames", 0) > 0:
            print(f"\n📷 Generating {config['flir_frames']} FLIR images...")
            flir_dir = maritime_dir / "flir"
            flir_dir.mkdir(exist_ok=True)
            generator.generate_flir_images(
                flir_dir,
                num_frames=config["flir_frames"]
            )
    
    print("\n✅ Maritime data generation complete!")
    print(f"📁 Output: {maritime_dir}")

if __name__ == "__main__":
    main()




