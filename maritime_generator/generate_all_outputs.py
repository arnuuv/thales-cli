#!/usr/bin/env python3
"""
Enhanced maritime generator that produces ALL required outputs:
1. vessels.geojson - Vessel positions and metadata
2. sea_lanes.geojson - Shipping lanes
3. coastal_assets.geojson - Ports, lighthouses, buoys
4. routes.geojson - Vessel routes

This script is called by the Go CLI to generate complete maritime datasets.
"""

import json
import sys
import argparse
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).parent))

from geometry import generate_ferry_route, generate_tss_route, generate_irregular_route

DEFAULT_BBOX = {
    "min_lon": -122.5,
    "min_lat": 37.5,
    "max_lon": -122.3,
    "max_lat": 37.7,
}


def line_to_feature(line, route_id: str, route_type: str, name: str):
    """Convert a Shapely LineString to a GeoJSON Feature."""
    coords = [[float(x), float(y)] for x, y in line.coords]
    return {
        "type": "Feature",
        "properties": {"id": route_id, "type": route_type, "name": name},
        "geometry": {"type": "LineString", "coordinates": coords},
    }


def generate_vessels(bbox, num_vessels=10, seed=42):
    """Generate vessel positions within the bounding box."""
    random.seed(seed)
    
    vessels = []
    vessel_types = ["cargo", "tanker", "passenger", "fishing", "military", "tug"]
    
    for i in range(num_vessels):
        lon = random.uniform(bbox["min_lon"], bbox["max_lon"])
        lat = random.uniform(bbox["min_lat"], bbox["max_lat"])
        vessel_type = random.choice(vessel_types)
        
        # Speed varies by vessel type
        speed_ranges = {
            "cargo": (10, 20),
            "tanker": (8, 15),
            "passenger": (20, 30),
            "fishing": (5, 12),
            "military": (15, 35),
            "tug": (8, 14),
        }
        
        speed = random.uniform(*speed_ranges[vessel_type])
        heading = random.uniform(0, 360)
        
        vessels.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            },
            "properties": {
                "id": f"vessel_{i+1:03d}",
                "name": f"{vessel_type.title()} {chr(65 + i % 26)}{i+1}",
                "type": vessel_type,
                "speed_knots": round(speed, 1),
                "heading": round(heading, 1),
                "mmsi": f"{200000000 + i}",
                "length_m": random.randint(50, 300),
            }
        })
    
    return {
        "type": "FeatureCollection",
        "features": vessels
    }


def generate_sea_lanes(bbox, num_lanes=4, seed=42):
    """Generate shipping lanes within the bounding box."""
    random.seed(seed)
    
    lanes = []
    
    # Generate parallel lanes (northbound/southbound or eastbound/westbound)
    for i in range(num_lanes):
        if i % 2 == 0:
            # North-South lane
            lon = bbox["min_lon"] + (i / num_lanes) * (bbox["max_lon"] - bbox["min_lon"])
            coords = [
                [lon, bbox["min_lat"]],
                [lon + 0.01, bbox["min_lat"] + (bbox["max_lat"] - bbox["min_lat"]) * 0.3],
                [lon - 0.01, bbox["min_lat"] + (bbox["max_lat"] - bbox["min_lat"]) * 0.7],
                [lon, bbox["max_lat"]],
            ]
            direction = "northbound" if i % 4 == 0 else "southbound"
        else:
            # East-West lane
            lat = bbox["min_lat"] + (i / num_lanes) * (bbox["max_lat"] - bbox["min_lat"])
            coords = [
                [bbox["min_lon"], lat],
                [bbox["min_lon"] + (bbox["max_lon"] - bbox["min_lon"]) * 0.3, lat + 0.01],
                [bbox["min_lon"] + (bbox["max_lon"] - bbox["min_lon"]) * 0.7, lat - 0.01],
                [bbox["max_lon"], lat],
            ]
            direction = "eastbound" if i % 4 == 1 else "westbound"
        
        lanes.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords
            },
            "properties": {
                "id": f"lane_{i+1:03d}",
                "name": f"Shipping Lane {chr(65 + i)}",
                "direction": direction,
                "width_m": random.randint(400, 800),
                "traffic_density": random.choice(["low", "medium", "high"]),
            }
        })
    
    return {
        "type": "FeatureCollection",
        "features": lanes
    }


def generate_coastal_assets(bbox, num_assets=15, seed=42):
    """Generate coastal assets (ports, lighthouses, buoys) within the bounding box."""
    random.seed(seed)
    
    assets = []
    asset_types = ["port", "lighthouse", "buoy", "anchorage", "pilot_station"]
    
    for i in range(num_assets):
        # Bias positions toward edges for coastal assets
        if random.random() < 0.6:
            # Place near edges
            edge = random.choice(["north", "south", "east", "west"])
            if edge == "north":
                lon = random.uniform(bbox["min_lon"], bbox["max_lon"])
                lat = bbox["max_lat"] - random.uniform(0, 0.05)
            elif edge == "south":
                lon = random.uniform(bbox["min_lon"], bbox["max_lon"])
                lat = bbox["min_lat"] + random.uniform(0, 0.05)
            elif edge == "east":
                lon = bbox["max_lon"] - random.uniform(0, 0.05)
                lat = random.uniform(bbox["min_lat"], bbox["max_lat"])
            else:  # west
                lon = bbox["min_lon"] + random.uniform(0, 0.05)
                lat = random.uniform(bbox["min_lat"], bbox["max_lat"])
        else:
            # Place anywhere
            lon = random.uniform(bbox["min_lon"], bbox["max_lon"])
            lat = random.uniform(bbox["min_lat"], bbox["max_lat"])
        
        asset_type = random.choice(asset_types)
        
        properties = {
            "id": f"{asset_type}_{i+1:03d}",
            "name": f"{asset_type.replace('_', ' ').title()} {chr(65 + i % 26)}{i+1}",
            "type": asset_type,
        }
        
        # Add type-specific properties
        if asset_type == "port":
            properties.update({
                "berths": random.randint(2, 20),
                "max_draft_m": random.randint(8, 18),
            })
        elif asset_type == "lighthouse":
            properties.update({
                "range_nm": random.randint(10, 30),
                "height_m": random.randint(15, 60),
            })
        elif asset_type == "buoy":
            properties.update({
                "buoy_type": random.choice(["navigation", "mooring", "special_mark"]),
                "light": random.choice([True, False]),
            })
        
        assets.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            },
            "properties": properties
        })
    
    return {
        "type": "FeatureCollection",
        "features": assets
    }


def generate_routes(bbox, num_routes=10, seed=42):
    """Generate vessel routes using existing route generation functions."""
    features = []
    
    # Mix of ferry, TSS, and irregular routes
    for i in range(num_routes):
        if i % 3 == 0:
            line = generate_ferry_route(bbox, seed=seed + i * 100)
            features.append(
                line_to_feature(line, f"route_{i}", "ferry", f"Ferry route {i + 1}")
            )
        elif i % 3 == 1:
            lines = generate_tss_route(bbox, seed=seed + i * 100)
            for j, line in enumerate(lines):
                features.append(
                    line_to_feature(
                        line, f"route_{i}_lane_{j}", "tss", f"TSS route {i + 1} lane {j + 1}"
                    )
                )
        else:
            data = generate_irregular_route(bbox, seed=seed + i * 100)
            line = data["route"]
            features.append(
                line_to_feature(line, f"route_{i}", "irregular", f"Irregular route {i + 1}")
            )
    
    return {
        "type": "FeatureCollection",
        "features": features
    }


def main():
    parser = argparse.ArgumentParser(description="Generate complete maritime dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for GeoJSON files",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num-vessels",
        type=int,
        default=10,
        help="Number of vessels to generate (default: 10)",
    )
    parser.add_argument(
        "--num-lanes",
        type=int,
        default=4,
        help="Number of sea lanes to generate (default: 4)",
    )
    parser.add_argument(
        "--num-assets",
        type=int,
        default=15,
        help="Number of coastal assets to generate (default: 15)",
    )
    parser.add_argument(
        "--num-routes",
        type=int,
        default=10,
        help="Number of routes to generate (default: 10)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🌊 Generating maritime data in {output_dir}")
    print(f"   Seed: {args.seed}")

    # Generate all outputs
    print("\n📍 Generating vessels...")
    vessels = generate_vessels(DEFAULT_BBOX, num_vessels=args.num_vessels, seed=args.seed)
    vessels_path = output_dir / "vessels.geojson"
    with open(vessels_path, "w") as f:
        json.dump(vessels, f, indent=2)
    print(f"   ✓ Saved {len(vessels['features'])} vessels to {vessels_path}")

    print("\n🛤️  Generating sea lanes...")
    sea_lanes = generate_sea_lanes(DEFAULT_BBOX, num_lanes=args.num_lanes, seed=args.seed + 1000)
    sea_lanes_path = output_dir / "sea_lanes.geojson"
    with open(sea_lanes_path, "w") as f:
        json.dump(sea_lanes, f, indent=2)
    print(f"   ✓ Saved {len(sea_lanes['features'])} sea lanes to {sea_lanes_path}")

    print("\n⚓ Generating coastal assets...")
    coastal_assets = generate_coastal_assets(DEFAULT_BBOX, num_assets=args.num_assets, seed=args.seed + 2000)
    coastal_assets_path = output_dir / "coastal_assets.geojson"
    with open(coastal_assets_path, "w") as f:
        json.dump(coastal_assets, f, indent=2)
    print(f"   ✓ Saved {len(coastal_assets['features'])} coastal assets to {coastal_assets_path}")

    print("\n🗺️  Generating routes...")
    routes = generate_routes(DEFAULT_BBOX, num_routes=args.num_routes, seed=args.seed + 3000)
    routes_path = output_dir / "routes.geojson"
    with open(routes_path, "w") as f:
        json.dump(routes, f, indent=2)
    print(f"   ✓ Saved {len(routes['features'])} routes to {routes_path}")

    print("\n✅ Maritime data generation complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"   - vessels.geojson ({len(vessels['features'])} features)")
    print(f"   - sea_lanes.geojson ({len(sea_lanes['features'])} features)")
    print(f"   - coastal_assets.geojson ({len(coastal_assets['features'])} features)")
    print(f"   - routes.geojson ({len(routes['features'])} features)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
