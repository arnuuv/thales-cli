#!/usr/bin/env python3
"""
Produce GeoJSON routes for CLI integration.
Generates procedural maritime routes and writes a single GeoJSON file.
"""

import json
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from geometry import generate_ferry_route, generate_tss_route, generate_irregular_route

DEFAULT_BBOX = {
    "min_lon": -5.0,
    "min_lat": 50.0,
    "max_lon": 0.0,
    "max_lat": 55.0,
}


def line_to_feature(line, route_id: str, route_type: str, name: str):
    """Convert a Shapely LineString to a GeoJSON Feature."""
    coords = [[float(x), float(y)] for x, y in line.coords]
    return {
        "type": "Feature",
        "properties": {"id": route_id, "type": route_type, "name": name},
        "geometry": {"type": "LineString", "coordinates": coords},
    }


def main():
    parser = argparse.ArgumentParser(description="Produce maritime routes GeoJSON")
    parser.add_argument(
        "--output",
        type=str,
        default="routes.geojson",
        help="Output GeoJSON file path (default: routes.geojson)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num-routes",
        type=int,
        default=5,
        help="Number of routes to generate (default: 5)",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    features = []
    seed = args.seed
    n = args.num_routes

    # Mix of ferry, TSS, and irregular routes
    for i in range(n):
        if i % 3 == 0:
            line = generate_ferry_route(DEFAULT_BBOX, seed=seed + i * 100)
            features.append(
                line_to_feature(line, f"route_{i}", "ferry", f"Ferry route {i + 1}")
            )
        elif i % 3 == 1:
            lines = generate_tss_route(DEFAULT_BBOX, seed=seed + i * 100)
            for j, line in enumerate(lines):
                features.append(
                    line_to_feature(
                        line, f"route_{i}_lane_{j}", "tss", f"TSS route {i + 1} lane {j + 1}"
                    )
                )
        else:
            data = generate_irregular_route(DEFAULT_BBOX, seed=seed + i * 100)
            line = data["route"]
            features.append(
                line_to_feature(line, f"route_{i}", "irregular", f"Irregular route {i + 1}")
            )

    fc = {"type": "FeatureCollection", "features": features}
    with open(out_path, "w") as f:
        json.dump(fc, f, indent=2)

    print(f"Exported {len(features)} routes to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
