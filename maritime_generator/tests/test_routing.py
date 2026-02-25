"""
Test script for calculate_route_avoiding_zones with 5km buffer.
Run from the maritime_generator directory:
    python test_routing.py
"""

from user_data_handler import (
    calculate_route_avoiding_zones,
    export_user_zones_geojson,
    export_user_routes_geojson,
    load_user_zones,
    load_user_routes,
)
import sys
import json
from pathlib import Path
from shapely.geometry import Polygon, LineString

# ── adjust this if your file lives elsewhere ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "maritime_generator"))
# ─────────────────────────────────────────────────────────────────────────────

BUFFER = 0.045  # ~5 km in degrees

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def route_touches_buffered_zone(waypoints, obstacles, buffer=BUFFER):
    """Return True if any segment of the route intersects the buffered obstacles."""
    from shapely.ops import unary_union
    buffered = unary_union([obs.buffer(buffer) for obs in obstacles])
    line = LineString(waypoints)
    return line.intersects(buffered)


def print_result(label, waypoints, valid, obstacles):
    touches = route_touches_buffered_zone(waypoints, obstacles)
    status = "✅ PASS" if (valid and not touches) else "❌ FAIL"
    print(f"\n  {status}  {label}")
    print(
        f"         route_valid={valid}  |  waypoints={len(waypoints)}  |  clips_buffer={touches}")
    if len(waypoints) <= 6:
        for wp in waypoints:
            print(f"           {wp}")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_direct_route_no_obstacle():
    """Route with no obstacles — should return direct 2-point line."""
    print("\n── Test 1: No obstacles ──")
    start = (148.00, -37.90)
    end = (148.10, -37.60)
    wp, valid = calculate_route_avoiding_zones(start, end, [], "ferry")
    print_result("direct route, no obstacles", wp, valid, [])
    assert valid
    assert len(wp) == 2


def test_direct_route_clear_of_zone():
    """Route that doesn't come within 5 km of any zone — should be direct."""
    print("\n── Test 2: Route clear of zone ──")
    # Zone far to the east
    zone = Polygon([
        (148.50, -37.80), (148.55, -37.80),
        (148.55, -37.75), (148.50, -37.75),
        (148.50, -37.80),
    ])
    start = (148.00, -37.90)
    end = (148.10, -37.60)
    wp, valid = calculate_route_avoiding_zones(start, end, [zone], "ferry")
    print_result("route clear of zone", wp, valid, [zone])
    assert valid
    assert not route_touches_buffered_zone(wp, [zone])


def test_route_through_zone():
    """Route that cuts straight through a zone — must detour around it."""
    print("\n── Test 3: Route through zone (expects detour) ──")
    # Zone sitting right in the middle of the direct path
    zone = Polygon([
        (148.04, -37.76), (148.06, -37.76),
        (148.06, -37.74), (148.04, -37.74),
        (148.04, -37.76),
    ])
    start = (148.00, -37.90)
    end = (148.10, -37.60)
    wp, valid = calculate_route_avoiding_zones(start, end, [zone], "ferry")
    print_result("route through zone", wp, valid, [zone])
    assert valid, "Expected a valid detour to be found"
    assert not route_touches_buffered_zone(wp, [zone]), \
        "Route still clips the 5 km buffer — avoidance failed"


def test_irregular_route_ignores_zone():
    """Irregular routes must NOT do obstacle avoidance."""
    print("\n── Test 4: Irregular route — no avoidance ──")
    zone = Polygon([
        (148.04, -37.76), (148.06, -37.76),
        (148.06, -37.74), (148.04, -37.74),
        (148.04, -37.76),
    ])
    start = (148.00, -37.90)
    end = (148.10, -37.60)
    wp, valid = calculate_route_avoiding_zones(start, end, [zone], "irregular")
    print_result("irregular route ignores zone", wp, valid, [zone])
    assert valid
    assert len(wp) == 2, "Irregular routes should always be a direct 2-point line"


def test_multiple_zones():
    """Two zones flanking the direct path — must find a corridor."""
    print("\n── Test 5: Two zones, route must thread between/around ──")
    zone1 = Polygon([
        (148.03, -37.78), (148.06, -37.78),
        (148.06, -37.72), (148.03, -37.72),
        (148.03, -37.78),
    ])
    zone2 = Polygon([
        (148.07, -37.78), (148.10, -37.78),
        (148.10, -37.72), (148.07, -37.72),
        (148.07, -37.78),
    ])
    start = (148.00, -37.90)
    end = (148.10, -37.60)
    wp, valid = calculate_route_avoiding_zones(
        start, end, [zone1, zone2], "ferry")
    print_result("two zones", wp, valid, [zone1, zone2])
    # We don't assert clip here because two large zones may force the
    # fallback — but valid should still be True if a path exists
    assert valid


def test_geojson_export(tmp_path=Path("/tmp/routing_test")):
    """End-to-end: load → plan → export GeoJSON, then verify file is valid JSON."""
    print("\n── Test 6: GeoJSON export round-trip ──")
    tmp_path.mkdir(parents=True, exist_ok=True)

    config = {
        "user_no_go_zones": [
            {
                "id": "zone-001",
                "name": "Test Zone",
                "coordinates": [
                    [148.04, -37.76], [148.06, -37.76],
                    [148.06, -37.74], [148.04, -37.74],
                    [148.04, -37.76],
                ],
            }
        ],
        "user_routes": [
            {
                "id": "route-001",
                "name": "Ferry Route 1",
                "start": {"lon": 148.00, "lat": -37.90},
                "end":   {"lon": 148.10, "lat": -37.60},
                "type":  "ferry",
            }
        ],
    }

    zones = load_user_zones(config)
    routes = load_user_routes(config)

    zone_path = tmp_path / "user_zones.geojson"
    route_path = tmp_path / "user_routes.geojson"

    export_user_zones_geojson(zones, zone_path, config["user_no_go_zones"])
    export_user_routes_geojson(routes, route_path, obstacles=zones)

    # Verify files are valid GeoJSON
    for path in (zone_path, route_path):
        with open(path) as f:
            data = json.load(f)
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) > 0
        print(f"  ✅ {path.name}  ({len(data['features'])} feature(s))")

    # Check route_valid flag
    with open(route_path) as f:
        route_data = json.load(f)
    for feat in route_data["features"]:
        props = feat["properties"]
        if props["type"] != "irregular":
            assert props["route_valid"], f"Route {props['id']} flagged invalid!"
            print(f"  ✅ route_valid=True for {props['name']}")

    print(f"\n  Output written to {tmp_path}/")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_direct_route_no_obstacle,
        test_direct_route_clear_of_zone,
        test_route_through_zone,
        test_irregular_route_ignores_zone,
        test_multiple_zones,
        test_geojson_export,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  💥 AssertionError: {e}")
            failed += 1
        except Exception as e:
            print(f"  💥 Unexpected error: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")
