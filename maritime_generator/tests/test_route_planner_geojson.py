import unittest
import os
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import route_planner
import maritime_io

class TestRoutePlannerGeoJSON(unittest.TestCase):
    def setUp(self):
        # Create a simple route using route_planner (if available)
        self.start = (0.0, 0.0)
        self.end = (1.0, 1.0)
        # If route_planner has a function to generate a route, use it; else, use a dummy route
        if hasattr(route_planner, 'generate_route'):
            self.route = route_planner.generate_route(self.start, self.end)
        else:
            self.route = [self.start, (0.5, 0.5), self.end]
        self.geojson_path = 'test_route_planner.geojson'

    def tearDown(self):
        if os.path.exists(self.geojson_path):
            os.remove(self.geojson_path)

    def test_route_to_geojson(self):
        # Save the route as GeoJSON
        maritime_io.save_route_as_geojson(self.route, self.geojson_path)
        self.assertTrue(os.path.exists(self.geojson_path))
        with open(self.geojson_path) as f:
            data = json.load(f)
        self.assertIn('features', data)
        self.assertEqual(data['features'][0]['geometry']['type'], 'LineString')
        coords = data['features'][0]['geometry']['coordinates']
        self.assertEqual(coords[0], list(self.start))
        self.assertEqual(coords[-1], list(self.end))

if __name__ == '__main__':
    unittest.main()
