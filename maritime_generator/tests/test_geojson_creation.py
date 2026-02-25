import unittest
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import generator
import maritime_io

class TestGeoJSONCreation(unittest.TestCase):
    def setUp(self):
        # Example route data for testing
        self.route = [
            (0.0, 0.0),
            (0.1, 0.1),
            (0.2, 0.2)
        ]
        self.output_path = 'test_output.geojson'

    def tearDown(self):
        if os.path.exists(self.output_path):
            os.remove(self.output_path)

    def test_geojson_creation(self):
        # Assuming maritime_io has a function to save route as geojson
        maritime_io.save_route_as_geojson(self.route, self.output_path)
        self.assertTrue(os.path.exists(self.output_path))
        with open(self.output_path) as f:
            data = json.load(f)
        self.assertIn('features', data)
        self.assertEqual(data['features'][0]['geometry']['type'], 'LineString')

    def test_geojson_structure(self):
        maritime_io.save_route_as_geojson(self.route, self.output_path)
        with open(self.output_path) as f:
            data = json.load(f)
        coords = data['features'][0]['geometry']['coordinates']
        self.assertEqual(coords, [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])

if __name__ == '__main__':
    unittest.main()
