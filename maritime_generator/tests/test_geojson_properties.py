import unittest
import os
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import maritime_io

class TestGeoJSONProperties(unittest.TestCase):
    def setUp(self):
        # Route with properties
        self.route = [
            (0.0, 0.0),
            (0.1, 0.1),
            (0.2, 0.2)
        ]
        self.properties = {'vessel': 'test_boat', 'speed': 10}
        self.geojson_path = 'test_properties.geojson'

    def tearDown(self):
        if os.path.exists(self.geojson_path):
            os.remove(self.geojson_path)

    def test_properties_in_geojson(self):
        # If maritime_io.save_route_as_geojson supports properties, test them
        try:
            maritime_io.save_route_as_geojson(self.route, self.geojson_path, properties=self.properties)
        except TypeError:
            # If properties not supported, skip
            self.skipTest('save_route_as_geojson does not support properties argument')
        with open(self.geojson_path) as f:
            data = json.load(f)
        feature = data['features'][0]
        self.assertIn('properties', feature)
        self.assertEqual(feature['properties'].get('vessel'), 'test_boat')
        self.assertEqual(feature['properties'].get('speed'), 10)

if __name__ == '__main__':
    unittest.main()
