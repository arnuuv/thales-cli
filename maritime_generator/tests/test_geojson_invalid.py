import unittest
import os
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import maritime_io

class TestGeoJSONInvalidInput(unittest.TestCase):
    def setUp(self):
        self.invalid_route = 'not_a_route'
        self.geojson_path = 'test_invalid.geojson'

    def tearDown(self):
        if os.path.exists(self.geojson_path):
            os.remove(self.geojson_path)

    def test_invalid_route_raises(self):
        with self.assertRaises(Exception):
            maritime_io.save_route_as_geojson(self.invalid_route, self.geojson_path)

if __name__ == '__main__':
    unittest.main()
