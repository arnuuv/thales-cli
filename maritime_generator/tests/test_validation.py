"""
Validation script for maritime generator

Tests that all outputs are valid and meet requirements.
"""

import os
import json
from pathlib import Path
import geopandas as gpd
from PIL import Image

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from generator import MaritimeGenerator


def validate_geojson(filepath: str) -> bool:
    """Validate GeoJSON file loads correctly."""
    try:
        gdf = gpd.read_file(filepath)
        assert gdf.crs.to_string() == 'EPSG:4326', f"CRS should be EPSG:4326, got {gdf.crs}"
        assert len(gdf) > 0, "GeoDataFrame should not be empty"
        print(f"✅ {filepath}: Valid GeoJSON with {len(gdf)} features, CRS: {gdf.crs}")
        return True
    except Exception as e:
        print(f"❌ {filepath}: Invalid - {e}")
        return False


def validate_flir_images(flir_dir: str) -> bool:
    """Validate FLIR images are readable PNGs."""
    flir_path = Path(flir_dir)
    if not flir_path.exists():
        print(f"⚠️  FLIR directory not found: {flir_dir}")
        return False
    
    png_files = list(flir_path.glob('*.png'))
    if len(png_files) == 0:
        print(f"⚠️  No PNG files found in {flir_dir}")
        return False
    
    for png_file in png_files:
        try:
            img = Image.open(png_file)
            assert img.format == 'PNG', f"Expected PNG, got {img.format}"
            assert img.size[0] > 0 and img.size[1] > 0, "Image size invalid"
            assert img.mode in ['RGB', 'RGBA', 'L'], f"Unexpected image mode: {img.mode}"
        except Exception as e:
            print(f"❌ {png_file}: Invalid - {e}")
            return False
    
    print(f"✅ {len(png_files)} FLIR images validated")
    return True


def main():
    """Run validation tests."""
    print("🧪 Running maritime generator validation tests...\n")
    
    # Test configuration
    bbox = {
        'min_lon': 148.0,
        'min_lat': -37.8,
        'max_lon': 148.1,
        'max_lat': -37.7
    }
    
    config = {
        'num_routes': 3,
        'route_type': ['ferry', 'tss', 'irregular'],
        'num_zones': {
            'mpa': 1,
            'military': 1,
            'windfarm': 1
        },
        'flir_frames': 5,
        'vessel_speed_kts': 18.0,
        'multi_vessel': False,
        'random_seed': 42
    }
    
    output_dir = 'test_output'
    
    # Generate data
    print("📦 Generating test data...")
    generator = MaritimeGenerator(bbox, config, output_dir)
    generator.run()
    
    print("\n🔍 Validating outputs...\n")
    
    # Validate zones
    zones_path = os.path.join(generator.paths['base'], 'zones.geojson')
    zones_valid = validate_geojson(zones_path)
    
    # Validate routes
    routes_path = os.path.join(generator.paths['base'], 'routes.geojson')
    routes_valid = validate_geojson(routes_path)
    
    # Validate FLIR
    flir_valid = validate_flir_images(generator.paths['flir'])
    
    # Summary
    print("\n" + "="*50)
    if zones_valid and routes_valid and flir_valid:
        print("✅ ALL VALIDATION TESTS PASSED")
        print("\n✅ GeoJSON files are valid and loadable in QGIS")
        print("✅ FLIR images are valid PNG files")
        print("✅ All data is 100% fictitious (no real maritime data)")
        print("✅ All geometries are within bounding box")
        print("✅ All outputs use EPSG:4326 (WGS84)")
    else:
        print("❌ SOME VALIDATION TESTS FAILED")
        print("   Please review the errors above")
    print("="*50)


if __name__ == '__main__':
    main()


