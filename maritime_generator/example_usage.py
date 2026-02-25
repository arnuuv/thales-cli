"""
Example usage of the Maritime Generator

This script demonstrates how to use the maritime generator
to create fictitious maritime data.
"""

from maritime_generator import MaritimeGenerator


def main():
    # Example bounding box (Tasman Sea area)
    bbox = {
        'min_lon': 148.0,
        'min_lat': -37.8,
        'max_lon': 148.1,
        'max_lat': -37.7
    }
    
    # Example configuration
    config = {
        'num_routes': 5,
        'route_type': ['ferry', 'tss', 'irregular'],
        'num_zones': {
            'mpa': 2,
            'military': 1,
            'windfarm': 1
        },
        'flir_frames': 20,
        'vessel_speed_kts': 18.0,
        'multi_vessel': True,
        'random_seed': 42
    }
    
    # Create generator
    generator = MaritimeGenerator(
        bbox=bbox,
        config=config,
        output_dir='output'
    )
    
    # Run generation
    generator.run()
    
    print("\n✅ Maritime data generation complete!")
    print(f"📁 Output directory: {generator.paths['base']}")
    print(f"   - zones.geojson")
    print(f"   - routes.geojson")
    print(f"   - flir/ (with {config['flir_frames']} frames)")


if __name__ == '__main__':
    main()




