THALES FICTITIOUS GENERATOR
===========================

A premium, cinematic terminal UI for a Thales-style synthetic imagery system.

INSTALLATION FROM GITHUB
-------------------------

Method 1 - One-line installer (easiest):
    $ curl -sSL https://raw.githubusercontent.com/YOUR_USERNAME/thales-cli/main/install.sh | bash

Method 2 - Manual clone:
    $ git clone https://github.com/YOUR_USERNAME/thales-cli.git
    $ cd thales-cli
    $ ./run.sh

Method 3 - Step by step:
    $ git clone https://github.com/YOUR_USERNAME/thales-cli.git
    $ cd thales-cli
    $ go mod download
    $ go build -o thales-fictitious-generator cmd/main.go
    $ ./thales-fictitious-generator

For detailed installation instructions, see INSTALL.txt

FEATURES
--------
✓ Stunning RGB gradient ASCII title with cinematic startup
✓ Smooth 30 FPS animations with rotating 3D-illusion objects
✓ Real-time terrain map and maritime ship visualizations
✓ Dynamic focus with alternating object emphasis (every 5 seconds)
✓ Aerospace/defense-grade UI aesthetics
✓ Responsive layout with dynamic centering
✓ Full keyboard navigation support

QUICK START
-----------

1. Install dependencies:
   make deps

2. Build the application:
   make build

3. Run the application:
   make run

   OR simply:
   ./thales-fictitious-generator

CONTROLS
--------
• [1] - Select Terrain & Map Imagery domain
• [2] - Select Maritime Imagery domain
• Arrow Keys - Navigate menu options
• Enter - Confirm selection
• [Q] - Quit application

PROJECT STRUCTURE
-----------------
cmd/
  main.go           - Application entry point and Bubble Tea setup

ui/
  colors.go         - RGB color system with 24-bit ANSI support
  layout.go         - Main layout rendering and centering logic

ascii/
  map.go            - Rotating terrain/map ASCII frames
  ship.go           - Rotating maritime ship ASCII frames

animation/
  ticker.go         - Animation timing system
  rotation.go       - Rotation state and frame management

state/
  app_state.go      - Application state management

input/
  keys.go           - Keyboard input handling

TECHNICAL DETAILS
-----------------
• Language: Go 1.22+
• UI Framework: Bubble Tea + Lip Gloss
• Animation: Time-based frame cycling (30 FPS render, 10 FPS rotation)
• Color System: 24-bit RGB ANSI escape sequences
• Terminal: Supports modern terminals with truecolor support

REQUIREMENTS
------------
• Go 1.22 or higher
• Terminal with 24-bit color support (iTerm2, modern Terminal.app, etc.)
• Minimum terminal size: 80x30 characters

COLOR PALETTE
-------------
The UI uses a neon RGB theme:
• Cyan (#00FFFF) - Primary highlights
• Teal (#00CED1) - Terrain/map elements
• Purple (#9370DB) - Accent elements
• Electric Blue (#7DF9FF) - Maritime/active elements
• Soft Green (#00FF9F) - Status indicators

DEVELOPMENT
-----------
Run in development mode (no build step):
  make dev

Clean build artifacts:
  make clean

Install system-wide:
  make install

BUILD COMMANDS
--------------
• make deps    - Download and tidy Go dependencies
• make build   - Build the binary
• make run     - Build and run
• make clean   - Remove build artifacts
• make install - Install to /usr/local/bin (requires sudo)
• make dev     - Run directly without building

NOTES
-----
This is a UI-only prototype. No actual imagery generation, GIS logic, 
or Python execution is implemented yet. The tool demonstrates the 
premium terminal interface that will be used for the full system.

The animations are smooth and continuous, creating an immersive 
experience comparable to modern defense/aerospace internal tools.

FUTURE ENHANCEMENTS
-------------------
• Integration with actual imagery generation systems
• GIS data processing and rendering
• Maritime vessel tracking integration
• Export capabilities for generated imagery
• Configuration system for resolution and parameters
