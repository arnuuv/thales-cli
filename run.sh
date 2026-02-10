#!/bin/bash

# THALES FICTITIOUS GENERATOR - Quick Run Script
# ===============================================

set -e

BINARY="thales-fictitious-generator"
CYAN='\033[38;2;0;255;255m'
TEAL='\033[38;2;0;206;209m'
PURPLE='\033[38;2;147;112;219m'
GREEN='\033[38;2;0;255;159m'
RESET='\033[0m'

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════╗"
echo "║  THALES FICTITIOUS GENERATOR - Quick Start   ║"
echo "╚═══════════════════════════════════════════════╝"
echo -e "${RESET}"

# Check if Go is installed
if ! command -v go &> /dev/null; then
    echo -e "${PURPLE}ERROR: Go is not installed or not in PATH${RESET}"
    echo ""
    echo "Please install Go 1.22 or higher:"
    echo "  macOS: brew install go"
    echo "  Linux: https://go.dev/dl/"
    echo ""
    exit 1
fi

GO_VERSION=$(go version | awk '{print $3}')
echo -e "${TEAL}✓ Found Go: ${GO_VERSION}${RESET}"

# Check if dependencies need to be downloaded
if [ ! -f "go.sum" ] || [ ! -d "vendor" ] && [ ! -f "$BINARY" ]; then
    echo -e "${TEAL}→ Downloading dependencies...${RESET}"
    go mod download
    go mod tidy
fi

# Build if binary doesn't exist or source is newer
if [ ! -f "$BINARY" ] || [ "cmd/main.go" -nt "$BINARY" ]; then
    echo -e "${TEAL}→ Building ${BINARY}...${RESET}"
    go build -o "$BINARY" cmd/main.go
    echo -e "${GREEN}✓ Build complete!${RESET}"
else
    echo -e "${GREEN}✓ Binary is up to date${RESET}"
fi

echo ""
echo -e "${CYAN}→ Launching THALES FICTITIOUS GENERATOR...${RESET}"
echo ""
sleep 0.5

# Run the application
./"$BINARY"

# Show cursor on exit (in case of crash)
echo -e "\033[?25h"
