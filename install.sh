#!/bin/bash
# THALES FICTITIOUS GENERATOR - One-Line Installer
# Usage: curl -sSL https://raw.githubusercontent.com/YOUR_USERNAME/thales-cli/main/install.sh | bash

set -e

CYAN='\033[38;2;0;255;255m'
TEAL='\033[38;2;0;206;209m'
GREEN='\033[38;2;0;255;159m'
RESET='\033[0m'

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════╗"
echo "║  THALES FICTITIOUS GENERATOR - Installer     ║"
echo "╚═══════════════════════════════════════════════╝"
echo -e "${RESET}"

# Check if Go is installed
if ! command -v go &> /dev/null; then
    echo -e "${TEAL}ERROR: Go is not installed${RESET}"
    echo ""
    echo "Please install Go first:"
    echo "  macOS:  brew install go"
    echo "  Linux:  sudo apt install golang-go"
    echo "  Or visit: https://go.dev/dl/"
    echo ""
    exit 1
fi

GO_VERSION=$(go version | awk '{print $3}')
echo -e "${GREEN}✓ Found Go: ${GO_VERSION}${RESET}"

# Clone repository
REPO_URL="https://github.com/YOUR_USERNAME/thales-cli.git"
INSTALL_DIR="$HOME/thales-cli"

echo -e "${TEAL}→ Cloning repository...${RESET}"
if [ -d "$INSTALL_DIR" ]; then
    echo -e "${TEAL}  Directory exists, updating...${RESET}"
    cd "$INSTALL_DIR"
    git pull
else
    git clone "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# Install dependencies
echo -e "${TEAL}→ Installing dependencies...${RESET}"
go mod download
go mod tidy

# Build
echo -e "${TEAL}→ Building application...${RESET}"
go build -o thales-fictitious-generator cmd/main.go

echo ""
echo -e "${GREEN}✓ Installation complete!${RESET}"
echo ""
echo "To run the application:"
echo "  cd $INSTALL_DIR"
echo "  ./thales-fictitious-generator"
echo ""
echo "Or install system-wide:"
echo "  sudo cp thales-fictitious-generator /usr/local/bin/"
echo ""
