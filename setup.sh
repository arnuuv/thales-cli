#!/bin/bash
# Setup script for Thales CLI
# Installs Python dependencies and builds the Go executable

set -e

echo "🔧 Setting up Thales Fictitious Generator..."
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    echo "   Please install Python 3 first: https://www.python.org/downloads/"
    exit 1
fi

# Check if Go is installed
if ! command -v go &> /dev/null; then
    echo "❌ Error: Go is not installed"
    echo "   Please install Go first: https://golang.org/dl/"
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo "✓ Go found: $(go version)"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment and install dependencies
echo ""
echo "📦 Installing Python dependencies in virtual environment..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "🔨 Building Go executable..."
go build -o thales-fictitious-generator ./cmd/main.go

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To run the CLI:"
echo "   ./run.sh"
echo ""
echo "   (The run.sh script will automatically activate the virtual environment)"
echo ""
