#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
BUILD_TYPE="native"
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --wasm)
            BUILD_TYPE="wasm"
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --wasm    Build WebAssembly version (requires Emscripten)"
            echo "  --clean   Clean build directory before building"
            echo "  --help    Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create dist directory
mkdir -p dist

if [ "$BUILD_TYPE" = "wasm" ]; then
    print_status "Building WebAssembly version..."

    # Check for Emscripten
    if ! command -v emcc &> /dev/null; then
        print_error "Emscripten (emcc) not found!"
        echo "Please install Emscripten:"
        echo "  brew install emscripten  # macOS"
        echo "  Or follow: https://emscripten.org/docs/getting_started/downloads.html"
        exit 1
    fi

    BUILD_DIR="build-wasm"

    if [ "$CLEAN" = true ] && [ -d "$BUILD_DIR" ]; then
        print_status "Cleaning $BUILD_DIR..."
        rm -rf "$BUILD_DIR"
    fi

    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    print_status "Configuring with Emscripten..."
    emcmake cmake .. -DBUILD_WASM=ON -DCMAKE_BUILD_TYPE=Release

    print_status "Building..."
    emmake make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)

    # Add ES module export for browser compatibility
    cd "$SCRIPT_DIR"
    print_status "Adding ES module export..."
    echo "" >> dist/cqt.js
    echo "export default createCQTModule;" >> dist/cqt.js

    print_status "WebAssembly build complete!"
    print_status "Output files:"
    echo "  - dist/cqt.js"
    echo "  - dist/cqt.wasm"

else
    print_status "Building native version..."

    BUILD_DIR="build"

    if [ "$CLEAN" = true ] && [ -d "$BUILD_DIR" ]; then
        print_status "Cleaning $BUILD_DIR..."
        rm -rf "$BUILD_DIR"
    fi

    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    print_status "Configuring..."
    cmake .. -DCMAKE_BUILD_TYPE=Release

    print_status "Building..."
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)

    print_status "Native build complete!"
    print_status "Running tests..."

    ./test_cqt

    print_status "Tests completed!"
fi
