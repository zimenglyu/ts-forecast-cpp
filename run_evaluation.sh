#!/bin/bash
# Run evaluation on Raspberry Pi
# Usage: ./run_evaluation.sh

set -e

echo "=== Time Series Model Evaluation ==="
echo ""

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: Run this script from the ts-forecast-cpp root directory"
    exit 1
fi

# Create build directory if needed
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir build
fi

cd build

# Build for Raspberry Pi (or regular build)
if [ "$(uname -m)" = "aarch64" ] || [ "$(uname -m)" = "armv7l" ]; then
    echo "Detected ARM architecture - building with Pi optimizations..."
    cmake -DBUILD_FOR_RPI=ON ..
else
    echo "Building for current platform..."
    cmake ..
fi

echo ""
echo "Compiling..."
make evaluate_models -j$(nproc 2>/dev/null || echo 4)

echo ""
echo "Running evaluation..."
echo ""
./evaluate_models

echo ""
echo "=== Done ==="
echo "Results saved to benchmark_datasets/evaluation_*.csv"
