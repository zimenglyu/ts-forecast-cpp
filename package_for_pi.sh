#!/bin/bash
# Package essential files for Raspberry Pi transfer
# Creates ts-forecast-pi.tar.gz with only what's needed for evaluation

set -e

echo "=== Packaging for Raspberry Pi ==="

cd "$(dirname "$0")"

# Create temp directory
rm -rf pi_package
mkdir -p pi_package/ts-forecast-cpp

# Copy source code
cp -r include pi_package/ts-forecast-cpp/
cp -r src pi_package/ts-forecast-cpp/
cp CMakeLists.txt pi_package/ts-forecast-cpp/
cp run_evaluation.sh pi_package/ts-forecast-cpp/

# Create benchmark_datasets structure with only essential files
mkdir -p pi_package/ts-forecast-cpp/benchmark_datasets/ETT-small/models
mkdir -p pi_package/ts-forecast-cpp/benchmark_datasets/exchange_rate/models
mkdir -p pi_package/ts-forecast-cpp/benchmark_datasets/illness/models
mkdir -p pi_package/ts-forecast-cpp/benchmark_datasets/weather/models
mkdir -p pi_package/ts-forecast-cpp/benchmark_datasets/electricity/models

# Copy test data (only minmax normalized)
for dataset in ETT-small exchange_rate illness weather electricity; do
    cp benchmark_datasets/$dataset/*_test_minmax.csv pi_package/ts-forecast-cpp/benchmark_datasets/$dataset/ 2>/dev/null || true
done

# Copy trained models (only run0-run9)
for dataset in ETT-small exchange_rate illness weather electricity; do
    cp benchmark_datasets/$dataset/models/*_run*.bin pi_package/ts-forecast-cpp/benchmark_datasets/$dataset/models/ 2>/dev/null || true
    # Also copy scaler
    cp benchmark_datasets/$dataset/models/*_scaler.bin pi_package/ts-forecast-cpp/benchmark_datasets/$dataset/models/ 2>/dev/null || true
done

# Create tarball
cd pi_package
tar -czf ../ts-forecast-pi.tar.gz ts-forecast-cpp
cd ..

# Cleanup
rm -rf pi_package

# Show result
echo ""
echo "Created: ts-forecast-pi.tar.gz"
ls -lh ts-forecast-pi.tar.gz
echo ""
echo "Transfer to Pi and run:"
echo "  scp ts-forecast-pi.tar.gz pi@raspberrypi:~/"
echo "  ssh pi@raspberrypi"
echo "  tar -xzf ts-forecast-pi.tar.gz"
echo "  cd ts-forecast-cpp"
echo "  ./run_evaluation.sh"
