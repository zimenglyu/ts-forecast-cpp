# Time Series Forecasting Library (C++)

A lightweight C++ library for time series forecasting, designed for embedded systems like Raspberry Pi. No external dependencies required.

## Features

- **ARIMA** - AutoRegressive Integrated Moving Average
  - ARIMA(p,d,q) models
  - SARIMA for seasonal data
  - Auto ARIMA for automatic order selection

- **Exponential Smoothing**
  - Simple Exponential Smoothing (SES)
  - Holt's Linear Trend Method
  - Holt-Winters Seasonal Method (additive and multiplicative)
  - ETS state-space models

- **Prophet-like Model**
  - Additive decomposition (trend + seasonality)
  - Piecewise linear/logistic growth
  - Custom seasonalities via Fourier series
  - Changepoint detection

- **Gradient Boosting** (XGBoost-like)
  - Decision tree ensembles
  - Time series wrapper with lag features
  - Rolling statistics features
  - Multiple loss functions (MSE, MAE, Huber)

- **DLinear** (Zeng et al., 2022)
  - Trend-seasonal decomposition with linear layers
  - NLinear - handles distribution shift
  - Linear - simple baseline

- **Data Preprocessing**
  - Train/validation/test split (configurable ratios)
  - MinMax normalization [0, 1]
  - Standard normalization (z-score)
  - Binary model save/load

## Requirements

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.10+
- **No external dependencies** - uses only C++ standard library

## Compilation

```bash
# Clone the repository
git clone https://github.com/zimenglyu/ts-forecast-cpp.git
cd ts-forecast-cpp

# Create build directory and compile
mkdir build && cd build
cmake ..
make -j4

# For Raspberry Pi optimization
cmake -DBUILD_FOR_RPI=ON ..
make -j4
```

## Running Examples and Tests

```bash
cd build

# Run examples
./example_arima
./example_exponential_smoothing
./example_prophet
./example_gradient_boosting
./demo

# Run tests
./run_tests          # Unit tests
./test_etth1         # ETTh1 dataset benchmark
./test_save_load     # Save/load functionality
```

## Usage

### Basic Forecasting

```cpp
#include "ts_forecast.hpp"

std::vector<double> data = {/* your time series */};

// ARIMA
ts::ARIMA arima(2, 1, 1);  // p=2, d=1, q=1
arima.fit(data);
auto forecast = arima.forecast(24);

// Holt-Winters
ts::HoltWinters hw(24, ts::HoltWinters::SeasonalType::ADDITIVE);
hw.fit(data);
auto forecast = hw.forecast(48);

// DLinear
ts::DLinear model(96, 24);  // seq_len=96, pred_len=24
model.fit(data, 100, 0.001, 32);  // epochs, lr, batch_size
auto forecast = model.forecast(24);
```

### Data Preprocessing

```cpp
#include "ts_forecast.hpp"

// Train/Validation/Test Split (70/15/15)
auto split = ts::train_val_test_split(data, 0.7, 0.15);
// Access: split.train, split.val, split.test

// Standard Normalization (fit on train only!)
ts::StandardScaler scaler;
scaler.fit(split.train);
auto train_norm = scaler.transform(split.train);
auto val_norm = scaler.transform(split.val);
auto test_norm = scaler.transform(split.test);

// Inverse transform predictions back to original scale
auto predictions_original = scaler.inverse_transform(predictions);

// MinMax Normalization [0, 1]
ts::MinMaxScaler minmax;
minmax.fit(split.train);
auto scaled = minmax.transform(data);
```

### Save and Load Models

```cpp
// Train and save
ts::StandardScaler scaler;
scaler.fit(train_data);
scaler.save("scaler.bin");

ts::DLinear model(96, 24);
model.fit(data);
model.save("model.bin");

// Later: load and predict
ts::StandardScaler loaded_scaler;
loaded_scaler.load("scaler.bin");

ts::DLinear loaded_model;
loaded_model.load("model.bin");

auto pred = loaded_model.predict(input);
auto result = loaded_scaler.inverse_transform(pred);
```

### Model Evaluation

```cpp
ts::Metrics metrics = ts::evaluate(actual, predicted);
std::cout << "RMSE: " << metrics.rmse << std::endl;
std::cout << "MAE:  " << metrics.mae << std::endl;
std::cout << "MAPE: " << metrics.mape << "%" << std::endl;
std::cout << "R2:   " << metrics.r2 << std::endl;
```

## Project Structure

```
ts-forecast-cpp/
├── include/
│   ├── ts_forecast.hpp          # Main header (includes all)
│   ├── arima.hpp
│   ├── exponential_smoothing.hpp
│   ├── prophet.hpp
│   ├── gradient_boosting.hpp
│   ├── dlinear.hpp
│   ├── utils.hpp                # Scalers, metrics, utilities
│   └── csv_reader.hpp
├── src/
│   ├── arima.cpp
│   ├── exponential_smoothing.cpp
│   ├── prophet.cpp
│   ├── gradient_boosting.cpp
│   ├── dlinear.cpp
│   └── utils.cpp
├── examples/
│   └── *.cpp
├── tests/
│   └── *.cpp
└── CMakeLists.txt
```

## Integration with Your Project

### As a Static/Shared Library

```cmake
# In your CMakeLists.txt
add_subdirectory(ts-forecast-cpp)
target_link_libraries(your_project ts_forecast)
```

### Copy Source Files

Copy `include/` and `src/` directories to your project and add to your build system.

## API Reference

### ForecastResult

```cpp
struct ForecastResult {
    std::vector<double> predictions;   // Point forecasts
    std::vector<double> lower_bound;   // Lower confidence interval
    std::vector<double> upper_bound;   // Upper confidence interval
    double confidence_level;           // Default 0.95
};
```

### Metrics

```cpp
struct Metrics {
    double mae;   // Mean Absolute Error
    double mse;   // Mean Squared Error
    double rmse;  // Root Mean Squared Error
    double mape;  // Mean Absolute Percentage Error
    double r2;    // R-squared
};
```

## Tested Datasets

Verified on standard time series benchmarks:
- ETTh1, ETTh2, ETTm1, ETTm2 (Electricity Transformer Temperature)
- Electricity (321 variables)
- Exchange Rate (8 currencies)
- Traffic (862 sensors)
- Weather (21 features)
- ILI (Influenza-like Illness)

## Citation

If you use this library in your research, please cite:

```bibtex
@software{ts_forecast_cpp,
  author = {Lyu, Zimeng},
  title = {Time Series Forecasting Library for C++},
  year = {2025},
  url = {https://github.com/zimenglyu/ts-forecast-cpp}
}
```

For DLinear models, please also cite the original paper:

```bibtex
@inproceedings{zeng2023dlinear,
  title = {Are Transformers Effective for Time Series Forecasting?},
  author = {Zeng, Ailing and Chen, Muxi and Zhang, Lei and Xu, Qiang},
  booktitle = {AAAI Conference on Artificial Intelligence},
  year = {2023}
}
```

## License

MIT License
