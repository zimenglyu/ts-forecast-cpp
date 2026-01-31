# Time Series Forecasting Library (C++)

A lightweight, header-only C++ library for time series forecasting, designed to run on embedded systems like Raspberry Pi.

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
  - Additive decomposition (trend + seasonality + holidays)
  - Piecewise linear/logistic growth
  - Custom seasonalities via Fourier series
  - Changepoint detection

- **Gradient Boosting** (XGBoost-like)
  - Decision tree ensembles
  - Time series specific wrapper with lag features
  - Rolling statistics features
  - Multiple loss functions (MSE, MAE, Huber)

## Requirements

- C++17 compatible compiler
- CMake 3.10+
- No external dependencies (uses only standard library)

## Building

```bash
mkdir build && cd build
cmake ..
make
```

### For Raspberry Pi

```bash
mkdir build && cd build
cmake -DBUILD_FOR_RPI=ON ..
make
```

### Cross-compiling for Raspberry Pi

```bash
cmake -DCMAKE_TOOLCHAIN_FILE=/path/to/rpi-toolchain.cmake -DBUILD_FOR_RPI=ON ..
make
```

## Usage

### ARIMA

```cpp
#include "ts_forecast.hpp"

std::vector<double> data = {/* your time series */};

// Fit ARIMA(2,1,1)
ts::ARIMA model(2, 1, 1);
model.fit(data);

// Forecast 10 steps ahead
ts::ForecastResult forecast = model.forecast(10);
for (double pred : forecast.predictions) {
    std::cout << pred << std::endl;
}

// Auto ARIMA
ts::AutoARIMA auto_model(5, 2, 5);  // max orders
auto_model.fit(data);
std::cout << "Selected: ARIMA(" << auto_model.selected_p()
          << "," << auto_model.selected_d()
          << "," << auto_model.selected_q() << ")" << std::endl;
```

### Exponential Smoothing

```cpp
#include "ts_forecast.hpp"

std::vector<double> data = {/* monthly data with seasonality */};

// Holt-Winters with monthly seasonality
ts::HoltWinters model(12, ts::HoltWinters::SeasonalType::ADDITIVE);
model.fit(data);

ts::ForecastResult forecast = model.forecast(12);
```

### Prophet

```cpp
#include "ts_forecast.hpp"

std::vector<double> data = {/* daily data */};

ts::Prophet model(ts::Prophet::GrowthType::LINEAR, true, true, false);
model.fit(data);

// Add custom seasonality
model.add_seasonality("monthly", 30.5, 5);

ts::ForecastResult forecast = model.forecast(30);
```

### Gradient Boosting

```cpp
#include "ts_forecast.hpp"

std::vector<double> data = {/* your time series */};

ts::TimeSeriesGradientBoosting model(7, 100, 6, 0.1);  // 7 lags, 100 trees
model.add_rolling_features({7, 14});
model.fit(data);

ts::ForecastResult forecast = model.forecast(14);
```

### Evaluation

```cpp
#include "ts_forecast.hpp"

std::vector<double> actual = {/* actual values */};
std::vector<double> predicted = {/* predicted values */};

ts::Metrics metrics = ts::evaluate(actual, predicted);
std::cout << "RMSE: " << metrics.rmse << std::endl;
std::cout << "MAPE: " << metrics.mape << "%" << std::endl;
```

## Running Examples

```bash
./example_arima
./example_exponential_smoothing
./example_prophet
./example_gradient_boosting
./demo
```

## Running Tests

```bash
./run_tests
```

## API Reference

### Forecast Result

All models return a `ForecastResult` struct:

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

## License

MIT License
