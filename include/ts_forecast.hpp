#ifndef TS_FORECAST_HPP
#define TS_FORECAST_HPP

/**
 * Time Series Forecasting Library
 *
 * Lightweight C++ implementations of common time series forecasting models
 * designed for embedded systems like Raspberry Pi.
 *
 * Models included:
 * - ARIMA (AutoRegressive Integrated Moving Average)
 * - Exponential Smoothing (Simple, Holt, Holt-Winters)
 * - Prophet-like additive model
 * - Gradient Boosting for time series
 */

#include "utils.hpp"
#include "arima.hpp"
#include "exponential_smoothing.hpp"
#include "prophet.hpp"
#include "gradient_boosting.hpp"
#include "dlinear.hpp"

#endif // TS_FORECAST_HPP
