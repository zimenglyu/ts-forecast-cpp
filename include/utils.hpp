#ifndef TS_UTILS_HPP
#define TS_UTILS_HPP

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <random>

namespace ts {

/**
 * Time series data point with optional timestamp
 */
struct DataPoint {
    double timestamp;  // Can be index or actual timestamp
    double value;

    DataPoint() : timestamp(0), value(0) {}
    DataPoint(double t, double v) : timestamp(t), value(v) {}
};

/**
 * Forecast result with confidence intervals
 */
struct ForecastResult {
    std::vector<double> predictions;
    std::vector<double> lower_bound;  // Lower confidence interval
    std::vector<double> upper_bound;  // Upper confidence interval
    double confidence_level;

    ForecastResult() : confidence_level(0.95) {}
};

/**
 * Model evaluation metrics
 */
struct Metrics {
    double mae;   // Mean Absolute Error
    double mse;   // Mean Squared Error
    double rmse;  // Root Mean Squared Error
    double mape;  // Mean Absolute Percentage Error
    double r2;    // R-squared

    std::string to_string() const;
};

// Statistical utility functions
namespace stats {

double mean(const std::vector<double>& data);
double variance(const std::vector<double>& data);
double std_dev(const std::vector<double>& data);
double covariance(const std::vector<double>& x, const std::vector<double>& y);
double correlation(const std::vector<double>& x, const std::vector<double>& y);

// Autocorrelation function
std::vector<double> acf(const std::vector<double>& data, int max_lag);

// Partial autocorrelation function
std::vector<double> pacf(const std::vector<double>& data, int max_lag);

// Differencing for stationarity
std::vector<double> difference(const std::vector<double>& data, int order = 1);

// Inverse differencing
std::vector<double> undifference(const std::vector<double>& diffed,
                                  const std::vector<double>& original,
                                  int order = 1);

// Seasonal differencing
std::vector<double> seasonal_difference(const std::vector<double>& data, int period);

} // namespace stats

// Model evaluation functions
Metrics evaluate(const std::vector<double>& actual, const std::vector<double>& predicted);

// Train/test split
std::pair<std::vector<double>, std::vector<double>>
train_test_split(const std::vector<double>& data, double test_ratio = 0.2);

// Simple matrix operations (for internal use)
namespace matrix {

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

Matrix zeros(size_t rows, size_t cols);
Matrix identity(size_t n);
Matrix transpose(const Matrix& m);
Matrix multiply(const Matrix& a, const Matrix& b);
Vector multiply(const Matrix& a, const Vector& v);
Matrix inverse(const Matrix& m);

// Solve linear system Ax = b using LU decomposition
Vector solve(const Matrix& A, const Vector& b);

// Cholesky decomposition for positive definite matrices
Matrix cholesky(const Matrix& m);

} // namespace matrix

} // namespace ts

#endif // TS_UTILS_HPP
