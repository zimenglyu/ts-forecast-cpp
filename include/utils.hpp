#ifndef TS_UTILS_HPP
#define TS_UTILS_HPP

#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <random>
#include <fstream>
#include <cstdint>

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

/**
 * Data split result for train/validation/test sets
 */
template<typename T>
struct DataSplit {
    T train;
    T val;
    T test;
};

/**
 * Train/validation/test split for univariate time series
 * @param data Input time series data
 * @param train_ratio Ratio of data for training (default 0.7)
 * @param val_ratio Ratio of data for validation (default 0.15)
 * @return DataSplit containing train, val, test vectors
 * Note: test_ratio = 1.0 - train_ratio - val_ratio
 */
DataSplit<std::vector<double>>
train_val_test_split(const std::vector<double>& data,
                     double train_ratio = 0.7,
                     double val_ratio = 0.15);

/**
 * Train/validation/test split for multivariate time series
 * @param data Input multivariate data (rows = time, cols = features)
 * @param train_ratio Ratio of data for training (default 0.7)
 * @param val_ratio Ratio of data for validation (default 0.15)
 * @return DataSplit containing train, val, test matrices
 */
DataSplit<std::vector<std::vector<double>>>
train_val_test_split(const std::vector<std::vector<double>>& data,
                     double train_ratio = 0.7,
                     double val_ratio = 0.15);

/**
 * Min-Max Scaler: scales data to [0, 1] range
 *
 * Formula: x_scaled = (x - min) / (max - min)
 *
 * Use for:
 * - Neural network inputs (bounded activation functions)
 * - When you need values in a fixed range [0, 1]
 * - Preserves zero values and sparsity structure
 */
class MinMaxScaler {
public:
    MinMaxScaler() : is_fitted_(false) {}

    /**
     * Fit the scaler on univariate data
     * @param data Data to compute min/max from
     */
    void fit(const std::vector<double>& data);

    /**
     * Fit the scaler on multivariate data (each column independently)
     * @param data Multivariate data (rows = samples, cols = features)
     */
    void fit(const std::vector<std::vector<double>>& data);

    /**
     * Transform univariate data using fitted parameters
     */
    std::vector<double> transform(const std::vector<double>& data) const;

    /**
     * Transform multivariate data using fitted parameters
     */
    std::vector<std::vector<double>> transform(
        const std::vector<std::vector<double>>& data) const;

    /**
     * Fit and transform in one step
     */
    std::vector<double> fit_transform(const std::vector<double>& data);
    std::vector<std::vector<double>> fit_transform(
        const std::vector<std::vector<double>>& data);

    /**
     * Inverse transform to original scale
     */
    std::vector<double> inverse_transform(const std::vector<double>& data) const;
    std::vector<std::vector<double>> inverse_transform(
        const std::vector<std::vector<double>>& data) const;

    bool is_fitted() const { return is_fitted_; }
    double min_val() const { return min_vals_.empty() ? 0.0 : min_vals_[0]; }
    double max_val() const { return max_vals_.empty() ? 1.0 : max_vals_[0]; }
    const std::vector<double>& min_vals() const { return min_vals_; }
    const std::vector<double>& max_vals() const { return max_vals_; }

    /**
     * Save scaler to binary file
     */
    void save(const std::string& filename) const;

    /**
     * Load scaler from binary file
     */
    void load(const std::string& filename);

private:
    std::vector<double> min_vals_;
    std::vector<double> max_vals_;
    bool is_fitted_;
};

/**
 * Standard Scaler: standardizes data to zero mean and unit variance
 *
 * Formula: x_scaled = (x - mean) / std
 *
 * Use for:
 * - Data with Gaussian distribution
 * - Algorithms sensitive to feature magnitudes (linear models, SVMs)
 * - When outliers should retain their relative importance
 */
class StandardScaler {
public:
    StandardScaler() : is_fitted_(false) {}

    /**
     * Fit the scaler on univariate data
     * @param data Data to compute mean/std from
     */
    void fit(const std::vector<double>& data);

    /**
     * Fit the scaler on multivariate data (each column independently)
     * @param data Multivariate data (rows = samples, cols = features)
     */
    void fit(const std::vector<std::vector<double>>& data);

    /**
     * Transform univariate data using fitted parameters
     */
    std::vector<double> transform(const std::vector<double>& data) const;

    /**
     * Transform multivariate data using fitted parameters
     */
    std::vector<std::vector<double>> transform(
        const std::vector<std::vector<double>>& data) const;

    /**
     * Fit and transform in one step
     */
    std::vector<double> fit_transform(const std::vector<double>& data);
    std::vector<std::vector<double>> fit_transform(
        const std::vector<std::vector<double>>& data);

    /**
     * Inverse transform to original scale
     */
    std::vector<double> inverse_transform(const std::vector<double>& data) const;
    std::vector<std::vector<double>> inverse_transform(
        const std::vector<std::vector<double>>& data) const;

    bool is_fitted() const { return is_fitted_; }
    double mean() const { return means_.empty() ? 0.0 : means_[0]; }
    double std() const { return stds_.empty() ? 1.0 : stds_[0]; }
    const std::vector<double>& means() const { return means_; }
    const std::vector<double>& stds() const { return stds_; }

    /**
     * Save scaler to binary file
     */
    void save(const std::string& filename) const;

    /**
     * Load scaler from binary file
     */
    void load(const std::string& filename);

private:
    std::vector<double> means_;
    std::vector<double> stds_;
    bool is_fitted_;
};

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
