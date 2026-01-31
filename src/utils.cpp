#include "utils.hpp"
#include <cmath>
#include <sstream>
#include <iomanip>
#include <stdexcept>

namespace ts {

std::string Metrics::to_string() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "MAE:  " << mae << "\n";
    oss << "MSE:  " << mse << "\n";
    oss << "RMSE: " << rmse << "\n";
    oss << "MAPE: " << mape << "%\n";
    oss << "R2:   " << r2;
    return oss.str();
}

namespace stats {

double mean(const std::vector<double>& data) {
    if (data.empty()) return 0.0;
    return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}

double variance(const std::vector<double>& data) {
    if (data.size() < 2) return 0.0;
    double m = mean(data);
    double sum = 0.0;
    for (double x : data) {
        sum += (x - m) * (x - m);
    }
    return sum / (data.size() - 1);
}

double std_dev(const std::vector<double>& data) {
    return std::sqrt(variance(data));
}

double covariance(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.size() < 2) return 0.0;
    double mx = mean(x);
    double my = mean(y);
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sum += (x[i] - mx) * (y[i] - my);
    }
    return sum / (x.size() - 1);
}

double correlation(const std::vector<double>& x, const std::vector<double>& y) {
    double cov = covariance(x, y);
    double sx = std_dev(x);
    double sy = std_dev(y);
    if (sx == 0 || sy == 0) return 0.0;
    return cov / (sx * sy);
}

std::vector<double> acf(const std::vector<double>& data, int max_lag) {
    int n = static_cast<int>(data.size());
    if (max_lag >= n) max_lag = n - 1;

    std::vector<double> result(max_lag + 1);
    double m = mean(data);
    double var = 0.0;

    for (int i = 0; i < n; ++i) {
        var += (data[i] - m) * (data[i] - m);
    }

    for (int lag = 0; lag <= max_lag; ++lag) {
        double sum = 0.0;
        for (int i = 0; i < n - lag; ++i) {
            sum += (data[i] - m) * (data[i + lag] - m);
        }
        result[lag] = sum / var;
    }

    return result;
}

std::vector<double> pacf(const std::vector<double>& data, int max_lag) {
    int n = static_cast<int>(data.size());
    if (max_lag >= n) max_lag = n - 1;

    std::vector<double> result(max_lag + 1);
    std::vector<double> autocorr = acf(data, max_lag);

    result[0] = 1.0;
    if (max_lag >= 1) {
        result[1] = autocorr[1];
    }

    // Durbin-Levinson algorithm
    for (int k = 2; k <= max_lag; ++k) {
        std::vector<double> phi(k + 1);

        double num = autocorr[k];
        double denom = 1.0;

        for (int j = 1; j < k; ++j) {
            num -= result[j] * autocorr[k - j];
            denom -= result[j] * autocorr[j];
        }

        if (std::abs(denom) < 1e-10) {
            result[k] = 0.0;
        } else {
            phi[k] = num / denom;
            result[k] = phi[k];

            for (int j = 1; j < k; ++j) {
                phi[j] = result[j] - phi[k] * result[k - j];
            }

            for (int j = 1; j < k; ++j) {
                result[j] = phi[j];
            }
        }
    }

    // Reset to actual PACF values
    result = acf(data, max_lag);
    result[0] = 1.0;
    if (max_lag >= 1) result[1] = autocorr[1];

    for (int k = 2; k <= max_lag; ++k) {
        std::vector<std::vector<double>> phi(k + 1, std::vector<double>(k + 1, 0.0));
        phi[1][1] = autocorr[1];

        for (int i = 2; i <= k; ++i) {
            double num = autocorr[i];
            double denom = 1.0;
            for (int j = 1; j < i; ++j) {
                num -= phi[i-1][j] * autocorr[i - j];
                denom -= phi[i-1][j] * autocorr[j];
            }
            phi[i][i] = (std::abs(denom) < 1e-10) ? 0.0 : num / denom;

            for (int j = 1; j < i; ++j) {
                phi[i][j] = phi[i-1][j] - phi[i][i] * phi[i-1][i - j];
            }
        }
        result[k] = phi[k][k];
    }

    return result;
}

std::vector<double> difference(const std::vector<double>& data, int order) {
    if (order <= 0) return data;

    std::vector<double> result = data;
    for (int d = 0; d < order; ++d) {
        std::vector<double> temp(result.size() - 1);
        for (size_t i = 1; i < result.size(); ++i) {
            temp[i - 1] = result[i] - result[i - 1];
        }
        result = temp;
    }
    return result;
}

std::vector<double> undifference(const std::vector<double>& diffed,
                                  const std::vector<double>& original,
                                  int order) {
    if (order <= 0) return diffed;

    std::vector<double> result = diffed;
    for (int d = order - 1; d >= 0; --d) {
        std::vector<double> temp(result.size() + 1);
        temp[0] = original[d];
        for (size_t i = 0; i < result.size(); ++i) {
            temp[i + 1] = temp[i] + result[i];
        }
        result = temp;
    }
    return result;
}

std::vector<double> seasonal_difference(const std::vector<double>& data, int period) {
    if (period <= 0 || static_cast<size_t>(period) >= data.size()) return data;

    std::vector<double> result(data.size() - period);
    for (size_t i = period; i < data.size(); ++i) {
        result[i - period] = data[i] - data[i - period];
    }
    return result;
}

} // namespace stats

Metrics evaluate(const std::vector<double>& actual, const std::vector<double>& predicted) {
    if (actual.size() != predicted.size()) {
        throw std::invalid_argument("Actual and predicted vectors must have same size");
    }

    Metrics m;
    int n = static_cast<int>(actual.size());
    if (n == 0) return m;

    double sum_ae = 0.0, sum_se = 0.0, sum_ape = 0.0;
    double mean_actual = stats::mean(actual);
    double ss_tot = 0.0, ss_res = 0.0;

    for (int i = 0; i < n; ++i) {
        double error = actual[i] - predicted[i];
        sum_ae += std::abs(error);
        sum_se += error * error;
        if (std::abs(actual[i]) > 1e-10) {
            sum_ape += std::abs(error / actual[i]);
        }
        ss_tot += (actual[i] - mean_actual) * (actual[i] - mean_actual);
        ss_res += error * error;
    }

    m.mae = sum_ae / n;
    m.mse = sum_se / n;
    m.rmse = std::sqrt(m.mse);
    m.mape = (sum_ape / n) * 100.0;
    m.r2 = (ss_tot > 0) ? (1.0 - ss_res / ss_tot) : 0.0;

    return m;
}

std::pair<std::vector<double>, std::vector<double>>
train_test_split(const std::vector<double>& data, double test_ratio) {
    size_t split_idx = static_cast<size_t>(data.size() * (1.0 - test_ratio));
    std::vector<double> train(data.begin(), data.begin() + split_idx);
    std::vector<double> test(data.begin() + split_idx, data.end());
    return {train, test};
}

DataSplit<std::vector<double>>
train_val_test_split(const std::vector<double>& data,
                     double train_ratio,
                     double val_ratio) {
    if (train_ratio < 0 || val_ratio < 0 || train_ratio + val_ratio > 1.0) {
        throw std::invalid_argument(
            "train_ratio and val_ratio must be non-negative and sum to <= 1.0");
    }

    size_t n = data.size();
    size_t train_end = static_cast<size_t>(n * train_ratio);
    size_t val_end = static_cast<size_t>(n * (train_ratio + val_ratio));

    DataSplit<std::vector<double>> split;
    split.train = std::vector<double>(data.begin(), data.begin() + train_end);
    split.val = std::vector<double>(data.begin() + train_end, data.begin() + val_end);
    split.test = std::vector<double>(data.begin() + val_end, data.end());

    return split;
}

DataSplit<std::vector<std::vector<double>>>
train_val_test_split(const std::vector<std::vector<double>>& data,
                     double train_ratio,
                     double val_ratio) {
    if (train_ratio < 0 || val_ratio < 0 || train_ratio + val_ratio > 1.0) {
        throw std::invalid_argument(
            "train_ratio and val_ratio must be non-negative and sum to <= 1.0");
    }

    size_t n = data.size();
    size_t train_end = static_cast<size_t>(n * train_ratio);
    size_t val_end = static_cast<size_t>(n * (train_ratio + val_ratio));

    DataSplit<std::vector<std::vector<double>>> split;
    split.train = std::vector<std::vector<double>>(data.begin(), data.begin() + train_end);
    split.val = std::vector<std::vector<double>>(data.begin() + train_end, data.begin() + val_end);
    split.test = std::vector<std::vector<double>>(data.begin() + val_end, data.end());

    return split;
}

// MinMaxScaler implementation
void MinMaxScaler::fit(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot fit on empty data");
    }

    double min_val = *std::min_element(data.begin(), data.end());
    double max_val = *std::max_element(data.begin(), data.end());

    min_vals_ = {min_val};
    max_vals_ = {max_val};
    is_fitted_ = true;
}

void MinMaxScaler::fit(const std::vector<std::vector<double>>& data) {
    if (data.empty() || data[0].empty()) {
        throw std::invalid_argument("Cannot fit on empty data");
    }

    size_t n_features = data[0].size();
    min_vals_.resize(n_features);
    max_vals_.resize(n_features);

    for (size_t j = 0; j < n_features; ++j) {
        double min_val = data[0][j];
        double max_val = data[0][j];
        for (size_t i = 1; i < data.size(); ++i) {
            if (data[i][j] < min_val) min_val = data[i][j];
            if (data[i][j] > max_val) max_val = data[i][j];
        }
        min_vals_[j] = min_val;
        max_vals_[j] = max_val;
    }
    is_fitted_ = true;
}

std::vector<double> MinMaxScaler::transform(const std::vector<double>& data) const {
    if (!is_fitted_) {
        throw std::runtime_error("Scaler not fitted. Call fit() first.");
    }

    std::vector<double> result(data.size());
    double range = max_vals_[0] - min_vals_[0];
    if (range < 1e-10) range = 1.0;  // Avoid division by zero

    for (size_t i = 0; i < data.size(); ++i) {
        result[i] = (data[i] - min_vals_[0]) / range;
    }
    return result;
}

std::vector<std::vector<double>> MinMaxScaler::transform(
    const std::vector<std::vector<double>>& data) const {
    if (!is_fitted_) {
        throw std::runtime_error("Scaler not fitted. Call fit() first.");
    }

    std::vector<std::vector<double>> result(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        result[i].resize(data[i].size());
        for (size_t j = 0; j < data[i].size(); ++j) {
            double range = max_vals_[j] - min_vals_[j];
            if (range < 1e-10) range = 1.0;
            result[i][j] = (data[i][j] - min_vals_[j]) / range;
        }
    }
    return result;
}

std::vector<double> MinMaxScaler::fit_transform(const std::vector<double>& data) {
    fit(data);
    return transform(data);
}

std::vector<std::vector<double>> MinMaxScaler::fit_transform(
    const std::vector<std::vector<double>>& data) {
    fit(data);
    return transform(data);
}

std::vector<double> MinMaxScaler::inverse_transform(const std::vector<double>& data) const {
    if (!is_fitted_) {
        throw std::runtime_error("Scaler not fitted. Call fit() first.");
    }

    std::vector<double> result(data.size());
    double range = max_vals_[0] - min_vals_[0];

    for (size_t i = 0; i < data.size(); ++i) {
        result[i] = data[i] * range + min_vals_[0];
    }
    return result;
}

std::vector<std::vector<double>> MinMaxScaler::inverse_transform(
    const std::vector<std::vector<double>>& data) const {
    if (!is_fitted_) {
        throw std::runtime_error("Scaler not fitted. Call fit() first.");
    }

    std::vector<std::vector<double>> result(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        result[i].resize(data[i].size());
        for (size_t j = 0; j < data[i].size(); ++j) {
            double range = max_vals_[j] - min_vals_[j];
            result[i][j] = data[i][j] * range + min_vals_[j];
        }
    }
    return result;
}

// StandardScaler implementation
void StandardScaler::fit(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot fit on empty data");
    }

    means_ = {stats::mean(data)};
    stds_ = {stats::std_dev(data)};
    if (stds_[0] < 1e-10) stds_[0] = 1.0;  // Avoid division by zero
    is_fitted_ = true;
}

void StandardScaler::fit(const std::vector<std::vector<double>>& data) {
    if (data.empty() || data[0].empty()) {
        throw std::invalid_argument("Cannot fit on empty data");
    }

    size_t n_features = data[0].size();
    means_.resize(n_features);
    stds_.resize(n_features);

    for (size_t j = 0; j < n_features; ++j) {
        // Extract column j
        std::vector<double> column(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            column[i] = data[i][j];
        }
        means_[j] = stats::mean(column);
        stds_[j] = stats::std_dev(column);
        if (stds_[j] < 1e-10) stds_[j] = 1.0;
    }
    is_fitted_ = true;
}

std::vector<double> StandardScaler::transform(const std::vector<double>& data) const {
    if (!is_fitted_) {
        throw std::runtime_error("Scaler not fitted. Call fit() first.");
    }

    std::vector<double> result(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        result[i] = (data[i] - means_[0]) / stds_[0];
    }
    return result;
}

std::vector<std::vector<double>> StandardScaler::transform(
    const std::vector<std::vector<double>>& data) const {
    if (!is_fitted_) {
        throw std::runtime_error("Scaler not fitted. Call fit() first.");
    }

    std::vector<std::vector<double>> result(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        result[i].resize(data[i].size());
        for (size_t j = 0; j < data[i].size(); ++j) {
            result[i][j] = (data[i][j] - means_[j]) / stds_[j];
        }
    }
    return result;
}

std::vector<double> StandardScaler::fit_transform(const std::vector<double>& data) {
    fit(data);
    return transform(data);
}

std::vector<std::vector<double>> StandardScaler::fit_transform(
    const std::vector<std::vector<double>>& data) {
    fit(data);
    return transform(data);
}

std::vector<double> StandardScaler::inverse_transform(const std::vector<double>& data) const {
    if (!is_fitted_) {
        throw std::runtime_error("Scaler not fitted. Call fit() first.");
    }

    std::vector<double> result(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        result[i] = data[i] * stds_[0] + means_[0];
    }
    return result;
}

std::vector<std::vector<double>> StandardScaler::inverse_transform(
    const std::vector<std::vector<double>>& data) const {
    if (!is_fitted_) {
        throw std::runtime_error("Scaler not fitted. Call fit() first.");
    }

    std::vector<std::vector<double>> result(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        result[i].resize(data[i].size());
        for (size_t j = 0; j < data[i].size(); ++j) {
            result[i][j] = data[i][j] * stds_[j] + means_[j];
        }
    }
    return result;
}

namespace matrix {

Matrix zeros(size_t rows, size_t cols) {
    return Matrix(rows, std::vector<double>(cols, 0.0));
}

Matrix identity(size_t n) {
    Matrix m = zeros(n, n);
    for (size_t i = 0; i < n; ++i) {
        m[i][i] = 1.0;
    }
    return m;
}

Matrix transpose(const Matrix& m) {
    if (m.empty()) return {};
    size_t rows = m.size();
    size_t cols = m[0].size();
    Matrix result = zeros(cols, rows);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[j][i] = m[i][j];
        }
    }
    return result;
}

Matrix multiply(const Matrix& a, const Matrix& b) {
    if (a.empty() || b.empty() || a[0].size() != b.size()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }

    size_t rows = a.size();
    size_t cols = b[0].size();
    size_t inner = b.size();
    Matrix result = zeros(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            for (size_t k = 0; k < inner; ++k) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return result;
}

Vector multiply(const Matrix& a, const Vector& v) {
    if (a.empty() || a[0].size() != v.size()) {
        throw std::invalid_argument("Matrix and vector dimensions don't match");
    }

    Vector result(a.size(), 0.0);
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < v.size(); ++j) {
            result[i] += a[i][j] * v[j];
        }
    }
    return result;
}

Matrix inverse(const Matrix& m) {
    size_t n = m.size();
    if (n == 0 || m[0].size() != n) {
        throw std::invalid_argument("Matrix must be square for inversion");
    }

    // Augmented matrix [m | I]
    Matrix aug(n, std::vector<double>(2 * n));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            aug[i][j] = m[i][j];
        }
        aug[i][n + i] = 1.0;
    }

    // Gauss-Jordan elimination
    for (size_t col = 0; col < n; ++col) {
        // Find pivot
        size_t pivot = col;
        for (size_t row = col + 1; row < n; ++row) {
            if (std::abs(aug[row][col]) > std::abs(aug[pivot][col])) {
                pivot = row;
            }
        }

        if (std::abs(aug[pivot][col]) < 1e-10) {
            throw std::runtime_error("Matrix is singular");
        }

        std::swap(aug[col], aug[pivot]);

        // Scale pivot row
        double scale = aug[col][col];
        for (size_t j = 0; j < 2 * n; ++j) {
            aug[col][j] /= scale;
        }

        // Eliminate column
        for (size_t row = 0; row < n; ++row) {
            if (row != col) {
                double factor = aug[row][col];
                for (size_t j = 0; j < 2 * n; ++j) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }

    // Extract inverse
    Matrix result(n, std::vector<double>(n));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result[i][j] = aug[i][n + j];
        }
    }
    return result;
}

Vector solve(const Matrix& A, const Vector& b) {
    size_t n = A.size();
    if (n == 0 || A[0].size() != n || b.size() != n) {
        throw std::invalid_argument("Invalid dimensions for linear solve");
    }

    // LU decomposition with partial pivoting
    Matrix L = zeros(n, n);
    Matrix U = A;
    std::vector<size_t> perm(n);
    for (size_t i = 0; i < n; ++i) perm[i] = i;

    for (size_t k = 0; k < n; ++k) {
        // Find pivot
        size_t pivot = k;
        for (size_t i = k + 1; i < n; ++i) {
            if (std::abs(U[i][k]) > std::abs(U[pivot][k])) {
                pivot = i;
            }
        }

        if (std::abs(U[pivot][k]) < 1e-10) {
            throw std::runtime_error("Matrix is singular in solve");
        }

        std::swap(U[k], U[pivot]);
        std::swap(L[k], L[pivot]);
        std::swap(perm[k], perm[pivot]);

        L[k][k] = 1.0;
        for (size_t i = k + 1; i < n; ++i) {
            L[i][k] = U[i][k] / U[k][k];
            for (size_t j = k; j < n; ++j) {
                U[i][j] -= L[i][k] * U[k][j];
            }
        }
    }
    L[n-1][n-1] = 1.0;

    // Permute b
    Vector pb(n);
    for (size_t i = 0; i < n; ++i) {
        pb[i] = b[perm[i]];
    }

    // Forward substitution: Ly = pb
    Vector y(n);
    for (size_t i = 0; i < n; ++i) {
        y[i] = pb[i];
        for (size_t j = 0; j < i; ++j) {
            y[i] -= L[i][j] * y[j];
        }
    }

    // Backward substitution: Ux = y
    Vector x(n);
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        x[i] = y[i];
        for (size_t j = i + 1; j < n; ++j) {
            x[i] -= U[i][j] * x[j];
        }
        x[i] /= U[i][i];
    }

    return x;
}

Matrix cholesky(const Matrix& m) {
    size_t n = m.size();
    Matrix L = zeros(n, n);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            if (i == j) {
                for (size_t k = 0; k < j; ++k) {
                    sum += L[j][k] * L[j][k];
                }
                double val = m[j][j] - sum;
                if (val <= 0) {
                    throw std::runtime_error("Matrix not positive definite");
                }
                L[j][j] = std::sqrt(val);
            } else {
                for (size_t k = 0; k < j; ++k) {
                    sum += L[i][k] * L[j][k];
                }
                if (std::abs(L[j][j]) < 1e-10) {
                    L[i][j] = 0.0;
                } else {
                    L[i][j] = (m[i][j] - sum) / L[j][j];
                }
            }
        }
    }
    return L;
}

} // namespace matrix

} // namespace ts
