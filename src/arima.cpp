#include "arima.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace ts {

ARIMA::ARIMA(int p, int d, int q)
    : p_(p), d_(d), q_(q), c_(0), sigma_(1.0), fitted_(false) {
    if (p < 0 || d < 0 || q < 0) {
        throw std::invalid_argument("ARIMA orders must be non-negative");
    }
    phi_.resize(p_, 0.0);
    theta_.resize(q_, 0.0);
}

void ARIMA::fit(const std::vector<double>& data, int max_iter, double tol) {
    if (data.size() < static_cast<size_t>(p_ + d_ + q_ + 1)) {
        throw std::invalid_argument("Insufficient data for ARIMA model");
    }

    data_ = data;
    diff_data_ = stats::difference(data, d_);

    estimate_parameters();

    // Iterative refinement using conditional sum of squares
    double prev_sse = std::numeric_limits<double>::max();
    for (int iter = 0; iter < max_iter; ++iter) {
        residuals_ = compute_residuals();

        double sse = 0.0;
        for (double r : residuals_) {
            sse += r * r;
        }

        if (std::abs(prev_sse - sse) < tol) {
            break;
        }
        prev_sse = sse;

        // Update MA coefficients using innovations
        if (q_ > 0) {
            for (int j = 0; j < q_; ++j) {
                double num = 0.0, denom = 0.0;
                for (size_t t = j + 1; t < residuals_.size(); ++t) {
                    num += diff_data_[t] * residuals_[t - j - 1];
                    denom += residuals_[t - j - 1] * residuals_[t - j - 1];
                }
                if (denom > 1e-10) {
                    theta_[j] = 0.9 * theta_[j] + 0.1 * (num / denom);
                    theta_[j] = std::max(-0.99, std::min(0.99, theta_[j]));
                }
            }
        }
    }

    residuals_ = compute_residuals();
    sigma_ = std::sqrt(stats::variance(residuals_));
    fitted_ = true;
}

void ARIMA::estimate_parameters() {
    c_ = stats::mean(diff_data_);

    // Estimate AR coefficients using Yule-Walker
    if (p_ > 0) {
        estimate_ar_yule_walker();
    }

    // Initial MA estimates
    if (q_ > 0) {
        estimate_ma_innovations();
    }
}

void ARIMA::estimate_ar_yule_walker() {
    std::vector<double> acf_vals = stats::acf(diff_data_, p_);

    // Build Toeplitz matrix
    matrix::Matrix R(p_, std::vector<double>(p_));
    for (int i = 0; i < p_; ++i) {
        for (int j = 0; j < p_; ++j) {
            R[i][j] = acf_vals[std::abs(i - j)];
        }
    }

    // Right-hand side
    matrix::Vector r(p_);
    for (int i = 0; i < p_; ++i) {
        r[i] = acf_vals[i + 1];
    }

    try {
        phi_ = matrix::solve(R, r);
    } catch (...) {
        // If solve fails, use simple estimation
        for (int i = 0; i < p_; ++i) {
            phi_[i] = acf_vals[i + 1] * 0.5;
        }
    }

    // Ensure stationarity
    for (int i = 0; i < p_; ++i) {
        phi_[i] = std::max(-0.99, std::min(0.99, phi_[i]));
    }
}

void ARIMA::estimate_ma_innovations() {
    // Simple innovations algorithm for MA estimation
    std::vector<double> residuals(diff_data_.size(), 0.0);
    double mean_val = stats::mean(diff_data_);

    for (size_t t = 0; t < diff_data_.size(); ++t) {
        double pred = mean_val;
        for (int i = 0; i < p_ && t > static_cast<size_t>(i); ++i) {
            pred += phi_[i] * (diff_data_[t - i - 1] - mean_val);
        }
        residuals[t] = diff_data_[t] - pred;
    }

    // Estimate theta from residual autocorrelation
    if (residuals.size() > static_cast<size_t>(q_)) {
        std::vector<double> res_acf = stats::acf(residuals, q_);
        for (int j = 0; j < q_; ++j) {
            theta_[j] = res_acf[j + 1] * 0.5;
            theta_[j] = std::max(-0.99, std::min(0.99, theta_[j]));
        }
    }
}

void ARIMA::estimate_arma_css() {
    // Conditional Sum of Squares estimation
    residuals_ = compute_residuals();
}

std::vector<double> ARIMA::compute_residuals() const {
    std::vector<double> residuals(diff_data_.size(), 0.0);
    double mean_val = c_;

    for (size_t t = 0; t < diff_data_.size(); ++t) {
        double pred = mean_val;

        // AR component
        for (int i = 0; i < p_ && t > static_cast<size_t>(i); ++i) {
            pred += phi_[i] * (diff_data_[t - i - 1] - mean_val);
        }

        // MA component
        for (int j = 0; j < q_ && t > static_cast<size_t>(j); ++j) {
            pred += theta_[j] * residuals[t - j - 1];
        }

        residuals[t] = diff_data_[t] - pred;
    }

    return residuals;
}

ForecastResult ARIMA::forecast(int steps) const {
    if (!fitted_) {
        throw std::runtime_error("Model must be fitted before forecasting");
    }

    ForecastResult result;
    result.predictions.resize(steps);
    result.lower_bound.resize(steps);
    result.upper_bound.resize(steps);
    result.confidence_level = 0.95;

    std::vector<double> extended_diff = diff_data_;
    std::vector<double> extended_res = residuals_;

    for (int h = 0; h < steps; ++h) {
        double pred = c_;

        // AR component
        for (int i = 0; i < p_; ++i) {
            size_t idx = extended_diff.size() - i - 1;
            pred += phi_[i] * (extended_diff[idx] - c_);
        }

        // MA component (residuals are 0 for future)
        for (int j = 0; j < q_ && h - j - 1 < 0; ++j) {
            size_t idx = extended_res.size() - (j + 1 - h);
            if (idx < extended_res.size()) {
                pred += theta_[j] * extended_res[idx];
            }
        }

        extended_diff.push_back(pred);
        extended_res.push_back(0.0);
        result.predictions[h] = pred;
    }

    // Undifference predictions
    if (d_ > 0) {
        result.predictions = stats::undifference(result.predictions, data_, d_);
        // Trim to match steps
        result.predictions.erase(result.predictions.begin(),
                                  result.predictions.begin() + d_);
    }

    // Confidence intervals
    double z = 1.96;  // 95% confidence
    for (int h = 0; h < steps; ++h) {
        // Forecast error variance increases with horizon
        double var_h = sigma_ * sigma_;
        for (int j = 0; j < std::min(h, q_); ++j) {
            double psi = (j < q_) ? theta_[j] : 0.0;
            var_h += psi * psi * sigma_ * sigma_;
        }
        double se = std::sqrt(var_h * (h + 1));
        result.lower_bound[h] = result.predictions[h] - z * se;
        result.upper_bound[h] = result.predictions[h] + z * se;
    }

    return result;
}

std::vector<double> ARIMA::fitted_values() const {
    if (!fitted_) return {};

    std::vector<double> fitted(diff_data_.size());
    for (size_t t = 0; t < diff_data_.size(); ++t) {
        fitted[t] = diff_data_[t] - residuals_[t];
    }

    if (d_ > 0) {
        fitted = stats::undifference(fitted, data_, d_);
    }

    return fitted;
}

std::vector<double> ARIMA::residuals() const {
    return residuals_;
}

double ARIMA::log_likelihood() const {
    if (!fitted_) return 0.0;

    int n = static_cast<int>(residuals_.size());
    double sse = 0.0;
    for (double r : residuals_) {
        sse += r * r;
    }

    return -0.5 * n * (std::log(2 * M_PI) + std::log(sse / n) + 1);
}

double ARIMA::aic() const {
    int k = p_ + q_ + 1;  // Number of parameters
    return -2 * log_likelihood() + 2 * k;
}

double ARIMA::bic() const {
    int k = p_ + q_ + 1;
    int n = static_cast<int>(residuals_.size());
    return -2 * log_likelihood() + k * std::log(n);
}

// SARIMA Implementation
SARIMA::SARIMA(int p, int d, int q, int P, int D, int Q, int m)
    : p_(p), d_(d), q_(q), P_(P), D_(D), Q_(Q), m_(m),
      c_(0), sigma_(1.0), fitted_(false) {
    phi_.resize(p_, 0.0);
    theta_.resize(q_, 0.0);
    Phi_.resize(P_, 0.0);
    Theta_.resize(Q_, 0.0);
}

void SARIMA::fit(const std::vector<double>& data, int max_iter, double tol) {
    data_ = data;

    // Apply seasonal differencing
    std::vector<double> diff_data = data;
    for (int i = 0; i < D_; ++i) {
        diff_data = stats::seasonal_difference(diff_data, m_);
    }

    // Apply regular differencing
    diff_data = stats::difference(diff_data, d_);

    // Estimate parameters (simplified approach)
    estimate_parameters();

    residuals_.resize(data_.size(), 0.0);
    sigma_ = stats::std_dev(diff_data);
    fitted_ = true;

    (void)max_iter;
    (void)tol;
}

void SARIMA::estimate_parameters() {
    // Simplified parameter estimation
    c_ = stats::mean(data_);

    // Use PACF for AR estimates
    std::vector<double> pacf = stats::pacf(data_, std::max(p_, P_ * m_));

    for (int i = 0; i < p_; ++i) {
        phi_[i] = (i + 1 < static_cast<int>(pacf.size())) ? pacf[i + 1] * 0.5 : 0.0;
    }

    for (int i = 0; i < P_; ++i) {
        int idx = (i + 1) * m_;
        Phi_[i] = (idx < static_cast<int>(pacf.size())) ? pacf[idx] * 0.5 : 0.0;
    }
}

ForecastResult SARIMA::forecast(int steps) const {
    if (!fitted_) {
        throw std::runtime_error("Model must be fitted before forecasting");
    }

    ForecastResult result;
    result.predictions.resize(steps);
    result.lower_bound.resize(steps);
    result.upper_bound.resize(steps);

    std::vector<double> extended = data_;

    for (int h = 0; h < steps; ++h) {
        double pred = c_;

        // Non-seasonal AR
        for (int i = 0; i < p_; ++i) {
            pred += phi_[i] * (extended[extended.size() - i - 1] - c_);
        }

        // Seasonal AR
        for (int i = 0; i < P_; ++i) {
            size_t idx = extended.size() - (i + 1) * m_;
            if (idx < extended.size()) {
                pred += Phi_[i] * (extended[idx] - c_);
            }
        }

        extended.push_back(pred);
        result.predictions[h] = pred;

        double z = 1.96;
        double se = sigma_ * std::sqrt(h + 1);
        result.lower_bound[h] = pred - z * se;
        result.upper_bound[h] = pred + z * se;
    }

    return result;
}

std::vector<double> SARIMA::fitted_values() const {
    if (!fitted_) return {};

    std::vector<double> fitted(data_.size());
    for (size_t t = 0; t < data_.size(); ++t) {
        fitted[t] = data_[t] - residuals_[t];
    }
    return fitted;
}

std::vector<double> SARIMA::residuals() const {
    return residuals_;
}

// AutoARIMA Implementation
AutoARIMA::AutoARIMA(int max_p, int max_d, int max_q)
    : max_p_(max_p), max_d_(max_d), max_q_(max_q),
      best_p_(0), best_d_(0), best_q_(0), best_aic_(std::numeric_limits<double>::max()) {}

void AutoARIMA::fit(const std::vector<double>& data) {
    // Determine d using variance reduction
    best_d_ = determine_d(data);

    best_aic_ = std::numeric_limits<double>::max();

    for (int p = 0; p <= max_p_; ++p) {
        for (int q = 0; q <= max_q_; ++q) {
            if (p == 0 && q == 0) continue;

            try {
                auto model = std::make_unique<ARIMA>(p, best_d_, q);
                model->fit(data);

                double aic = model->aic();
                if (aic < best_aic_) {
                    best_aic_ = aic;
                    best_p_ = p;
                    best_q_ = q;
                    best_model_ = std::move(model);
                }
            } catch (...) {
                continue;
            }
        }
    }

    if (!best_model_) {
        best_model_ = std::make_unique<ARIMA>(1, best_d_, 0);
        best_model_->fit(data);
        best_p_ = 1;
        best_q_ = 0;
    }
}

ForecastResult AutoARIMA::forecast(int steps) const {
    if (!best_model_) {
        throw std::runtime_error("Model must be fitted before forecasting");
    }
    return best_model_->forecast(steps);
}

int AutoARIMA::determine_d(const std::vector<double>& data) const {
    double prev_var = stats::variance(data);
    std::vector<double> current = data;

    for (int d = 1; d <= max_d_; ++d) {
        current = stats::difference(current, 1);
        double var = stats::variance(current);

        // If variance increases, stop differencing
        if (var >= prev_var * 0.9) {
            return d - 1;
        }
        prev_var = var;
    }

    return max_d_;
}

// ============================================================
// Binary Serialization for ARIMA
// ============================================================

constexpr uint32_t ARIMA_MAGIC = 0x4152494D;  // "ARIM"
constexpr uint32_t ARIMA_VERSION = 1;

size_t ARIMA::parameter_count() const {
    // phi (p) + theta (q) + c + sigma
    return static_cast<size_t>(p_ + q_ + 2);
}

void ARIMA::save(const std::string& filename) const {
    if (!fitted_) {
        throw std::runtime_error("Cannot save unfitted model");
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    // Write header
    file.write(reinterpret_cast<const char*>(&ARIMA_MAGIC), sizeof(ARIMA_MAGIC));
    file.write(reinterpret_cast<const char*>(&ARIMA_VERSION), sizeof(ARIMA_VERSION));

    // Write model orders
    file.write(reinterpret_cast<const char*>(&p_), sizeof(p_));
    file.write(reinterpret_cast<const char*>(&d_), sizeof(d_));
    file.write(reinterpret_cast<const char*>(&q_), sizeof(q_));

    // Write parameters
    file.write(reinterpret_cast<const char*>(&c_), sizeof(c_));
    file.write(reinterpret_cast<const char*>(&sigma_), sizeof(sigma_));

    // Write phi coefficients
    file.write(reinterpret_cast<const char*>(phi_.data()), p_ * sizeof(double));

    // Write theta coefficients
    file.write(reinterpret_cast<const char*>(theta_.data()), q_ * sizeof(double));

    // Write data for forecasting
    uint32_t data_size = static_cast<uint32_t>(data_.size());
    file.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));
    file.write(reinterpret_cast<const char*>(data_.data()), data_size * sizeof(double));

    // Write diff_data
    uint32_t diff_size = static_cast<uint32_t>(diff_data_.size());
    file.write(reinterpret_cast<const char*>(&diff_size), sizeof(diff_size));
    file.write(reinterpret_cast<const char*>(diff_data_.data()), diff_size * sizeof(double));

    // Write residuals
    uint32_t res_size = static_cast<uint32_t>(residuals_.size());
    file.write(reinterpret_cast<const char*>(&res_size), sizeof(res_size));
    file.write(reinterpret_cast<const char*>(residuals_.data()), res_size * sizeof(double));

    file.close();
}

void ARIMA::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }

    // Read and validate header
    uint32_t magic, version;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != ARIMA_MAGIC) {
        throw std::runtime_error("Invalid file format: not an ARIMA model file");
    }
    if (version != ARIMA_VERSION) {
        throw std::runtime_error("Unsupported model version");
    }

    // Read model orders
    file.read(reinterpret_cast<char*>(&p_), sizeof(p_));
    file.read(reinterpret_cast<char*>(&d_), sizeof(d_));
    file.read(reinterpret_cast<char*>(&q_), sizeof(q_));

    // Read parameters
    file.read(reinterpret_cast<char*>(&c_), sizeof(c_));
    file.read(reinterpret_cast<char*>(&sigma_), sizeof(sigma_));

    // Read phi coefficients
    phi_.resize(p_);
    file.read(reinterpret_cast<char*>(phi_.data()), p_ * sizeof(double));

    // Read theta coefficients
    theta_.resize(q_);
    file.read(reinterpret_cast<char*>(theta_.data()), q_ * sizeof(double));

    // Read data
    uint32_t data_size;
    file.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
    data_.resize(data_size);
    file.read(reinterpret_cast<char*>(data_.data()), data_size * sizeof(double));

    // Read diff_data
    uint32_t diff_size;
    file.read(reinterpret_cast<char*>(&diff_size), sizeof(diff_size));
    diff_data_.resize(diff_size);
    file.read(reinterpret_cast<char*>(diff_data_.data()), diff_size * sizeof(double));

    // Read residuals
    uint32_t res_size;
    file.read(reinterpret_cast<char*>(&res_size), sizeof(res_size));
    residuals_.resize(res_size);
    file.read(reinterpret_cast<char*>(residuals_.data()), res_size * sizeof(double));

    fitted_ = true;
    file.close();
}

} // namespace ts
