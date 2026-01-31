#include "dlinear.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>

namespace ts {

// ============================================================================
// DLinear Implementation
// ============================================================================

DLinear::DLinear(int seq_len, int pred_len, int kernel_size, bool individual)
    : seq_len_(seq_len), pred_len_(pred_len), kernel_size_(kernel_size),
      individual_(individual), n_features_(1), target_idx_(0),
      mean_(0), std_(1), is_fitted_(false), is_multivariate_(false) {

    // Ensure kernel size is odd for symmetric moving average
    if (kernel_size_ % 2 == 0) {
        kernel_size_++;
    }
}

void DLinear::initialize_weights() {
    std::random_device rd;
    std::mt19937 gen(rd());

    int input_dim = is_multivariate_ ? seq_len_ * n_features_ : seq_len_;

    // Xavier initialization
    double scale = std::sqrt(2.0 / (input_dim + pred_len_));
    std::normal_distribution<> dist(0.0, scale);

    // Initialize trend weights
    W_trend_.resize(pred_len_, std::vector<double>(input_dim));
    b_trend_.resize(pred_len_, 0.0);

    for (int i = 0; i < pred_len_; ++i) {
        for (int j = 0; j < input_dim; ++j) {
            W_trend_[i][j] = dist(gen);
        }
    }

    // Initialize seasonal weights
    W_seasonal_.resize(pred_len_, std::vector<double>(input_dim));
    b_seasonal_.resize(pred_len_, 0.0);

    for (int i = 0; i < pred_len_; ++i) {
        for (int j = 0; j < input_dim; ++j) {
            W_seasonal_[i][j] = dist(gen);
        }
    }
}

std::vector<double> DLinear::moving_average(const std::vector<double>& x, int kernel_size) const {
    int n = static_cast<int>(x.size());
    std::vector<double> result(n);

    int half = kernel_size / 2;

    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        int count = 0;

        for (int j = -half; j <= half; ++j) {
            int idx = i + j;
            if (idx >= 0 && idx < n) {
                sum += x[idx];
                count++;
            }
        }

        result[i] = sum / count;
    }

    return result;
}

std::pair<std::vector<double>, std::vector<double>>
DLinear::decompose(const std::vector<double>& x) const {
    // Trend = moving average
    std::vector<double> trend = moving_average(x, kernel_size_);

    // Seasonal = residual
    std::vector<double> seasonal(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        seasonal[i] = x[i] - trend[i];
    }

    return {trend, seasonal};
}

std::vector<double> DLinear::forward(const std::vector<double>& trend,
                                      const std::vector<double>& seasonal) const {
    std::vector<double> output(pred_len_, 0.0);

    // Linear projection of trend
    for (int i = 0; i < pred_len_; ++i) {
        double sum = b_trend_[i];
        for (size_t j = 0; j < trend.size(); ++j) {
            sum += W_trend_[i][j] * trend[j];
        }
        output[i] += sum;
    }

    // Linear projection of seasonal
    for (int i = 0; i < pred_len_; ++i) {
        double sum = b_seasonal_[i];
        for (size_t j = 0; j < seasonal.size(); ++j) {
            sum += W_seasonal_[i][j] * seasonal[j];
        }
        output[i] += sum;
    }

    return output;
}

std::vector<double> DLinear::normalize(const std::vector<double>& x) const {
    std::vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = (x[i] - mean_) / (std_ + 1e-8);
    }
    return result;
}

std::vector<double> DLinear::denormalize(const std::vector<double>& x) const {
    std::vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = x[i] * std_ + mean_;
    }
    return result;
}

double DLinear::train_step(const std::vector<std::vector<double>>& batch_x,
                            const std::vector<std::vector<double>>& batch_y,
                            double learning_rate) {
    int batch_size = static_cast<int>(batch_x.size());
    double total_loss = 0.0;

    // Accumulate gradients
    std::vector<std::vector<double>> grad_W_trend(pred_len_, std::vector<double>(seq_len_, 0.0));
    std::vector<double> grad_b_trend(pred_len_, 0.0);
    std::vector<std::vector<double>> grad_W_seasonal(pred_len_, std::vector<double>(seq_len_, 0.0));
    std::vector<double> grad_b_seasonal(pred_len_, 0.0);

    for (int b = 0; b < batch_size; ++b) {
        // Decompose input
        auto [trend, seasonal] = decompose(batch_x[b]);

        // Forward pass
        std::vector<double> pred = forward(trend, seasonal);

        // Compute loss and gradients (MSE)
        for (int i = 0; i < pred_len_; ++i) {
            double error = pred[i] - batch_y[b][i];
            total_loss += error * error;

            // Gradient of loss w.r.t. output
            double d_out = 2.0 * error / (pred_len_ * batch_size);

            // Gradients for trend weights
            for (int j = 0; j < seq_len_; ++j) {
                grad_W_trend[i][j] += d_out * trend[j];
            }
            grad_b_trend[i] += d_out;

            // Gradients for seasonal weights
            for (int j = 0; j < seq_len_; ++j) {
                grad_W_seasonal[i][j] += d_out * seasonal[j];
            }
            grad_b_seasonal[i] += d_out;
        }
    }

    // Update weights with gradient descent
    for (int i = 0; i < pred_len_; ++i) {
        for (int j = 0; j < seq_len_; ++j) {
            W_trend_[i][j] -= learning_rate * grad_W_trend[i][j];
            W_seasonal_[i][j] -= learning_rate * grad_W_seasonal[i][j];
        }
        b_trend_[i] -= learning_rate * grad_b_trend[i];
        b_seasonal_[i] -= learning_rate * grad_b_seasonal[i];
    }

    return total_loss / (batch_size * pred_len_);
}

void DLinear::fit(const std::vector<double>& data, int epochs,
                  double learning_rate, int batch_size) {
    if (static_cast<int>(data.size()) < seq_len_ + pred_len_) {
        throw std::invalid_argument("Insufficient data for DLinear");
    }

    data_ = data;
    is_multivariate_ = false;
    n_features_ = 1;

    // Compute normalization parameters
    mean_ = stats::mean(data);
    std_ = stats::std_dev(data);

    // Normalize data
    std::vector<double> normalized = normalize(data);

    // Initialize weights
    initialize_weights();

    // Create training samples
    int n_samples = static_cast<int>(normalized.size()) - seq_len_ - pred_len_ + 1;
    std::vector<std::vector<double>> X(n_samples);
    std::vector<std::vector<double>> Y(n_samples);

    for (int i = 0; i < n_samples; ++i) {
        X[i] = std::vector<double>(normalized.begin() + i,
                                    normalized.begin() + i + seq_len_);
        Y[i] = std::vector<double>(normalized.begin() + i + seq_len_,
                                    normalized.begin() + i + seq_len_ + pred_len_);
    }

    // Training loop
    std::random_device rd;
    std::mt19937 gen(rd());

    loss_history_.clear();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle samples
        std::vector<int> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);

        double epoch_loss = 0.0;
        int n_batches = 0;

        for (int i = 0; i < n_samples; i += batch_size) {
            int actual_batch_size = std::min(batch_size, n_samples - i);

            std::vector<std::vector<double>> batch_x(actual_batch_size);
            std::vector<std::vector<double>> batch_y(actual_batch_size);

            for (int j = 0; j < actual_batch_size; ++j) {
                batch_x[j] = X[indices[i + j]];
                batch_y[j] = Y[indices[i + j]];
            }

            double batch_loss = train_step(batch_x, batch_y, learning_rate);
            epoch_loss += batch_loss;
            n_batches++;
        }

        loss_history_.push_back(epoch_loss / n_batches);
    }

    is_fitted_ = true;
}

void DLinear::fit(const std::vector<std::vector<double>>& data, int target_idx,
                  int epochs, double learning_rate, int batch_size) {
    if (static_cast<int>(data.size()) < seq_len_ + pred_len_) {
        throw std::invalid_argument("Insufficient data for DLinear");
    }

    multivariate_data_ = data;
    is_multivariate_ = true;
    n_features_ = static_cast<int>(data[0].size());
    target_idx_ = target_idx;

    // Extract target for normalization
    std::vector<double> target(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        target[i] = data[i][target_idx];
    }

    mean_ = stats::mean(target);
    std_ = stats::std_dev(target);

    // For multivariate, we use the univariate fit on target only
    // (simplified version - full version would use all features)
    data_ = target;
    is_multivariate_ = false;

    fit(target, epochs, learning_rate, batch_size);
}

std::vector<double> DLinear::predict(const std::vector<double>& input) const {
    if (!is_fitted_) {
        throw std::runtime_error("Model must be fitted before prediction");
    }

    if (static_cast<int>(input.size()) != seq_len_) {
        throw std::invalid_argument("Input size must equal seq_len");
    }

    // Normalize input
    std::vector<double> normalized = normalize(input);

    // Decompose
    auto [trend, seasonal] = decompose(normalized);

    // Forward pass
    std::vector<double> pred = forward(trend, seasonal);

    // Denormalize
    return denormalize(pred);
}

std::vector<double> DLinear::predict(const std::vector<std::vector<double>>& input) const {
    // Extract target feature and predict
    std::vector<double> target_input(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        target_input[i] = input[i][target_idx_];
    }
    return predict(target_input);
}

size_t DLinear::parameter_count() const {
    // Trend: W_trend (pred_len x input_dim) + b_trend (pred_len)
    // Seasonal: W_seasonal (pred_len x input_dim) + b_seasonal (pred_len)
    size_t input_dim = is_multivariate_ ? seq_len_ * n_features_ : seq_len_;
    size_t trend_params = pred_len_ * input_dim + pred_len_;
    size_t seasonal_params = pred_len_ * input_dim + pred_len_;
    return trend_params + seasonal_params;
}

ForecastResult DLinear::forecast(int steps) const {
    if (!is_fitted_) {
        throw std::runtime_error("Model must be fitted before forecasting");
    }

    ForecastResult result;
    result.confidence_level = 0.95;

    // Get the last seq_len values
    std::vector<double> input(data_.end() - seq_len_, data_.end());

    // If steps > pred_len, we need to do autoregressive forecasting
    std::vector<double> predictions;
    std::vector<double> current_input = input;

    while (static_cast<int>(predictions.size()) < steps) {
        std::vector<double> pred = predict(current_input);

        // Add predictions
        for (int i = 0; i < pred_len_ && static_cast<int>(predictions.size()) < steps; ++i) {
            predictions.push_back(pred[i]);
        }

        // Slide window for next iteration
        if (static_cast<int>(predictions.size()) < steps) {
            int shift = std::min(pred_len_, steps - static_cast<int>(predictions.size()));
            current_input.erase(current_input.begin(), current_input.begin() + shift);
            current_input.insert(current_input.end(), pred.begin(), pred.begin() + shift);
        }
    }

    result.predictions = predictions;

    // Estimate confidence intervals from training residuals
    double sigma = 0.0;
    int count = 0;

    for (size_t i = seq_len_; i + pred_len_ <= data_.size(); ++i) {
        std::vector<double> x(data_.begin() + i - seq_len_, data_.begin() + i);
        std::vector<double> pred = predict(x);

        for (int j = 0; j < pred_len_ && i + j < data_.size(); ++j) {
            double error = data_[i + j] - pred[j];
            sigma += error * error;
            count++;
        }

        if (count > 100) break;  // Limit computation
    }

    sigma = std::sqrt(sigma / std::max(1, count));
    double z = 1.96;

    result.lower_bound.resize(steps);
    result.upper_bound.resize(steps);

    for (int i = 0; i < steps; ++i) {
        double se = sigma * std::sqrt(1 + i * 0.1);  // Increasing uncertainty
        result.lower_bound[i] = predictions[i] - z * se;
        result.upper_bound[i] = predictions[i] + z * se;
    }

    return result;
}

// ============================================================================
// NLinear Implementation
// ============================================================================

NLinear::NLinear(int seq_len, int pred_len)
    : seq_len_(seq_len), pred_len_(pred_len),
      mean_(0), std_(1), is_fitted_(false) {}

void NLinear::initialize_weights() {
    std::random_device rd;
    std::mt19937 gen(rd());

    double scale = std::sqrt(2.0 / (seq_len_ + pred_len_));
    std::normal_distribution<> dist(0.0, scale);

    W_.resize(pred_len_, std::vector<double>(seq_len_));
    b_.resize(pred_len_, 0.0);

    for (int i = 0; i < pred_len_; ++i) {
        for (int j = 0; j < seq_len_; ++j) {
            W_[i][j] = dist(gen);
        }
    }
}

std::vector<double> NLinear::forward(const std::vector<double>& x, double last_val) const {
    std::vector<double> output(pred_len_);

    for (int i = 0; i < pred_len_; ++i) {
        double sum = b_[i];
        for (int j = 0; j < seq_len_; ++j) {
            sum += W_[i][j] * x[j];
        }
        output[i] = sum + last_val;
    }

    return output;
}

void NLinear::fit(const std::vector<double>& data, int epochs,
                  double learning_rate, int batch_size) {
    if (static_cast<int>(data.size()) < seq_len_ + pred_len_) {
        throw std::invalid_argument("Insufficient data for NLinear");
    }

    data_ = data;
    mean_ = stats::mean(data);
    std_ = stats::std_dev(data);

    // Normalize
    std::vector<double> normalized(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        normalized[i] = (data[i] - mean_) / (std_ + 1e-8);
    }

    initialize_weights();

    int n_samples = static_cast<int>(normalized.size()) - seq_len_ - pred_len_ + 1;

    std::random_device rd;
    std::mt19937 gen(rd());

    loss_history_.clear();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<int> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);

        double epoch_loss = 0.0;

        for (int i = 0; i < n_samples; i += batch_size) {
            int actual_batch = std::min(batch_size, n_samples - i);

            // Accumulate gradients
            std::vector<std::vector<double>> grad_W(pred_len_, std::vector<double>(seq_len_, 0.0));
            std::vector<double> grad_b(pred_len_, 0.0);

            for (int b = 0; b < actual_batch; ++b) {
                int idx = indices[i + b];

                // Input with last value subtracted
                std::vector<double> x(seq_len_);
                double last_val = normalized[idx + seq_len_ - 1];
                for (int j = 0; j < seq_len_; ++j) {
                    x[j] = normalized[idx + j] - last_val;
                }

                // Target
                std::vector<double> y(pred_len_);
                for (int j = 0; j < pred_len_; ++j) {
                    y[j] = normalized[idx + seq_len_ + j];
                }

                // Forward
                std::vector<double> pred = forward(x, last_val);

                // Compute gradients
                for (int j = 0; j < pred_len_; ++j) {
                    double error = pred[j] - y[j];
                    epoch_loss += error * error;

                    double d_out = 2.0 * error / (pred_len_ * actual_batch);

                    for (int k = 0; k < seq_len_; ++k) {
                        grad_W[j][k] += d_out * x[k];
                    }
                    grad_b[j] += d_out;
                }
            }

            // Update weights
            for (int j = 0; j < pred_len_; ++j) {
                for (int k = 0; k < seq_len_; ++k) {
                    W_[j][k] -= learning_rate * grad_W[j][k];
                }
                b_[j] -= learning_rate * grad_b[j];
            }
        }

        loss_history_.push_back(epoch_loss / n_samples);
    }

    is_fitted_ = true;
}

std::vector<double> NLinear::predict(const std::vector<double>& input) const {
    if (!is_fitted_) {
        throw std::runtime_error("Model must be fitted before prediction");
    }

    // Normalize input
    std::vector<double> normalized(seq_len_);
    for (int i = 0; i < seq_len_; ++i) {
        normalized[i] = (input[i] - mean_) / (std_ + 1e-8);
    }

    double last_val = normalized[seq_len_ - 1];

    // Subtract last value
    std::vector<double> x(seq_len_);
    for (int i = 0; i < seq_len_; ++i) {
        x[i] = normalized[i] - last_val;
    }

    // Forward
    std::vector<double> pred = forward(x, last_val);

    // Denormalize
    for (int i = 0; i < pred_len_; ++i) {
        pred[i] = pred[i] * std_ + mean_;
    }

    return pred;
}

size_t NLinear::parameter_count() const {
    // W (pred_len x seq_len) + b (pred_len)
    return static_cast<size_t>(pred_len_) * seq_len_ + pred_len_;
}

ForecastResult NLinear::forecast(int steps) const {
    if (!is_fitted_) {
        throw std::runtime_error("Model must be fitted before forecasting");
    }

    ForecastResult result;
    result.confidence_level = 0.95;

    std::vector<double> input(data_.end() - seq_len_, data_.end());
    std::vector<double> predictions;
    std::vector<double> current_input = input;

    while (static_cast<int>(predictions.size()) < steps) {
        std::vector<double> pred = predict(current_input);

        for (int i = 0; i < pred_len_ && static_cast<int>(predictions.size()) < steps; ++i) {
            predictions.push_back(pred[i]);
        }

        if (static_cast<int>(predictions.size()) < steps) {
            int shift = std::min(pred_len_, steps - static_cast<int>(predictions.size()));
            current_input.erase(current_input.begin(), current_input.begin() + shift);
            current_input.insert(current_input.end(), pred.begin(), pred.begin() + shift);
        }
    }

    result.predictions = predictions;

    // Simple confidence intervals
    double sigma = std_ * 0.1;
    double z = 1.96;

    result.lower_bound.resize(steps);
    result.upper_bound.resize(steps);

    for (int i = 0; i < steps; ++i) {
        double se = sigma * std::sqrt(1 + i * 0.1);
        result.lower_bound[i] = predictions[i] - z * se;
        result.upper_bound[i] = predictions[i] + z * se;
    }

    return result;
}

// ============================================================================
// Linear Implementation
// ============================================================================

Linear::Linear(int seq_len, int pred_len)
    : seq_len_(seq_len), pred_len_(pred_len),
      mean_(0), std_(1), is_fitted_(false) {}

void Linear::initialize_weights() {
    std::random_device rd;
    std::mt19937 gen(rd());

    double scale = std::sqrt(2.0 / (seq_len_ + pred_len_));
    std::normal_distribution<> dist(0.0, scale);

    W_.resize(pred_len_, std::vector<double>(seq_len_));
    b_.resize(pred_len_, 0.0);

    for (int i = 0; i < pred_len_; ++i) {
        for (int j = 0; j < seq_len_; ++j) {
            W_[i][j] = dist(gen);
        }
    }
}

void Linear::fit(const std::vector<double>& data, int epochs,
                 double learning_rate, int batch_size) {
    if (static_cast<int>(data.size()) < seq_len_ + pred_len_) {
        throw std::invalid_argument("Insufficient data for Linear model");
    }

    data_ = data;
    mean_ = stats::mean(data);
    std_ = stats::std_dev(data);

    std::vector<double> normalized(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        normalized[i] = (data[i] - mean_) / (std_ + 1e-8);
    }

    initialize_weights();

    int n_samples = static_cast<int>(normalized.size()) - seq_len_ - pred_len_ + 1;

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<int> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);

        for (int i = 0; i < n_samples; i += batch_size) {
            int actual_batch = std::min(batch_size, n_samples - i);

            std::vector<std::vector<double>> grad_W(pred_len_, std::vector<double>(seq_len_, 0.0));
            std::vector<double> grad_b(pred_len_, 0.0);

            for (int b = 0; b < actual_batch; ++b) {
                int idx = indices[i + b];

                // Forward pass
                std::vector<double> pred(pred_len_);
                for (int j = 0; j < pred_len_; ++j) {
                    pred[j] = b_[j];
                    for (int k = 0; k < seq_len_; ++k) {
                        pred[j] += W_[j][k] * normalized[idx + k];
                    }
                }

                // Compute gradients
                for (int j = 0; j < pred_len_; ++j) {
                    double error = pred[j] - normalized[idx + seq_len_ + j];
                    double d_out = 2.0 * error / (pred_len_ * actual_batch);

                    for (int k = 0; k < seq_len_; ++k) {
                        grad_W[j][k] += d_out * normalized[idx + k];
                    }
                    grad_b[j] += d_out;
                }
            }

            // Update weights
            for (int j = 0; j < pred_len_; ++j) {
                for (int k = 0; k < seq_len_; ++k) {
                    W_[j][k] -= learning_rate * grad_W[j][k];
                }
                b_[j] -= learning_rate * grad_b[j];
            }
        }
    }

    is_fitted_ = true;
}

std::vector<double> Linear::predict(const std::vector<double>& input) const {
    if (!is_fitted_) {
        throw std::runtime_error("Model must be fitted before prediction");
    }

    std::vector<double> normalized(seq_len_);
    for (int i = 0; i < seq_len_; ++i) {
        normalized[i] = (input[i] - mean_) / (std_ + 1e-8);
    }

    std::vector<double> pred(pred_len_);
    for (int i = 0; i < pred_len_; ++i) {
        pred[i] = b_[i];
        for (int j = 0; j < seq_len_; ++j) {
            pred[i] += W_[i][j] * normalized[j];
        }
        pred[i] = pred[i] * std_ + mean_;
    }

    return pred;
}

size_t Linear::parameter_count() const {
    // W (pred_len x seq_len) + b (pred_len)
    return static_cast<size_t>(pred_len_) * seq_len_ + pred_len_;
}

ForecastResult Linear::forecast(int steps) const {
    if (!is_fitted_) {
        throw std::runtime_error("Model must be fitted before forecasting");
    }

    ForecastResult result;
    result.confidence_level = 0.95;

    std::vector<double> input(data_.end() - seq_len_, data_.end());
    std::vector<double> predictions;
    std::vector<double> current_input = input;

    while (static_cast<int>(predictions.size()) < steps) {
        std::vector<double> pred = predict(current_input);

        for (int i = 0; i < pred_len_ && static_cast<int>(predictions.size()) < steps; ++i) {
            predictions.push_back(pred[i]);
        }

        if (static_cast<int>(predictions.size()) < steps) {
            int shift = std::min(pred_len_, steps - static_cast<int>(predictions.size()));
            current_input.erase(current_input.begin(), current_input.begin() + shift);
            current_input.insert(current_input.end(), pred.begin(), pred.begin() + shift);
        }
    }

    result.predictions = predictions;

    double sigma = std_ * 0.1;
    double z = 1.96;

    result.lower_bound.resize(steps);
    result.upper_bound.resize(steps);

    for (int i = 0; i < steps; ++i) {
        double se = sigma * std::sqrt(1 + i * 0.1);
        result.lower_bound[i] = predictions[i] - z * se;
        result.upper_bound[i] = predictions[i] + z * se;
    }

    return result;
}

// ============================================================
// Binary Serialization
// ============================================================

// Magic numbers for file format validation
constexpr uint32_t DLINEAR_MAGIC = 0x444C494E;  // "DLIN"
constexpr uint32_t NLINEAR_MAGIC = 0x4E4C494E;  // "NLIN"
constexpr uint32_t LINEAR_MAGIC = 0x4C494E45;   // "LINE"
constexpr uint32_t MODEL_VERSION = 1;

// Helper functions for binary I/O
namespace {

void write_vector(std::ofstream& file, const std::vector<double>& vec) {
    uint32_t size = static_cast<uint32_t>(vec.size());
    file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    file.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(double));
}

void read_vector(std::ifstream& file, std::vector<double>& vec) {
    uint32_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));
    vec.resize(size);
    file.read(reinterpret_cast<char*>(vec.data()), size * sizeof(double));
}

void write_matrix(std::ofstream& file, const std::vector<std::vector<double>>& mat) {
    uint32_t rows = static_cast<uint32_t>(mat.size());
    uint32_t cols = rows > 0 ? static_cast<uint32_t>(mat[0].size()) : 0;
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    for (const auto& row : mat) {
        file.write(reinterpret_cast<const char*>(row.data()), cols * sizeof(double));
    }
}

void read_matrix(std::ifstream& file, std::vector<std::vector<double>>& mat) {
    uint32_t rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    mat.resize(rows);
    for (uint32_t i = 0; i < rows; ++i) {
        mat[i].resize(cols);
        file.read(reinterpret_cast<char*>(mat[i].data()), cols * sizeof(double));
    }
}

} // anonymous namespace

void DLinear::save(const std::string& filename) const {
    if (!is_fitted_) {
        throw std::runtime_error("Cannot save unfitted model");
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    // Write header
    file.write(reinterpret_cast<const char*>(&DLINEAR_MAGIC), sizeof(DLINEAR_MAGIC));
    file.write(reinterpret_cast<const char*>(&MODEL_VERSION), sizeof(MODEL_VERSION));

    // Write model parameters
    file.write(reinterpret_cast<const char*>(&seq_len_), sizeof(seq_len_));
    file.write(reinterpret_cast<const char*>(&pred_len_), sizeof(pred_len_));
    file.write(reinterpret_cast<const char*>(&kernel_size_), sizeof(kernel_size_));
    file.write(reinterpret_cast<const char*>(&n_features_), sizeof(n_features_));
    file.write(reinterpret_cast<const char*>(&target_idx_), sizeof(target_idx_));
    file.write(reinterpret_cast<const char*>(&mean_), sizeof(mean_));
    file.write(reinterpret_cast<const char*>(&std_), sizeof(std_));
    file.write(reinterpret_cast<const char*>(&is_multivariate_), sizeof(is_multivariate_));

    // Write weights
    write_matrix(file, W_trend_);
    write_vector(file, b_trend_);
    write_matrix(file, W_seasonal_);
    write_vector(file, b_seasonal_);

    // Write stored data for forecasting
    write_vector(file, data_);

    // Write multivariate data if applicable
    uint32_t has_mv = is_multivariate_ ? 1 : 0;
    file.write(reinterpret_cast<const char*>(&has_mv), sizeof(has_mv));
    if (is_multivariate_) {
        write_matrix(file, multivariate_data_);
    }

    file.close();
}

void DLinear::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }

    // Read and validate header
    uint32_t magic, version;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != DLINEAR_MAGIC) {
        throw std::runtime_error("Invalid file format: not a DLinear model file");
    }
    if (version != MODEL_VERSION) {
        throw std::runtime_error("Unsupported model version");
    }

    // Read model parameters
    file.read(reinterpret_cast<char*>(&seq_len_), sizeof(seq_len_));
    file.read(reinterpret_cast<char*>(&pred_len_), sizeof(pred_len_));
    file.read(reinterpret_cast<char*>(&kernel_size_), sizeof(kernel_size_));
    file.read(reinterpret_cast<char*>(&n_features_), sizeof(n_features_));
    file.read(reinterpret_cast<char*>(&target_idx_), sizeof(target_idx_));
    file.read(reinterpret_cast<char*>(&mean_), sizeof(mean_));
    file.read(reinterpret_cast<char*>(&std_), sizeof(std_));
    file.read(reinterpret_cast<char*>(&is_multivariate_), sizeof(is_multivariate_));

    // Read weights
    read_matrix(file, W_trend_);
    read_vector(file, b_trend_);
    read_matrix(file, W_seasonal_);
    read_vector(file, b_seasonal_);

    // Read stored data
    read_vector(file, data_);

    // Read multivariate data if applicable
    uint32_t has_mv;
    file.read(reinterpret_cast<char*>(&has_mv), sizeof(has_mv));
    if (has_mv) {
        read_matrix(file, multivariate_data_);
    }

    is_fitted_ = true;
    file.close();
}

void NLinear::save(const std::string& filename) const {
    if (!is_fitted_) {
        throw std::runtime_error("Cannot save unfitted model");
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    // Write header
    file.write(reinterpret_cast<const char*>(&NLINEAR_MAGIC), sizeof(NLINEAR_MAGIC));
    file.write(reinterpret_cast<const char*>(&MODEL_VERSION), sizeof(MODEL_VERSION));

    // Write model parameters
    file.write(reinterpret_cast<const char*>(&seq_len_), sizeof(seq_len_));
    file.write(reinterpret_cast<const char*>(&pred_len_), sizeof(pred_len_));
    file.write(reinterpret_cast<const char*>(&mean_), sizeof(mean_));
    file.write(reinterpret_cast<const char*>(&std_), sizeof(std_));

    // Write weights
    write_matrix(file, W_);
    write_vector(file, b_);

    // Write stored data
    write_vector(file, data_);

    file.close();
}

void NLinear::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }

    // Read and validate header
    uint32_t magic, version;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != NLINEAR_MAGIC) {
        throw std::runtime_error("Invalid file format: not a NLinear model file");
    }
    if (version != MODEL_VERSION) {
        throw std::runtime_error("Unsupported model version");
    }

    // Read model parameters
    file.read(reinterpret_cast<char*>(&seq_len_), sizeof(seq_len_));
    file.read(reinterpret_cast<char*>(&pred_len_), sizeof(pred_len_));
    file.read(reinterpret_cast<char*>(&mean_), sizeof(mean_));
    file.read(reinterpret_cast<char*>(&std_), sizeof(std_));

    // Read weights
    read_matrix(file, W_);
    read_vector(file, b_);

    // Read stored data
    read_vector(file, data_);

    is_fitted_ = true;
    file.close();
}

void Linear::save(const std::string& filename) const {
    if (!is_fitted_) {
        throw std::runtime_error("Cannot save unfitted model");
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    // Write header
    file.write(reinterpret_cast<const char*>(&LINEAR_MAGIC), sizeof(LINEAR_MAGIC));
    file.write(reinterpret_cast<const char*>(&MODEL_VERSION), sizeof(MODEL_VERSION));

    // Write model parameters
    file.write(reinterpret_cast<const char*>(&seq_len_), sizeof(seq_len_));
    file.write(reinterpret_cast<const char*>(&pred_len_), sizeof(pred_len_));
    file.write(reinterpret_cast<const char*>(&mean_), sizeof(mean_));
    file.write(reinterpret_cast<const char*>(&std_), sizeof(std_));

    // Write weights
    write_matrix(file, W_);
    write_vector(file, b_);

    // Write stored data
    write_vector(file, data_);

    file.close();
}

void Linear::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }

    // Read and validate header
    uint32_t magic, version;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != LINEAR_MAGIC) {
        throw std::runtime_error("Invalid file format: not a Linear model file");
    }
    if (version != MODEL_VERSION) {
        throw std::runtime_error("Unsupported model version");
    }

    // Read model parameters
    file.read(reinterpret_cast<char*>(&seq_len_), sizeof(seq_len_));
    file.read(reinterpret_cast<char*>(&pred_len_), sizeof(pred_len_));
    file.read(reinterpret_cast<char*>(&mean_), sizeof(mean_));
    file.read(reinterpret_cast<char*>(&std_), sizeof(std_));

    // Read weights
    read_matrix(file, W_);
    read_vector(file, b_);

    // Read stored data
    read_vector(file, data_);

    is_fitted_ = true;
    file.close();
}

} // namespace ts
