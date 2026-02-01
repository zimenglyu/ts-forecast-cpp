#include "prophet.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace ts {

Prophet::Prophet(GrowthType growth, bool yearly_seasonality,
                 bool weekly_seasonality, bool daily_seasonality)
    : growth_type_(growth), cap_(1.0), floor_(0.0),
      yearly_seasonality_(yearly_seasonality),
      weekly_seasonality_(weekly_seasonality),
      daily_seasonality_(daily_seasonality),
      n_changepoints_(25), changepoint_range_(0.8),
      changepoint_prior_scale_(0.05),
      k_(0), m_(0), t_min_(0), t_max_(1), fitted_(false) {}

void Prophet::add_seasonality(const std::string& name, double period, int fourier_order) {
    custom_seasonalities_.push_back({name, period, fourier_order});
}

void Prophet::add_holiday(const Holiday& holiday) {
    holidays_.push_back(holiday);
}

void Prophet::add_holidays(const std::vector<Holiday>& holidays) {
    for (const auto& h : holidays) {
        holidays_.push_back(h);
    }
}

void Prophet::set_changepoints(int n_changepoints, double changepoint_range) {
    n_changepoints_ = n_changepoints;
    changepoint_range_ = changepoint_range;
}

void Prophet::fit(const std::vector<double>& values) {
    std::vector<double> timestamps(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        timestamps[i] = static_cast<double>(i);
    }
    fit(timestamps, values);
}

void Prophet::fit(const std::vector<double>& timestamps, const std::vector<double>& values) {
    if (timestamps.size() != values.size()) {
        throw std::invalid_argument("Timestamps and values must have same length");
    }
    if (timestamps.empty()) {
        throw std::invalid_argument("Data cannot be empty");
    }

    timestamps_ = timestamps;
    values_ = values;

    // Normalize timestamps to [0, 1]
    t_min_ = *std::min_element(timestamps_.begin(), timestamps_.end());
    t_max_ = *std::max_element(timestamps_.begin(), timestamps_.end());

    initialize_changepoints();
    fit_trend();
    fit_seasonality();
    fit_holidays();

    fitted_ = true;
}

void Prophet::initialize_changepoints() {
    changepoints_.clear();
    delta_.clear();

    if (n_changepoints_ <= 0) return;

    int n = static_cast<int>(timestamps_.size());
    int range_end = static_cast<int>(n * changepoint_range_);

    if (range_end <= n_changepoints_) {
        n_changepoints_ = range_end / 2;
    }

    if (n_changepoints_ > 0) {
        int step = range_end / n_changepoints_;
        for (int i = 1; i <= n_changepoints_; ++i) {
            int idx = std::min(i * step, range_end - 1);
            changepoints_.push_back({timestamps_[idx], 0.0});
        }
        delta_.resize(n_changepoints_, 0.0);
    }
}

void Prophet::fit_trend() {
    // Linear regression for base trend
    double sum_t = 0, sum_y = 0, sum_tt = 0, sum_ty = 0;
    int n = static_cast<int>(values_.size());

    for (int i = 0; i < n; ++i) {
        double t = (timestamps_[i] - t_min_) / (t_max_ - t_min_ + 1e-10);
        sum_t += t;
        sum_y += values_[i];
        sum_tt += t * t;
        sum_ty += t * values_[i];
    }

    double denom = n * sum_tt - sum_t * sum_t;
    if (std::abs(denom) < 1e-10) {
        k_ = 0;
        m_ = sum_y / n;
    } else {
        k_ = (n * sum_ty - sum_t * sum_y) / denom;
        m_ = (sum_y - k_ * sum_t) / n;
    }

    // Estimate changepoint magnitudes
    if (!changepoints_.empty()) {
        std::vector<double> residuals(n);
        for (int i = 0; i < n; ++i) {
            double t = (timestamps_[i] - t_min_) / (t_max_ - t_min_ + 1e-10);
            residuals[i] = values_[i] - (k_ * t + m_);
        }

        // Simple estimation: average residual change at each changepoint
        for (size_t c = 0; c < changepoints_.size(); ++c) {
            double cp_t = changepoints_[c].timestamp;
            int cp_idx = 0;
            for (int i = 0; i < n; ++i) {
                if (timestamps_[i] >= cp_t) {
                    cp_idx = i;
                    break;
                }
            }

            if (cp_idx > 0 && cp_idx < n - 1) {
                double before = 0, after = 0;
                int count_before = 0, count_after = 0;

                for (int i = std::max(0, cp_idx - 10); i < cp_idx; ++i) {
                    before += residuals[i];
                    count_before++;
                }
                for (int i = cp_idx; i < std::min(n, cp_idx + 10); ++i) {
                    after += residuals[i];
                    count_after++;
                }

                if (count_before > 0 && count_after > 0) {
                    delta_[c] = (after / count_after - before / count_before) * changepoint_prior_scale_;
                }
                changepoints_[c].rate_change = delta_[c];
            }
        }
    }
}

void Prophet::fit_seasonality() {
    seasonality_coeffs_.clear();

    // Compute residuals after trend
    std::vector<double> residuals(values_.size());
    for (size_t i = 0; i < values_.size(); ++i) {
        residuals[i] = values_[i] - compute_trend(timestamps_[i]);
    }

    // Yearly seasonality (Fourier series with period 365.25 days)
    if (yearly_seasonality_) {
        int order = 10;
        std::vector<double> coeffs(2 * order, 0.0);

        for (int k = 1; k <= order; ++k) {
            double sin_sum = 0, cos_sum = 0, sin2_sum = 0, cos2_sum = 0;
            double sincos_sum = 0, sin_y_sum = 0, cos_y_sum = 0;

            for (size_t i = 0; i < values_.size(); ++i) {
                double angle = 2 * M_PI * k * timestamps_[i] / 365.25;
                double s = std::sin(angle);
                double c = std::cos(angle);

                sin_sum += s;
                cos_sum += c;
                sin2_sum += s * s;
                cos2_sum += c * c;
                sincos_sum += s * c;
                sin_y_sum += s * residuals[i];
                cos_y_sum += c * residuals[i];
            }

            int n = static_cast<int>(values_.size());
            double denom = sin2_sum * cos2_sum - sincos_sum * sincos_sum;
            if (std::abs(denom) > 1e-10) {
                coeffs[2 * (k - 1)] = (cos2_sum * sin_y_sum - sincos_sum * cos_y_sum) / denom;
                coeffs[2 * (k - 1) + 1] = (sin2_sum * cos_y_sum - sincos_sum * sin_y_sum) / denom;
            } else {
                coeffs[2 * (k - 1)] = sin_y_sum / (n + 1e-10);
                coeffs[2 * (k - 1) + 1] = cos_y_sum / (n + 1e-10);
            }
        }
        seasonality_coeffs_["yearly"] = coeffs;
    }

    // Weekly seasonality (period 7 days)
    if (weekly_seasonality_) {
        int order = 3;
        std::vector<double> coeffs(2 * order, 0.0);

        for (int k = 1; k <= order; ++k) {
            double sin_y_sum = 0, cos_y_sum = 0;
            double sin2_sum = 0, cos2_sum = 0;

            for (size_t i = 0; i < values_.size(); ++i) {
                double angle = 2 * M_PI * k * timestamps_[i] / 7.0;
                double s = std::sin(angle);
                double c = std::cos(angle);

                sin_y_sum += s * residuals[i];
                cos_y_sum += c * residuals[i];
                sin2_sum += s * s;
                cos2_sum += c * c;
            }

            coeffs[2 * (k - 1)] = sin_y_sum / (sin2_sum + 1e-10);
            coeffs[2 * (k - 1) + 1] = cos_y_sum / (cos2_sum + 1e-10);
        }
        seasonality_coeffs_["weekly"] = coeffs;
    }

    // Daily seasonality (period 1 day, for sub-daily data)
    if (daily_seasonality_) {
        int order = 4;
        std::vector<double> coeffs(2 * order, 0.0);

        for (int k = 1; k <= order; ++k) {
            double sin_y_sum = 0, cos_y_sum = 0;
            double sin2_sum = 0, cos2_sum = 0;

            for (size_t i = 0; i < values_.size(); ++i) {
                double angle = 2 * M_PI * k * timestamps_[i];
                double s = std::sin(angle);
                double c = std::cos(angle);

                sin_y_sum += s * residuals[i];
                cos_y_sum += c * residuals[i];
                sin2_sum += s * s;
                cos2_sum += c * c;
            }

            coeffs[2 * (k - 1)] = sin_y_sum / (sin2_sum + 1e-10);
            coeffs[2 * (k - 1) + 1] = cos_y_sum / (cos2_sum + 1e-10);
        }
        seasonality_coeffs_["daily"] = coeffs;
    }

    // Custom seasonalities
    for (const auto& spec : custom_seasonalities_) {
        std::vector<double> coeffs(2 * spec.fourier_order, 0.0);

        for (int k = 1; k <= spec.fourier_order; ++k) {
            double sin_y_sum = 0, cos_y_sum = 0;
            double sin2_sum = 0, cos2_sum = 0;

            for (size_t i = 0; i < values_.size(); ++i) {
                double angle = 2 * M_PI * k * timestamps_[i] / spec.period;
                double s = std::sin(angle);
                double c = std::cos(angle);

                sin_y_sum += s * residuals[i];
                cos_y_sum += c * residuals[i];
                sin2_sum += s * s;
                cos2_sum += c * c;
            }

            coeffs[2 * (k - 1)] = sin_y_sum / (sin2_sum + 1e-10);
            coeffs[2 * (k - 1) + 1] = cos_y_sum / (cos2_sum + 1e-10);
        }
        seasonality_coeffs_[spec.name] = coeffs;
    }
}

void Prophet::fit_holidays() {
    holiday_coeffs_.clear();

    if (holidays_.empty()) return;

    // Compute residuals after trend and seasonality
    std::vector<double> residuals(values_.size());
    for (size_t i = 0; i < values_.size(); ++i) {
        residuals[i] = values_[i] - compute_trend(timestamps_[i]) -
                       compute_seasonality(timestamps_[i]);
    }

    for (const auto& holiday : holidays_) {
        double sum = 0.0;
        int count = 0;

        for (size_t i = 0; i < timestamps_.size(); ++i) {
            for (double hdate : holiday.dates) {
                if (timestamps_[i] >= hdate + holiday.lower_window &&
                    timestamps_[i] <= hdate + holiday.upper_window) {
                    sum += residuals[i];
                    count++;
                    break;
                }
            }
        }

        holiday_coeffs_[holiday.name] = (count > 0) ? sum / count : 0.0;
    }
}

double Prophet::compute_trend(double t) const {
    double t_norm = (t - t_min_) / (t_max_ - t_min_ + 1e-10);

    if (growth_type_ == GrowthType::LINEAR) {
        return piecewise_linear(t);
    } else {
        return piecewise_logistic(t);
    }
}

double Prophet::piecewise_linear(double t) const {
    double t_norm = (t - t_min_) / (t_max_ - t_min_ + 1e-10);
    double trend = k_ * t_norm + m_;

    // Add changepoint effects
    for (size_t c = 0; c < changepoints_.size(); ++c) {
        if (t >= changepoints_[c].timestamp) {
            double t_cp = (changepoints_[c].timestamp - t_min_) / (t_max_ - t_min_ + 1e-10);
            trend += delta_[c] * (t_norm - t_cp);
        }
    }

    return trend;
}

double Prophet::piecewise_logistic(double t) const {
    double t_norm = (t - t_min_) / (t_max_ - t_min_ + 1e-10);

    // Logistic growth: cap / (1 + exp(-k * (t - m)))
    double growth_rate = k_;
    for (size_t c = 0; c < changepoints_.size(); ++c) {
        if (t >= changepoints_[c].timestamp) {
            growth_rate += delta_[c];
        }
    }

    double capacity = cap_ - floor_;
    return floor_ + capacity / (1.0 + std::exp(-growth_rate * (t_norm - 0.5)));
}

double Prophet::compute_seasonality(double t) const {
    double seasonal = 0.0;

    if (yearly_seasonality_ && seasonality_coeffs_.count("yearly")) {
        const auto& coeffs = seasonality_coeffs_.at("yearly");
        int order = static_cast<int>(coeffs.size()) / 2;
        for (int k = 1; k <= order; ++k) {
            double angle = 2 * M_PI * k * t / 365.25;
            seasonal += coeffs[2 * (k - 1)] * std::sin(angle);
            seasonal += coeffs[2 * (k - 1) + 1] * std::cos(angle);
        }
    }

    if (weekly_seasonality_ && seasonality_coeffs_.count("weekly")) {
        const auto& coeffs = seasonality_coeffs_.at("weekly");
        int order = static_cast<int>(coeffs.size()) / 2;
        for (int k = 1; k <= order; ++k) {
            double angle = 2 * M_PI * k * t / 7.0;
            seasonal += coeffs[2 * (k - 1)] * std::sin(angle);
            seasonal += coeffs[2 * (k - 1) + 1] * std::cos(angle);
        }
    }

    if (daily_seasonality_ && seasonality_coeffs_.count("daily")) {
        const auto& coeffs = seasonality_coeffs_.at("daily");
        int order = static_cast<int>(coeffs.size()) / 2;
        for (int k = 1; k <= order; ++k) {
            double angle = 2 * M_PI * k * t;
            seasonal += coeffs[2 * (k - 1)] * std::sin(angle);
            seasonal += coeffs[2 * (k - 1) + 1] * std::cos(angle);
        }
    }

    // Custom seasonalities
    for (const auto& spec : custom_seasonalities_) {
        if (seasonality_coeffs_.count(spec.name)) {
            const auto& coeffs = seasonality_coeffs_.at(spec.name);
            for (int k = 1; k <= spec.fourier_order; ++k) {
                double angle = 2 * M_PI * k * t / spec.period;
                seasonal += coeffs[2 * (k - 1)] * std::sin(angle);
                seasonal += coeffs[2 * (k - 1) + 1] * std::cos(angle);
            }
        }
    }

    return seasonal;
}

double Prophet::compute_holidays(double t) const {
    double effect = 0.0;

    for (const auto& holiday : holidays_) {
        for (double hdate : holiday.dates) {
            if (t >= hdate + holiday.lower_window &&
                t <= hdate + holiday.upper_window) {
                if (holiday_coeffs_.count(holiday.name)) {
                    effect += holiday_coeffs_.at(holiday.name);
                }
                break;
            }
        }
    }

    return effect;
}

std::vector<double> Prophet::fourier_series(double t, double period, int order) const {
    std::vector<double> features(2 * order);
    for (int k = 1; k <= order; ++k) {
        double angle = 2 * M_PI * k * t / period;
        features[2 * (k - 1)] = std::sin(angle);
        features[2 * (k - 1) + 1] = std::cos(angle);
    }
    return features;
}

ForecastResult Prophet::predict(const std::vector<double>& timestamps) const {
    if (!fitted_) {
        throw std::runtime_error("Model must be fitted before prediction");
    }

    ForecastResult result;
    result.predictions.resize(timestamps.size());
    result.lower_bound.resize(timestamps.size());
    result.upper_bound.resize(timestamps.size());

    for (size_t i = 0; i < timestamps.size(); ++i) {
        result.predictions[i] = compute_trend(timestamps[i]) +
                                 compute_seasonality(timestamps[i]) +
                                 compute_holidays(timestamps[i]);
    }

    // Estimate uncertainty
    double sigma = 0.0;
    for (size_t i = 0; i < values_.size(); ++i) {
        double pred = compute_trend(timestamps_[i]) +
                      compute_seasonality(timestamps_[i]) +
                      compute_holidays(timestamps_[i]);
        double error = values_[i] - pred;
        sigma += error * error;
    }
    sigma = std::sqrt(sigma / values_.size());

    double z = 1.96;
    for (size_t i = 0; i < timestamps.size(); ++i) {
        result.lower_bound[i] = result.predictions[i] - z * sigma;
        result.upper_bound[i] = result.predictions[i] + z * sigma;
    }

    return result;
}

ForecastResult Prophet::forecast(int steps) const {
    if (!fitted_) {
        throw std::runtime_error("Model must be fitted before forecasting");
    }

    std::vector<double> future_timestamps(steps);
    double last_t = timestamps_.back();
    double step_size = (timestamps_.size() > 1) ?
                       (timestamps_.back() - timestamps_.front()) / (timestamps_.size() - 1) : 1.0;

    for (int i = 0; i < steps; ++i) {
        future_timestamps[i] = last_t + (i + 1) * step_size;
    }

    return predict(future_timestamps);
}

std::vector<double> Prophet::get_trend(const std::vector<double>& timestamps) const {
    std::vector<double> trend(timestamps.size());
    for (size_t i = 0; i < timestamps.size(); ++i) {
        trend[i] = compute_trend(timestamps[i]);
    }
    return trend;
}

std::vector<double> Prophet::get_seasonality(const std::vector<double>& timestamps) const {
    std::vector<double> seasonal(timestamps.size());
    for (size_t i = 0; i < timestamps.size(); ++i) {
        seasonal[i] = compute_seasonality(timestamps[i]);
    }
    return seasonal;
}

std::vector<double> Prophet::get_holidays(const std::vector<double>& timestamps) const {
    std::vector<double> holiday_effect(timestamps.size());
    for (size_t i = 0; i < timestamps.size(); ++i) {
        holiday_effect[i] = compute_holidays(timestamps[i]);
    }
    return holiday_effect;
}

// NeuralProphet Implementation
NeuralProphet::NeuralProphet(int n_lags, int n_forecasts,
                             bool yearly_seasonality, bool weekly_seasonality)
    : n_lags_(n_lags), n_forecasts_(n_forecasts),
      yearly_seasonality_(yearly_seasonality),
      weekly_seasonality_(weekly_seasonality),
      hidden_size_(32), b2_(0), fitted_(false) {}

void NeuralProphet::fit(const std::vector<double>& values, int epochs, double learning_rate) {
    if (values.size() < static_cast<size_t>(n_lags_ + n_forecasts_)) {
        throw std::invalid_argument("Insufficient data for NeuralProphet");
    }

    data_ = values;

    // Initialize weights
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, 0.1);

    int input_size = n_lags_;
    W1_.resize(hidden_size_, std::vector<double>(input_size));
    b1_.resize(hidden_size_, 0.0);
    W2_.resize(hidden_size_, 0.0);
    b2_ = 0.0;

    for (int i = 0; i < hidden_size_; ++i) {
        for (int j = 0; j < input_size; ++j) {
            W1_[i][j] = dist(gen);
        }
        b1_[i] = dist(gen) * 0.01;
        W2_[i] = dist(gen);
    }
    b2_ = dist(gen) * 0.01;

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        int n_samples = static_cast<int>(values.size()) - n_lags_;

        for (int t = 0; t < n_samples; ++t) {
            int idx = n_lags_ + t;
            // Create input from lags
            std::vector<double> input(n_lags_);
            for (int i = 0; i < n_lags_; ++i) {
                input[i] = values[idx - n_lags_ + i];
            }

            double target = values[idx];
            backward(input, target, learning_rate);

            std::vector<double> pred = forward(input);
            double error = target - pred[0];
            total_loss += error * error;
        }

        // Decay learning rate
        if ((epoch + 1) % 100 == 0) {
            learning_rate *= 0.95;
        }
    }

    fitted_ = true;
}

std::vector<double> NeuralProphet::forward(const std::vector<double>& input) const {
    // Hidden layer
    std::vector<double> hidden(hidden_size_);
    for (int i = 0; i < hidden_size_; ++i) {
        double sum = b1_[i];
        for (size_t j = 0; j < input.size(); ++j) {
            sum += W1_[i][j] * input[j];
        }
        hidden[i] = relu(sum);
    }

    // Output layer
    double output = b2_;
    for (int i = 0; i < hidden_size_; ++i) {
        output += W2_[i] * hidden[i];
    }

    return {output};
}

void NeuralProphet::backward(const std::vector<double>& input, double target,
                             double learning_rate) {
    // Forward pass with intermediate values
    std::vector<double> z1(hidden_size_);
    std::vector<double> a1(hidden_size_);

    for (int i = 0; i < hidden_size_; ++i) {
        z1[i] = b1_[i];
        for (size_t j = 0; j < input.size(); ++j) {
            z1[i] += W1_[i][j] * input[j];
        }
        a1[i] = relu(z1[i]);
    }

    double z2 = b2_;
    for (int i = 0; i < hidden_size_; ++i) {
        z2 += W2_[i] * a1[i];
    }

    // Backward pass
    double dz2 = z2 - target;  // MSE gradient

    // Gradient for W2 and b2
    for (int i = 0; i < hidden_size_; ++i) {
        W2_[i] -= learning_rate * dz2 * a1[i];
    }
    b2_ -= learning_rate * dz2;

    // Gradient for hidden layer
    for (int i = 0; i < hidden_size_; ++i) {
        double da1 = dz2 * W2_[i];
        double dz1 = da1 * relu_derivative(z1[i]);

        for (size_t j = 0; j < input.size(); ++j) {
            W1_[i][j] -= learning_rate * dz1 * input[j];
        }
        b1_[i] -= learning_rate * dz1;
    }
}

ForecastResult NeuralProphet::forecast(int steps) const {
    if (!fitted_) {
        throw std::runtime_error("Model must be fitted before forecasting");
    }

    ForecastResult result;
    result.predictions.resize(steps);
    result.lower_bound.resize(steps);
    result.upper_bound.resize(steps);

    std::vector<double> history = data_;

    for (int h = 0; h < steps; ++h) {
        std::vector<double> input(n_lags_);
        for (int i = 0; i < n_lags_; ++i) {
            input[i] = history[history.size() - n_lags_ + i];
        }

        std::vector<double> pred = forward(input);
        result.predictions[h] = pred[0];
        history.push_back(pred[0]);
    }

    // Simple confidence intervals
    double sigma = stats::std_dev(data_) * 0.1;
    double z = 1.96;
    for (int h = 0; h < steps; ++h) {
        double se = sigma * std::sqrt(h + 1);
        result.lower_bound[h] = result.predictions[h] - z * se;
        result.upper_bound[h] = result.predictions[h] + z * se;
    }

    return result;
}

// ============================================================
// Binary Serialization for Prophet
// ============================================================

constexpr uint32_t PROPHET_MAGIC = 0x50524F50;  // "PROP"
constexpr uint32_t PROPHET_VERSION = 1;

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

void write_string(std::ofstream& file, const std::string& str) {
    uint32_t size = static_cast<uint32_t>(str.size());
    file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    file.write(str.data(), size);
}

void read_string(std::ifstream& file, std::string& str) {
    uint32_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));
    str.resize(size);
    file.read(&str[0], size);
}

} // anonymous namespace

size_t Prophet::parameter_count() const {
    size_t count = 2;  // k_, m_
    count += delta_.size();  // changepoint deltas

    // Seasonality coefficients
    for (const auto& [name, coeffs] : seasonality_coeffs_) {
        count += coeffs.size();
    }

    // Holiday coefficients
    count += holiday_coeffs_.size();

    return count;
}

void Prophet::save(const std::string& filename) const {
    if (!fitted_) {
        throw std::runtime_error("Cannot save unfitted model");
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    file.write(reinterpret_cast<const char*>(&PROPHET_MAGIC), sizeof(PROPHET_MAGIC));
    file.write(reinterpret_cast<const char*>(&PROPHET_VERSION), sizeof(PROPHET_VERSION));

    // Growth type
    int gt = static_cast<int>(growth_type_);
    file.write(reinterpret_cast<const char*>(&gt), sizeof(gt));

    // Parameters
    file.write(reinterpret_cast<const char*>(&cap_), sizeof(cap_));
    file.write(reinterpret_cast<const char*>(&floor_), sizeof(floor_));
    file.write(reinterpret_cast<const char*>(&yearly_seasonality_), sizeof(yearly_seasonality_));
    file.write(reinterpret_cast<const char*>(&weekly_seasonality_), sizeof(weekly_seasonality_));
    file.write(reinterpret_cast<const char*>(&daily_seasonality_), sizeof(daily_seasonality_));
    file.write(reinterpret_cast<const char*>(&n_changepoints_), sizeof(n_changepoints_));
    file.write(reinterpret_cast<const char*>(&changepoint_range_), sizeof(changepoint_range_));
    file.write(reinterpret_cast<const char*>(&changepoint_prior_scale_), sizeof(changepoint_prior_scale_));

    // Fitted parameters
    file.write(reinterpret_cast<const char*>(&k_), sizeof(k_));
    file.write(reinterpret_cast<const char*>(&m_), sizeof(m_));
    file.write(reinterpret_cast<const char*>(&t_min_), sizeof(t_min_));
    file.write(reinterpret_cast<const char*>(&t_max_), sizeof(t_max_));

    write_vector(file, delta_);
    write_vector(file, timestamps_);
    write_vector(file, values_);

    // Changepoints
    uint32_t n_cp = static_cast<uint32_t>(changepoints_.size());
    file.write(reinterpret_cast<const char*>(&n_cp), sizeof(n_cp));
    for (const auto& cp : changepoints_) {
        file.write(reinterpret_cast<const char*>(&cp.timestamp), sizeof(cp.timestamp));
        file.write(reinterpret_cast<const char*>(&cp.rate_change), sizeof(cp.rate_change));
    }

    // Custom seasonalities
    uint32_t n_custom = static_cast<uint32_t>(custom_seasonalities_.size());
    file.write(reinterpret_cast<const char*>(&n_custom), sizeof(n_custom));
    for (const auto& cs : custom_seasonalities_) {
        write_string(file, cs.name);
        file.write(reinterpret_cast<const char*>(&cs.period), sizeof(cs.period));
        file.write(reinterpret_cast<const char*>(&cs.fourier_order), sizeof(cs.fourier_order));
    }

    // Seasonality coefficients
    uint32_t n_seas = static_cast<uint32_t>(seasonality_coeffs_.size());
    file.write(reinterpret_cast<const char*>(&n_seas), sizeof(n_seas));
    for (const auto& [name, coeffs] : seasonality_coeffs_) {
        write_string(file, name);
        write_vector(file, coeffs);
    }

    // Holiday coefficients
    uint32_t n_hol = static_cast<uint32_t>(holiday_coeffs_.size());
    file.write(reinterpret_cast<const char*>(&n_hol), sizeof(n_hol));
    for (const auto& [name, coeff] : holiday_coeffs_) {
        write_string(file, name);
        file.write(reinterpret_cast<const char*>(&coeff), sizeof(coeff));
    }

    file.close();
}

void Prophet::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }

    uint32_t magic, version;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != PROPHET_MAGIC) {
        throw std::runtime_error("Invalid file format: not a Prophet model file");
    }

    int gt;
    file.read(reinterpret_cast<char*>(&gt), sizeof(gt));
    growth_type_ = static_cast<GrowthType>(gt);

    file.read(reinterpret_cast<char*>(&cap_), sizeof(cap_));
    file.read(reinterpret_cast<char*>(&floor_), sizeof(floor_));
    file.read(reinterpret_cast<char*>(&yearly_seasonality_), sizeof(yearly_seasonality_));
    file.read(reinterpret_cast<char*>(&weekly_seasonality_), sizeof(weekly_seasonality_));
    file.read(reinterpret_cast<char*>(&daily_seasonality_), sizeof(daily_seasonality_));
    file.read(reinterpret_cast<char*>(&n_changepoints_), sizeof(n_changepoints_));
    file.read(reinterpret_cast<char*>(&changepoint_range_), sizeof(changepoint_range_));
    file.read(reinterpret_cast<char*>(&changepoint_prior_scale_), sizeof(changepoint_prior_scale_));

    file.read(reinterpret_cast<char*>(&k_), sizeof(k_));
    file.read(reinterpret_cast<char*>(&m_), sizeof(m_));
    file.read(reinterpret_cast<char*>(&t_min_), sizeof(t_min_));
    file.read(reinterpret_cast<char*>(&t_max_), sizeof(t_max_));

    read_vector(file, delta_);
    read_vector(file, timestamps_);
    read_vector(file, values_);

    // Changepoints
    uint32_t n_cp;
    file.read(reinterpret_cast<char*>(&n_cp), sizeof(n_cp));
    changepoints_.resize(n_cp);
    for (uint32_t i = 0; i < n_cp; ++i) {
        file.read(reinterpret_cast<char*>(&changepoints_[i].timestamp), sizeof(double));
        file.read(reinterpret_cast<char*>(&changepoints_[i].rate_change), sizeof(double));
    }

    // Custom seasonalities
    uint32_t n_custom;
    file.read(reinterpret_cast<char*>(&n_custom), sizeof(n_custom));
    custom_seasonalities_.resize(n_custom);
    for (uint32_t i = 0; i < n_custom; ++i) {
        read_string(file, custom_seasonalities_[i].name);
        file.read(reinterpret_cast<char*>(&custom_seasonalities_[i].period), sizeof(double));
        file.read(reinterpret_cast<char*>(&custom_seasonalities_[i].fourier_order), sizeof(int));
    }

    // Seasonality coefficients
    uint32_t n_seas;
    file.read(reinterpret_cast<char*>(&n_seas), sizeof(n_seas));
    seasonality_coeffs_.clear();
    for (uint32_t i = 0; i < n_seas; ++i) {
        std::string name;
        std::vector<double> coeffs;
        read_string(file, name);
        read_vector(file, coeffs);
        seasonality_coeffs_[name] = coeffs;
    }

    // Holiday coefficients
    uint32_t n_hol;
    file.read(reinterpret_cast<char*>(&n_hol), sizeof(n_hol));
    holiday_coeffs_.clear();
    for (uint32_t i = 0; i < n_hol; ++i) {
        std::string name;
        double coeff;
        read_string(file, name);
        file.read(reinterpret_cast<char*>(&coeff), sizeof(coeff));
        holiday_coeffs_[name] = coeff;
    }

    fitted_ = true;
    file.close();
}

} // namespace ts
