#include "exponential_smoothing.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>

namespace ts {

// Simple Exponential Smoothing
SimpleExponentialSmoothing::SimpleExponentialSmoothing(double alpha)
    : alpha_(alpha), level_(0), is_fitted_(false) {}

void SimpleExponentialSmoothing::fit(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data cannot be empty");
    }

    data_ = data;

    if (alpha_ < 0 || alpha_ > 1) {
        optimize_alpha();
    }

    // Initialize level with first observation
    level_ = data_[0];
    fitted_vals_.resize(data_.size());
    fitted_vals_[0] = level_;

    // Apply exponential smoothing
    for (size_t t = 1; t < data_.size(); ++t) {
        level_ = alpha_ * data_[t] + (1 - alpha_) * level_;
        fitted_vals_[t] = level_;
    }

    is_fitted_ = true;
}

void SimpleExponentialSmoothing::optimize_alpha() {
    double best_alpha = 0.5;
    double best_sse = std::numeric_limits<double>::max();

    for (double a = 0.01; a <= 0.99; a += 0.01) {
        double level = data_[0];
        double sse = 0.0;

        for (size_t t = 1; t < data_.size(); ++t) {
            double error = data_[t] - level;
            sse += error * error;
            level = a * data_[t] + (1 - a) * level;
        }

        if (sse < best_sse) {
            best_sse = sse;
            best_alpha = a;
        }
    }

    alpha_ = best_alpha;
}

ForecastResult SimpleExponentialSmoothing::forecast(int steps) const {
    if (!is_fitted_) {
        throw std::runtime_error("Model must be fitted before forecasting");
    }

    ForecastResult result;
    result.predictions.resize(steps, level_);
    result.lower_bound.resize(steps);
    result.upper_bound.resize(steps);
    result.confidence_level = 0.95;

    double sigma = 0.0;
    for (size_t t = 1; t < data_.size(); ++t) {
        double error = data_[t] - fitted_vals_[t - 1];
        sigma += error * error;
    }
    sigma = std::sqrt(sigma / (data_.size() - 1));

    double z = 1.96;
    for (int h = 0; h < steps; ++h) {
        double se = sigma * std::sqrt(1 + h * alpha_ * alpha_);
        result.lower_bound[h] = level_ - z * se;
        result.upper_bound[h] = level_ + z * se;
    }

    return result;
}

std::vector<double> SimpleExponentialSmoothing::fitted_values() const {
    return fitted_vals_;
}

// Holt's Linear Trend
HoltLinear::HoltLinear(double alpha, double beta, bool damped, double phi)
    : alpha_(alpha), beta_(beta), level_(0), trend_(0),
      damped_(damped), phi_(phi), is_fitted_(false) {}

void HoltLinear::fit(const std::vector<double>& data) {
    if (data.size() < 2) {
        throw std::invalid_argument("Need at least 2 observations");
    }

    data_ = data;

    if (alpha_ < 0 || alpha_ > 1 || beta_ < 0 || beta_ > 1) {
        optimize_parameters();
    }

    // Initialize
    level_ = data_[0];
    trend_ = data_[1] - data_[0];

    fitted_vals_.resize(data_.size());
    fitted_vals_[0] = level_;

    for (size_t t = 1; t < data_.size(); ++t) {
        double prev_level = level_;
        double damping = damped_ ? phi_ : 1.0;

        level_ = alpha_ * data_[t] + (1 - alpha_) * (prev_level + damping * trend_);
        trend_ = beta_ * (level_ - prev_level) + (1 - beta_) * damping * trend_;
        fitted_vals_[t] = level_ + damping * trend_;
    }

    is_fitted_ = true;
}

void HoltLinear::optimize_parameters() {
    double best_alpha = 0.5, best_beta = 0.1;
    double best_sse = std::numeric_limits<double>::max();

    for (double a = 0.05; a <= 0.95; a += 0.05) {
        for (double b = 0.01; b <= 0.5; b += 0.02) {
            double level = data_[0];
            double trend = data_[1] - data_[0];
            double sse = 0.0;

            for (size_t t = 1; t < data_.size(); ++t) {
                double pred = level + (damped_ ? phi_ : 1.0) * trend;
                double error = data_[t] - pred;
                sse += error * error;

                double prev_level = level;
                double damping = damped_ ? phi_ : 1.0;
                level = a * data_[t] + (1 - a) * (prev_level + damping * trend);
                trend = b * (level - prev_level) + (1 - b) * damping * trend;
            }

            if (sse < best_sse) {
                best_sse = sse;
                best_alpha = a;
                best_beta = b;
            }
        }
    }

    alpha_ = best_alpha;
    beta_ = best_beta;
}

ForecastResult HoltLinear::forecast(int steps) const {
    if (!is_fitted_) {
        throw std::runtime_error("Model must be fitted before forecasting");
    }

    ForecastResult result;
    result.predictions.resize(steps);
    result.lower_bound.resize(steps);
    result.upper_bound.resize(steps);

    for (int h = 0; h < steps; ++h) {
        double damping_sum = 0.0;
        if (damped_) {
            for (int i = 1; i <= h + 1; ++i) {
                damping_sum += std::pow(phi_, i);
            }
        } else {
            damping_sum = h + 1;
        }
        result.predictions[h] = level_ + damping_sum * trend_;
    }

    // Estimate forecast error variance
    double sigma = 0.0;
    for (size_t t = 1; t < data_.size(); ++t) {
        double error = data_[t] - fitted_vals_[t - 1];
        sigma += error * error;
    }
    sigma = std::sqrt(sigma / (data_.size() - 1));

    double z = 1.96;
    for (int h = 0; h < steps; ++h) {
        double se = sigma * std::sqrt(1 + h);
        result.lower_bound[h] = result.predictions[h] - z * se;
        result.upper_bound[h] = result.predictions[h] + z * se;
    }

    return result;
}

std::vector<double> HoltLinear::fitted_values() const {
    return fitted_vals_;
}

// Holt-Winters
HoltWinters::HoltWinters(int period, SeasonalType seasonal_type,
                         double alpha, double beta, double gamma,
                         bool damped, double phi)
    : period_(period), seasonal_type_(seasonal_type),
      alpha_(alpha), beta_(beta), gamma_(gamma),
      level_(0), trend_(0), damped_(damped), phi_(phi), is_fitted_(false) {
    seasonal_.resize(period_, 0.0);
}

void HoltWinters::fit(const std::vector<double>& data) {
    if (data.size() < static_cast<size_t>(2 * period_)) {
        throw std::invalid_argument("Need at least 2 full seasons of data");
    }

    data_ = data;
    initialize_components();

    if (alpha_ < 0 || alpha_ > 1 || beta_ < 0 || beta_ > 1 || gamma_ < 0 || gamma_ > 1) {
        optimize_parameters();
    }

    fitted_vals_.resize(data_.size());

    // Re-initialize with optimized parameters
    initialize_components();

    for (size_t t = period_; t < data_.size(); ++t) {
        int season_idx = t % period_;
        double prev_level = level_;
        double damping = damped_ ? phi_ : 1.0;

        if (seasonal_type_ == SeasonalType::ADDITIVE) {
            level_ = alpha_ * (data_[t] - seasonal_[season_idx]) +
                     (1 - alpha_) * (prev_level + damping * trend_);
            trend_ = beta_ * (level_ - prev_level) + (1 - beta_) * damping * trend_;
            seasonal_[season_idx] = gamma_ * (data_[t] - prev_level - damping * trend_) +
                                     (1 - gamma_) * seasonal_[season_idx];
            fitted_vals_[t] = level_ + damping * trend_ + seasonal_[season_idx];
        } else {
            level_ = alpha_ * (data_[t] / seasonal_[season_idx]) +
                     (1 - alpha_) * (prev_level + damping * trend_);
            trend_ = beta_ * (level_ - prev_level) + (1 - beta_) * damping * trend_;
            seasonal_[season_idx] = gamma_ * (data_[t] / (prev_level + damping * trend_)) +
                                     (1 - gamma_) * seasonal_[season_idx];
            fitted_vals_[t] = (level_ + damping * trend_) * seasonal_[season_idx];
        }
    }

    // Fill initial fitted values
    for (size_t t = 0; t < static_cast<size_t>(period_); ++t) {
        fitted_vals_[t] = data_[t];
    }

    is_fitted_ = true;
}

void HoltWinters::initialize_components() {
    // Initialize level as mean of first season
    level_ = 0.0;
    for (int i = 0; i < period_; ++i) {
        level_ += data_[i];
    }
    level_ /= period_;

    // Initialize trend
    double sum_trend = 0.0;
    for (int i = 0; i < period_; ++i) {
        sum_trend += (data_[period_ + i] - data_[i]) / period_;
    }
    trend_ = sum_trend / period_;

    // Initialize seasonal components
    if (seasonal_type_ == SeasonalType::ADDITIVE) {
        for (int i = 0; i < period_; ++i) {
            double sum = 0.0;
            int count = 0;
            for (size_t j = i; j < data_.size(); j += period_) {
                sum += data_[j] - (level_ + (j / period_) * trend_);
                count++;
            }
            seasonal_[i] = sum / count;
        }
    } else {
        for (int i = 0; i < period_; ++i) {
            double sum = 0.0;
            int count = 0;
            for (size_t j = i; j < data_.size(); j += period_) {
                double base = level_ + (j / period_) * trend_;
                if (base > 0) {
                    sum += data_[j] / base;
                    count++;
                }
            }
            seasonal_[i] = (count > 0) ? sum / count : 1.0;
        }
    }
}

void HoltWinters::optimize_parameters() {
    double best_alpha = 0.3, best_beta = 0.1, best_gamma = 0.1;
    double best_sse = std::numeric_limits<double>::max();

    // Grid search (coarse)
    for (double a = 0.1; a <= 0.9; a += 0.1) {
        for (double b = 0.01; b <= 0.3; b += 0.05) {
            for (double g = 0.01; g <= 0.5; g += 0.05) {
                double sse = compute_sse(a, b, g);
                if (sse < best_sse) {
                    best_sse = sse;
                    best_alpha = a;
                    best_beta = b;
                    best_gamma = g;
                }
            }
        }
    }

    alpha_ = best_alpha;
    beta_ = best_beta;
    gamma_ = best_gamma;
}

double HoltWinters::compute_sse(double alpha, double beta, double gamma) const {
    std::vector<double> seasonal(period_);
    double level = 0.0, trend = 0.0;

    // Initialize
    for (int i = 0; i < period_; ++i) {
        level += data_[i];
    }
    level /= period_;

    for (int i = 0; i < period_; ++i) {
        double sum = (data_[period_ + i] - data_[i]) / period_;
        trend += sum;
    }
    trend /= period_;

    if (seasonal_type_ == SeasonalType::ADDITIVE) {
        for (int i = 0; i < period_; ++i) {
            seasonal[i] = data_[i] - level;
        }
    } else {
        for (int i = 0; i < period_; ++i) {
            seasonal[i] = (level > 0) ? data_[i] / level : 1.0;
        }
    }

    double sse = 0.0;
    double damping = damped_ ? phi_ : 1.0;

    for (size_t t = period_; t < data_.size(); ++t) {
        int idx = t % period_;
        double pred;

        if (seasonal_type_ == SeasonalType::ADDITIVE) {
            pred = level + damping * trend + seasonal[idx];
        } else {
            pred = (level + damping * trend) * seasonal[idx];
        }

        double error = data_[t] - pred;
        sse += error * error;

        double prev_level = level;
        if (seasonal_type_ == SeasonalType::ADDITIVE) {
            level = alpha * (data_[t] - seasonal[idx]) +
                    (1 - alpha) * (prev_level + damping * trend);
            trend = beta * (level - prev_level) + (1 - beta) * damping * trend;
            seasonal[idx] = gamma * (data_[t] - prev_level - damping * trend) +
                           (1 - gamma) * seasonal[idx];
        } else {
            level = alpha * (data_[t] / seasonal[idx]) +
                    (1 - alpha) * (prev_level + damping * trend);
            trend = beta * (level - prev_level) + (1 - beta) * damping * trend;
            seasonal[idx] = gamma * (data_[t] / (prev_level + damping * trend)) +
                           (1 - gamma) * seasonal[idx];
        }
    }

    return sse;
}

ForecastResult HoltWinters::forecast(int steps) const {
    if (!is_fitted_) {
        throw std::runtime_error("Model must be fitted before forecasting");
    }

    ForecastResult result;
    result.predictions.resize(steps);
    result.lower_bound.resize(steps);
    result.upper_bound.resize(steps);

    for (int h = 0; h < steps; ++h) {
        int season_idx = (data_.size() + h) % period_;
        double damping_sum = 0.0;

        if (damped_) {
            for (int i = 1; i <= h + 1; ++i) {
                damping_sum += std::pow(phi_, i);
            }
        } else {
            damping_sum = h + 1;
        }

        if (seasonal_type_ == SeasonalType::ADDITIVE) {
            result.predictions[h] = level_ + damping_sum * trend_ + seasonal_[season_idx];
        } else {
            result.predictions[h] = (level_ + damping_sum * trend_) * seasonal_[season_idx];
        }
    }

    // Confidence intervals
    double sigma = 0.0;
    for (size_t t = period_; t < data_.size(); ++t) {
        double error = data_[t] - fitted_vals_[t];
        sigma += error * error;
    }
    sigma = std::sqrt(sigma / (data_.size() - period_));

    double z = 1.96;
    for (int h = 0; h < steps; ++h) {
        double se = sigma * std::sqrt(1 + h);
        result.lower_bound[h] = result.predictions[h] - z * se;
        result.upper_bound[h] = result.predictions[h] + z * se;
    }

    return result;
}

std::vector<double> HoltWinters::fitted_values() const {
    return fitted_vals_;
}

// ETS Model
ETS::ETS(int period)
    : period_(period), auto_select_(true),
      alpha_(0.5), beta_(0.1), gamma_(0.1), phi_(0.98),
      level_(0), trend_(0), aic_(0), is_fitted_(false) {
    error_type_ = ErrorType::ADDITIVE;
    trend_type_ = TrendType::NONE;
    season_type_ = SeasonType::NONE;
}

ETS::ETS(ErrorType error, TrendType trend, SeasonType season, int period)
    : period_(period), error_type_(error), trend_type_(trend), season_type_(season),
      auto_select_(false), alpha_(0.5), beta_(0.1), gamma_(0.1), phi_(0.98),
      level_(0), trend_(0), aic_(0), is_fitted_(false) {}

void ETS::fit(const std::vector<double>& data) {
    data_ = data;

    if (auto_select_) {
        auto_fit(data);
    } else {
        fit_specific_model(data);
    }

    aic_ = compute_aic();
    is_fitted_ = true;
}

void ETS::auto_fit(const std::vector<double>& data) {
    double best_aic = std::numeric_limits<double>::max();

    std::vector<TrendType> trends = {TrendType::NONE, TrendType::ADDITIVE};
    std::vector<SeasonType> seasons = {SeasonType::NONE};

    if (period_ > 1 && data.size() >= static_cast<size_t>(2 * period_)) {
        seasons.push_back(SeasonType::ADDITIVE);
    }

    for (auto t : trends) {
        for (auto s : seasons) {
            trend_type_ = t;
            season_type_ = s;

            try {
                fit_specific_model(data);
                double aic = compute_aic();

                if (aic < best_aic) {
                    best_aic = aic;
                    // Store best configuration
                }
            } catch (...) {
                continue;
            }
        }
    }

    aic_ = best_aic;
}

void ETS::fit_specific_model(const std::vector<double>& data) {
    // Use appropriate exponential smoothing based on model type
    if (trend_type_ == TrendType::NONE && season_type_ == SeasonType::NONE) {
        SimpleExponentialSmoothing ses(alpha_);
        ses.fit(data);
        fitted_vals_ = ses.fitted_values();
        level_ = stats::mean(data);
    } else if (season_type_ == SeasonType::NONE) {
        bool damped = (trend_type_ == TrendType::ADDITIVE_DAMPED ||
                       trend_type_ == TrendType::MULTIPLICATIVE_DAMPED);
        HoltLinear holt(alpha_, beta_, damped, phi_);
        holt.fit(data);
        fitted_vals_ = holt.fitted_values();
        level_ = stats::mean(data);
    } else {
        HoltWinters::SeasonalType st = (season_type_ == SeasonType::ADDITIVE)
            ? HoltWinters::SeasonalType::ADDITIVE
            : HoltWinters::SeasonalType::MULTIPLICATIVE;
        bool damped = (trend_type_ == TrendType::ADDITIVE_DAMPED ||
                       trend_type_ == TrendType::MULTIPLICATIVE_DAMPED);
        HoltWinters hw(period_, st, alpha_, beta_, gamma_, damped, phi_);
        hw.fit(data);
        fitted_vals_ = hw.fitted_values();
        level_ = stats::mean(data);
    }
}

double ETS::compute_aic() const {
    if (fitted_vals_.empty()) return std::numeric_limits<double>::max();

    double sse = 0.0;
    for (size_t t = 0; t < data_.size(); ++t) {
        double error = data_[t] - fitted_vals_[t];
        sse += error * error;
    }

    int k = 2;  // alpha + initial level
    if (trend_type_ != TrendType::NONE) k += 2;
    if (season_type_ != SeasonType::NONE) k += period_ + 1;

    int n = static_cast<int>(data_.size());
    return n * std::log(sse / n) + 2 * k;
}

ForecastResult ETS::forecast(int steps) const {
    if (!is_fitted_) {
        throw std::runtime_error("Model must be fitted before forecasting");
    }

    ForecastResult result;
    result.predictions.resize(steps, level_);
    result.lower_bound.resize(steps);
    result.upper_bound.resize(steps);

    // Simple forecast using last level
    double sigma = 0.0;
    for (size_t t = 1; t < data_.size(); ++t) {
        double error = data_[t] - fitted_vals_[t];
        sigma += error * error;
    }
    sigma = std::sqrt(sigma / (data_.size() - 1));

    double z = 1.96;
    for (int h = 0; h < steps; ++h) {
        result.lower_bound[h] = result.predictions[h] - z * sigma * std::sqrt(h + 1);
        result.upper_bound[h] = result.predictions[h] + z * sigma * std::sqrt(h + 1);
    }

    return result;
}

std::vector<double> ETS::fitted_values() const {
    return fitted_vals_;
}

std::string ETS::model_type() const {
    std::string error = (error_type_ == ErrorType::ADDITIVE) ? "A" : "M";
    std::string trend;
    switch (trend_type_) {
        case TrendType::NONE: trend = "N"; break;
        case TrendType::ADDITIVE: trend = "A"; break;
        case TrendType::MULTIPLICATIVE: trend = "M"; break;
        case TrendType::ADDITIVE_DAMPED: trend = "Ad"; break;
        case TrendType::MULTIPLICATIVE_DAMPED: trend = "Md"; break;
    }
    std::string season;
    switch (season_type_) {
        case SeasonType::NONE: season = "N"; break;
        case SeasonType::ADDITIVE: season = "A"; break;
        case SeasonType::MULTIPLICATIVE: season = "M"; break;
    }
    return "ETS(" + error + "," + trend + "," + season + ")";
}

} // namespace ts
