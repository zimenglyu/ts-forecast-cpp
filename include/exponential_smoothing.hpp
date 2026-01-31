#ifndef TS_EXPONENTIAL_SMOOTHING_HPP
#define TS_EXPONENTIAL_SMOOTHING_HPP

#include "utils.hpp"
#include <vector>

namespace ts {

/**
 * Simple Exponential Smoothing (SES)
 *
 * For data with no trend or seasonality.
 * Forecast equation: y_hat_{t+h} = alpha * y_t + (1 - alpha) * y_hat_t
 */
class SimpleExponentialSmoothing {
public:
    /**
     * Constructor
     * @param alpha Smoothing parameter (0 < alpha < 1), if -1 it will be optimized
     */
    explicit SimpleExponentialSmoothing(double alpha = -1.0);

    void fit(const std::vector<double>& data);
    ForecastResult forecast(int steps) const;
    std::vector<double> fitted_values() const;

    double alpha() const { return alpha_; }
    bool is_fitted() const { return is_fitted_; }

private:
    double alpha_;
    double level_;
    std::vector<double> data_;
    std::vector<double> fitted_vals_;
    bool is_fitted_;

    void optimize_alpha();
};

/**
 * Holt's Linear Trend Method (Double Exponential Smoothing)
 *
 * For data with trend but no seasonality.
 * Level: l_t = alpha * y_t + (1 - alpha) * (l_{t-1} + b_{t-1})
 * Trend: b_t = beta * (l_t - l_{t-1}) + (1 - beta) * b_{t-1}
 * Forecast: y_hat_{t+h} = l_t + h * b_t
 */
class HoltLinear {
public:
    /**
     * Constructor
     * @param alpha Level smoothing parameter
     * @param beta Trend smoothing parameter
     * @param damped Use damped trend if true
     * @param phi Damping parameter (only used if damped=true)
     */
    HoltLinear(double alpha = -1.0, double beta = -1.0,
               bool damped = false, double phi = 0.98);

    void fit(const std::vector<double>& data);
    ForecastResult forecast(int steps) const;
    std::vector<double> fitted_values() const;

    double alpha() const { return alpha_; }
    double beta() const { return beta_; }
    bool is_fitted() const { return is_fitted_; }

private:
    double alpha_, beta_;
    double level_, trend_;
    bool damped_;
    double phi_;
    std::vector<double> data_;
    std::vector<double> fitted_vals_;
    bool is_fitted_;

    void optimize_parameters();
};

/**
 * Holt-Winters Seasonal Method (Triple Exponential Smoothing)
 *
 * For data with trend and seasonality.
 *
 * Additive seasonality:
 * Level: l_t = alpha * (y_t - s_{t-m}) + (1 - alpha) * (l_{t-1} + b_{t-1})
 * Trend: b_t = beta * (l_t - l_{t-1}) + (1 - beta) * b_{t-1}
 * Season: s_t = gamma * (y_t - l_{t-1} - b_{t-1}) + (1 - gamma) * s_{t-m}
 * Forecast: y_hat_{t+h} = l_t + h * b_t + s_{t+h-m(k+1)}
 *
 * Multiplicative seasonality:
 * Level: l_t = alpha * (y_t / s_{t-m}) + (1 - alpha) * (l_{t-1} + b_{t-1})
 * Trend: b_t = beta * (l_t - l_{t-1}) + (1 - beta) * b_{t-1}
 * Season: s_t = gamma * (y_t / (l_{t-1} + b_{t-1})) + (1 - gamma) * s_{t-m}
 * Forecast: y_hat_{t+h} = (l_t + h * b_t) * s_{t+h-m(k+1)}
 */
class HoltWinters {
public:
    enum class SeasonalType { ADDITIVE, MULTIPLICATIVE };

    /**
     * Constructor
     * @param period Seasonal period (e.g., 12 for monthly, 4 for quarterly)
     * @param seasonal_type Type of seasonality
     * @param alpha Level smoothing parameter (optimized if -1)
     * @param beta Trend smoothing parameter (optimized if -1)
     * @param gamma Seasonal smoothing parameter (optimized if -1)
     * @param damped Use damped trend
     */
    HoltWinters(int period, SeasonalType seasonal_type = SeasonalType::ADDITIVE,
                double alpha = -1.0, double beta = -1.0, double gamma = -1.0,
                bool damped = false, double phi = 0.98);

    void fit(const std::vector<double>& data);
    ForecastResult forecast(int steps) const;
    std::vector<double> fitted_values() const;

    double alpha() const { return alpha_; }
    double beta() const { return beta_; }
    double gamma() const { return gamma_; }
    int period() const { return period_; }
    bool is_fitted() const { return is_fitted_; }

private:
    int period_;
    SeasonalType seasonal_type_;
    double alpha_, beta_, gamma_;
    double level_, trend_;
    std::vector<double> seasonal_;
    bool damped_;
    double phi_;

    std::vector<double> data_;
    std::vector<double> fitted_vals_;
    bool is_fitted_;

    void initialize_components();
    void optimize_parameters();
    double compute_sse(double alpha, double beta, double gamma) const;
};

/**
 * ETS (Error, Trend, Seasonality) State Space Model
 *
 * General framework for exponential smoothing models.
 * Automatically selects the best model from the ETS family.
 */
class ETS {
public:
    enum class ErrorType { ADDITIVE, MULTIPLICATIVE };
    enum class TrendType { NONE, ADDITIVE, MULTIPLICATIVE, ADDITIVE_DAMPED, MULTIPLICATIVE_DAMPED };
    enum class SeasonType { NONE, ADDITIVE, MULTIPLICATIVE };

    /**
     * Constructor for automatic model selection
     * @param period Seasonal period (0 for non-seasonal)
     */
    explicit ETS(int period = 0);

    /**
     * Constructor for specific model
     */
    ETS(ErrorType error, TrendType trend, SeasonType season, int period = 0);

    void fit(const std::vector<double>& data);
    ForecastResult forecast(int steps) const;
    std::vector<double> fitted_values() const;

    std::string model_type() const;
    double aic() const { return aic_; }
    bool is_fitted() const { return is_fitted_; }

private:
    int period_;
    ErrorType error_type_;
    TrendType trend_type_;
    SeasonType season_type_;
    bool auto_select_;

    double alpha_, beta_, gamma_, phi_;
    double level_, trend_;
    std::vector<double> seasonal_;

    std::vector<double> data_;
    std::vector<double> fitted_vals_;
    double aic_;
    bool is_fitted_;

    void auto_fit(const std::vector<double>& data);
    void fit_specific_model(const std::vector<double>& data);
    double compute_aic() const;
};

} // namespace ts

#endif // TS_EXPONENTIAL_SMOOTHING_HPP
