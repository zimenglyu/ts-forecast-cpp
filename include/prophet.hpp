#ifndef TS_PROPHET_HPP
#define TS_PROPHET_HPP

#include "utils.hpp"
#include <vector>
#include <string>
#include <map>

namespace ts {

/**
 * Prophet-like Additive Regression Model
 *
 * A simplified implementation inspired by Facebook's Prophet.
 * The model decomposes time series into:
 * y(t) = g(t) + s(t) + h(t) + epsilon_t
 *
 * where:
 * - g(t): trend (linear or logistic growth)
 * - s(t): seasonality (Fourier series)
 * - h(t): holiday/event effects
 * - epsilon_t: error term
 *
 * Note: This is a lightweight implementation suitable for embedded systems.
 * It doesn't include all Prophet features but captures the core functionality.
 */
class Prophet {
public:
    enum class GrowthType { LINEAR, LOGISTIC };

    struct Holiday {
        std::vector<double> dates;  // Timestamps of holidays
        std::string name;
        int lower_window;           // Days before holiday to include
        int upper_window;           // Days after holiday to include

        Holiday() : lower_window(0), upper_window(0) {}
        Holiday(const std::string& n, const std::vector<double>& d,
                int lower = 0, int upper = 0)
            : dates(d), name(n), lower_window(lower), upper_window(upper) {}
    };

    struct Changepoint {
        double timestamp;
        double rate_change;  // Change in growth rate at this point
    };

    /**
     * Constructor
     * @param growth Type of trend growth
     * @param yearly_seasonality Enable yearly seasonality
     * @param weekly_seasonality Enable weekly seasonality
     * @param daily_seasonality Enable daily seasonality
     */
    Prophet(GrowthType growth = GrowthType::LINEAR,
            bool yearly_seasonality = true,
            bool weekly_seasonality = true,
            bool daily_seasonality = false);

    /**
     * Set carrying capacity for logistic growth
     */
    void set_capacity(double cap) { cap_ = cap; }

    /**
     * Set floor for logistic growth
     */
    void set_floor(double floor) { floor_ = floor; }

    /**
     * Add custom seasonality
     * @param name Name of the seasonality
     * @param period Period in days (e.g., 365.25 for yearly)
     * @param fourier_order Number of Fourier terms
     */
    void add_seasonality(const std::string& name, double period, int fourier_order);

    /**
     * Add holidays/special events
     */
    void add_holiday(const Holiday& holiday);
    void add_holidays(const std::vector<Holiday>& holidays);

    /**
     * Set number of potential changepoints for piecewise trend
     * @param n_changepoints Number of changepoints
     * @param changepoint_range Fraction of history to place changepoints
     */
    void set_changepoints(int n_changepoints, double changepoint_range = 0.8);

    /**
     * Fit the model
     * @param timestamps Vector of timestamps (e.g., days since start)
     * @param values Vector of corresponding values
     */
    void fit(const std::vector<double>& timestamps, const std::vector<double>& values);

    /**
     * Convenience fit for equally spaced data
     * @param values Vector of values (assumes unit spacing)
     */
    void fit(const std::vector<double>& values);

    /**
     * Make predictions for given timestamps
     */
    ForecastResult predict(const std::vector<double>& timestamps) const;

    /**
     * Forecast future values
     * @param steps Number of future time steps
     */
    ForecastResult forecast(int steps) const;

    /**
     * Get trend component
     */
    std::vector<double> get_trend(const std::vector<double>& timestamps) const;

    /**
     * Get seasonality component
     */
    std::vector<double> get_seasonality(const std::vector<double>& timestamps) const;

    /**
     * Get holiday component
     */
    std::vector<double> get_holidays(const std::vector<double>& timestamps) const;

    /**
     * Get detected changepoints
     */
    std::vector<Changepoint> get_changepoints() const { return changepoints_; }

    bool is_fitted() const { return fitted_; }

private:
    GrowthType growth_type_;
    double cap_, floor_;

    bool yearly_seasonality_;
    bool weekly_seasonality_;
    bool daily_seasonality_;

    struct SeasonalitySpec {
        std::string name;
        double period;
        int fourier_order;
    };
    std::vector<SeasonalitySpec> custom_seasonalities_;
    std::vector<Holiday> holidays_;

    int n_changepoints_;
    double changepoint_range_;
    double changepoint_prior_scale_;

    // Fitted parameters
    double k_;              // Base growth rate
    double m_;              // Offset
    std::vector<double> delta_;  // Rate changes at changepoints
    std::vector<Changepoint> changepoints_;

    std::map<std::string, std::vector<double>> seasonality_coeffs_;
    std::map<std::string, double> holiday_coeffs_;

    std::vector<double> timestamps_;
    std::vector<double> values_;
    double t_min_, t_max_;
    bool fitted_;

    // Internal methods
    void initialize_changepoints();
    void fit_trend();
    void fit_seasonality();
    void fit_holidays();

    double compute_trend(double t) const;
    double compute_seasonality(double t) const;
    double compute_holidays(double t) const;

    std::vector<double> fourier_series(double t, double period, int order) const;
    double piecewise_linear(double t) const;
    double piecewise_logistic(double t) const;
};

/**
 * Simplified Neural Prophet-like model
 *
 * Uses simple neural network for autoregression combined with
 * Prophet-style decomposition.
 */
class NeuralProphet {
public:
    NeuralProphet(int n_lags = 7, int n_forecasts = 1,
                  bool yearly_seasonality = true,
                  bool weekly_seasonality = true);

    void fit(const std::vector<double>& values, int epochs = 100,
             double learning_rate = 0.01);
    ForecastResult forecast(int steps) const;

    bool is_fitted() const { return fitted_; }

private:
    int n_lags_;
    int n_forecasts_;
    bool yearly_seasonality_;
    bool weekly_seasonality_;

    // Simple neural network weights (single hidden layer)
    std::vector<std::vector<double>> W1_;  // Input to hidden
    std::vector<double> b1_;               // Hidden bias
    std::vector<double> W2_;               // Hidden to output
    double b2_;                            // Output bias

    int hidden_size_;

    std::vector<double> data_;
    bool fitted_;

    std::vector<double> forward(const std::vector<double>& input) const;
    void backward(const std::vector<double>& input, double target,
                  double learning_rate);
    double relu(double x) const { return x > 0 ? x : 0; }
    double relu_derivative(double x) const { return x > 0 ? 1.0 : 0.0; }
};

} // namespace ts

#endif // TS_PROPHET_HPP
