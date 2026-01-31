#ifndef TS_ARIMA_HPP
#define TS_ARIMA_HPP

#include "utils.hpp"
#include <vector>

namespace ts {

/**
 * ARIMA (AutoRegressive Integrated Moving Average) Model
 *
 * ARIMA(p, d, q) where:
 * - p: order of autoregressive part
 * - d: degree of differencing
 * - q: order of moving average part
 *
 * The model equation is:
 * (1 - sum(phi_i * L^i)) * (1 - L)^d * y_t = (1 + sum(theta_j * L^j)) * epsilon_t
 *
 * where L is the lag operator.
 */
class ARIMA {
public:
    /**
     * Constructor
     * @param p Order of AR component
     * @param d Order of differencing
     * @param q Order of MA component
     */
    ARIMA(int p = 1, int d = 1, int q = 1);

    /**
     * Fit the model to data
     * @param data Time series data
     * @param max_iter Maximum iterations for optimization
     * @param tol Convergence tolerance
     */
    void fit(const std::vector<double>& data, int max_iter = 1000, double tol = 1e-6);

    /**
     * Forecast future values
     * @param steps Number of steps to forecast
     * @return ForecastResult with predictions and confidence intervals
     */
    ForecastResult forecast(int steps) const;

    /**
     * Get fitted values (in-sample predictions)
     */
    std::vector<double> fitted_values() const;

    /**
     * Get residuals
     */
    std::vector<double> residuals() const;

    // Getters for model parameters
    std::vector<double> ar_coefficients() const { return phi_; }
    std::vector<double> ma_coefficients() const { return theta_; }
    double intercept() const { return c_; }
    double sigma() const { return sigma_; }

    // Model order
    int p() const { return p_; }
    int d() const { return d_; }
    int q() const { return q_; }

    // Check if model is fitted
    bool is_fitted() const { return fitted_; }

    // Get AIC and BIC for model selection
    double aic() const;
    double bic() const;

    // Parameter count
    size_t parameter_count() const;

    // Save/Load
    void save(const std::string& filename) const;
    void load(const std::string& filename);

private:
    int p_, d_, q_;                    // Model orders
    std::vector<double> phi_;          // AR coefficients
    std::vector<double> theta_;        // MA coefficients
    double c_;                         // Intercept/constant term
    double sigma_;                     // Standard deviation of residuals

    std::vector<double> data_;         // Original data
    std::vector<double> diff_data_;    // Differenced data
    std::vector<double> residuals_;    // Residuals after fitting
    bool fitted_;

    // Internal methods
    void estimate_parameters();
    void estimate_ar_yule_walker();
    void estimate_ma_innovations();
    void estimate_arma_css();
    std::vector<double> compute_residuals() const;
    double log_likelihood() const;
};

/**
 * SARIMA (Seasonal ARIMA) Model
 *
 * SARIMA(p, d, q)(P, D, Q)_m where:
 * - p, d, q: non-seasonal orders
 * - P, D, Q: seasonal orders
 * - m: seasonal period
 */
class SARIMA {
public:
    SARIMA(int p, int d, int q, int P, int D, int Q, int m);

    void fit(const std::vector<double>& data, int max_iter = 1000, double tol = 1e-6);
    ForecastResult forecast(int steps) const;

    std::vector<double> fitted_values() const;
    std::vector<double> residuals() const;

    bool is_fitted() const { return fitted_; }

private:
    int p_, d_, q_;      // Non-seasonal orders
    int P_, D_, Q_, m_;  // Seasonal orders and period

    std::vector<double> phi_;        // Non-seasonal AR coefficients
    std::vector<double> theta_;      // Non-seasonal MA coefficients
    std::vector<double> Phi_;        // Seasonal AR coefficients
    std::vector<double> Theta_;      // Seasonal MA coefficients
    double c_;
    double sigma_;

    std::vector<double> data_;
    std::vector<double> residuals_;
    bool fitted_;

    void estimate_parameters();
};

/**
 * Auto ARIMA - Automatic order selection
 * Uses AIC/BIC to select best (p, d, q) within given ranges
 */
class AutoARIMA {
public:
    AutoARIMA(int max_p = 5, int max_d = 2, int max_q = 5);

    void fit(const std::vector<double>& data);
    ForecastResult forecast(int steps) const;

    // Get selected orders
    int selected_p() const { return best_p_; }
    int selected_d() const { return best_d_; }
    int selected_q() const { return best_q_; }
    double best_aic() const { return best_aic_; }

    const ARIMA& best_model() const { return *best_model_; }

private:
    int max_p_, max_d_, max_q_;
    int best_p_, best_d_, best_q_;
    double best_aic_;
    std::unique_ptr<ARIMA> best_model_;

    // Augmented Dickey-Fuller test for stationarity
    int determine_d(const std::vector<double>& data) const;
};

} // namespace ts

#endif // TS_ARIMA_HPP
