#ifndef TS_DLINEAR_HPP
#define TS_DLINEAR_HPP

#include "utils.hpp"
#include <vector>
#include <random>
#include <cmath>

namespace ts {

/**
 * DLinear Model for Time Series Forecasting
 *
 * Based on "Are Transformers Effective for Time Series Forecasting?" (Zeng et al., 2022)
 *
 * DLinear decomposes the input time series into trend and seasonal components
 * using a moving average, then applies separate linear layers to each component.
 *
 * Architecture:
 *   Input (seq_len) -> Decomposition -> Trend + Seasonal
 *   Trend -> Linear Layer -> Trend Prediction (pred_len)
 *   Seasonal -> Linear Layer -> Seasonal Prediction (pred_len)
 *   Output = Trend Prediction + Seasonal Prediction
 *
 * This simple architecture often outperforms complex transformer models.
 */
class DLinear {
public:
    /**
     * Constructor
     * @param seq_len Input sequence length (lookback window)
     * @param pred_len Prediction length (forecast horizon)
     * @param kernel_size Moving average kernel size for decomposition
     * @param individual If true, use individual linear layers per feature (for multivariate)
     */
    DLinear(int seq_len = 96, int pred_len = 24, int kernel_size = 25, bool individual = false);

    /**
     * Fit the model on univariate time series
     * @param data Time series data
     * @param epochs Number of training epochs
     * @param learning_rate Learning rate for gradient descent
     * @param batch_size Mini-batch size
     */
    void fit(const std::vector<double>& data, int epochs = 100,
             double learning_rate = 0.001, int batch_size = 32);

    /**
     * Fit the model on multivariate time series
     * @param data Multivariate time series (rows = time, cols = features)
     * @param target_idx Index of target variable to predict
     * @param epochs Number of training epochs
     * @param learning_rate Learning rate
     * @param batch_size Mini-batch size
     */
    void fit(const std::vector<std::vector<double>>& data, int target_idx,
             int epochs = 100, double learning_rate = 0.001, int batch_size = 32);

    /**
     * Forecast future values
     * @param steps Number of steps to forecast (should equal pred_len)
     * @return ForecastResult with predictions
     */
    ForecastResult forecast(int steps) const;

    /**
     * Predict from a given input sequence
     * @param input Input sequence of length seq_len
     * @return Predictions of length pred_len
     */
    std::vector<double> predict(const std::vector<double>& input) const;

    /**
     * Predict multivariate
     * @param input Input sequence (seq_len x n_features)
     * @return Predictions of length pred_len
     */
    std::vector<double> predict(const std::vector<std::vector<double>>& input) const;

    bool is_fitted() const { return is_fitted_; }
    int seq_len() const { return seq_len_; }
    int pred_len() const { return pred_len_; }

    // Get training loss history
    std::vector<double> loss_history() const { return loss_history_; }

    /**
     * Get total number of trainable parameters
     * @return Number of parameters (weights + biases)
     */
    size_t parameter_count() const;

    /**
     * Save model to binary file
     */
    void save(const std::string& filename) const;

    /**
     * Load model from binary file
     */
    void load(const std::string& filename);

private:
    int seq_len_;
    int pred_len_;
    int kernel_size_;
    bool individual_;
    int n_features_;
    int target_idx_;

    // Linear layer weights for trend component
    // Shape: (pred_len, seq_len) for univariate
    // Shape: (pred_len, seq_len * n_features) for multivariate
    std::vector<std::vector<double>> W_trend_;
    std::vector<double> b_trend_;

    // Linear layer weights for seasonal component
    std::vector<std::vector<double>> W_seasonal_;
    std::vector<double> b_seasonal_;

    // For normalization
    double mean_;
    double std_;

    // Training history
    std::vector<double> data_;
    std::vector<std::vector<double>> multivariate_data_;
    std::vector<double> loss_history_;
    bool is_fitted_;
    bool is_multivariate_;

    // Decomposition: extract trend using moving average
    std::pair<std::vector<double>, std::vector<double>>
    decompose(const std::vector<double>& x) const;

    // Moving average
    std::vector<double> moving_average(const std::vector<double>& x, int kernel_size) const;

    // Forward pass
    std::vector<double> forward(const std::vector<double>& trend,
                                const std::vector<double>& seasonal) const;

    // Initialize weights
    void initialize_weights();

    // Training step
    double train_step(const std::vector<std::vector<double>>& batch_x,
                      const std::vector<std::vector<double>>& batch_y,
                      double learning_rate);

    // Normalize data
    std::vector<double> normalize(const std::vector<double>& x) const;
    std::vector<double> denormalize(const std::vector<double>& x) const;
};

/**
 * NLinear Model - Even simpler baseline
 *
 * NLinear subtracts the last value of the input sequence before applying
 * a linear layer, then adds it back. This handles distribution shift.
 *
 * Architecture:
 *   Input -> Subtract last value -> Linear -> Add last value back -> Output
 */
class NLinear {
public:
    NLinear(int seq_len = 96, int pred_len = 24);

    void fit(const std::vector<double>& data, int epochs = 100,
             double learning_rate = 0.001, int batch_size = 32);

    ForecastResult forecast(int steps) const;
    std::vector<double> predict(const std::vector<double>& input) const;

    bool is_fitted() const { return is_fitted_; }
    int seq_len() const { return seq_len_; }
    int pred_len() const { return pred_len_; }

    /**
     * Get total number of trainable parameters
     */
    size_t parameter_count() const;

    void save(const std::string& filename) const;
    void load(const std::string& filename);

private:
    int seq_len_;
    int pred_len_;

    std::vector<std::vector<double>> W_;
    std::vector<double> b_;

    double mean_;
    double std_;

    std::vector<double> data_;
    std::vector<double> loss_history_;
    bool is_fitted_;

    void initialize_weights();
    std::vector<double> forward(const std::vector<double>& x, double last_val) const;
};

/**
 * Linear Model - Simplest baseline
 *
 * Direct linear projection from input to output.
 */
class Linear {
public:
    Linear(int seq_len = 96, int pred_len = 24);

    void fit(const std::vector<double>& data, int epochs = 100,
             double learning_rate = 0.001, int batch_size = 32);

    ForecastResult forecast(int steps) const;
    std::vector<double> predict(const std::vector<double>& input) const;

    bool is_fitted() const { return is_fitted_; }
    int seq_len() const { return seq_len_; }
    int pred_len() const { return pred_len_; }

    /**
     * Get total number of trainable parameters
     */
    size_t parameter_count() const;

    void save(const std::string& filename) const;
    void load(const std::string& filename);

private:
    int seq_len_;
    int pred_len_;

    std::vector<std::vector<double>> W_;
    std::vector<double> b_;

    double mean_;
    double std_;

    std::vector<double> data_;
    bool is_fitted_;

    void initialize_weights();
};

} // namespace ts

#endif // TS_DLINEAR_HPP
