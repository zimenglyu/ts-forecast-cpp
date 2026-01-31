#ifndef TS_GRADIENT_BOOSTING_HPP
#define TS_GRADIENT_BOOSTING_HPP

#include "utils.hpp"
#include <vector>
#include <memory>
#include <functional>

namespace ts {

/**
 * Decision Tree Node for Gradient Boosting
 */
struct TreeNode {
    bool is_leaf;
    double value;           // Prediction value for leaf nodes
    int feature_index;      // Feature to split on
    double threshold;       // Split threshold
    std::unique_ptr<TreeNode> left;
    std::unique_ptr<TreeNode> right;

    TreeNode() : is_leaf(true), value(0), feature_index(-1), threshold(0) {}
};

/**
 * Decision Tree Regressor
 *
 * A simple CART-style decision tree for regression.
 */
class DecisionTree {
public:
    /**
     * Constructor
     * @param max_depth Maximum depth of the tree
     * @param min_samples_split Minimum samples required to split
     * @param min_samples_leaf Minimum samples in leaf node
     */
    DecisionTree(int max_depth = 6, int min_samples_split = 2,
                 int min_samples_leaf = 1);

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<double>& y,
             const std::vector<double>& sample_weights = {});

    std::vector<double> predict(const std::vector<std::vector<double>>& X) const;
    double predict_single(const std::vector<double>& x) const;

private:
    int max_depth_;
    int min_samples_split_;
    int min_samples_leaf_;
    std::unique_ptr<TreeNode> root_;

    TreeNode* build_tree(const std::vector<std::vector<double>>& X,
                         const std::vector<double>& y,
                         const std::vector<double>& weights,
                         int depth);

    void find_best_split(const std::vector<std::vector<double>>& X,
                         const std::vector<double>& y,
                         const std::vector<double>& weights,
                         int& best_feature, double& best_threshold,
                         double& best_gain) const;

    double compute_variance(const std::vector<double>& y,
                           const std::vector<double>& weights) const;
    double weighted_mean(const std::vector<double>& y,
                        const std::vector<double>& weights) const;
};

/**
 * Gradient Boosting Regressor (XGBoost-like)
 *
 * A simplified implementation of gradient boosting for time series forecasting.
 * Uses decision trees as base learners with gradient descent optimization.
 *
 * Features:
 * - Customizable loss functions (MSE, MAE, Huber)
 * - L1 and L2 regularization
 * - Learning rate (shrinkage)
 * - Feature subsampling
 */
class GradientBoosting {
public:
    enum class LossFunction { MSE, MAE, HUBER };

    /**
     * Constructor
     * @param n_estimators Number of boosting rounds
     * @param max_depth Maximum tree depth
     * @param learning_rate Shrinkage parameter
     * @param subsample Fraction of samples for each tree
     * @param colsample Fraction of features for each tree
     * @param reg_lambda L2 regularization
     * @param reg_alpha L1 regularization
     */
    GradientBoosting(int n_estimators = 100,
                     int max_depth = 6,
                     double learning_rate = 0.1,
                     double subsample = 1.0,
                     double colsample = 1.0,
                     double reg_lambda = 1.0,
                     double reg_alpha = 0.0,
                     LossFunction loss = LossFunction::MSE);

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<double>& y);

    std::vector<double> predict(const std::vector<std::vector<double>>& X) const;

    /**
     * Get feature importance scores
     */
    std::vector<double> feature_importance() const { return feature_importance_; }

private:
    int n_estimators_;
    int max_depth_;
    double learning_rate_;
    double subsample_;
    double colsample_;
    double reg_lambda_;
    double reg_alpha_;
    LossFunction loss_;

    double initial_prediction_;
    std::vector<std::unique_ptr<DecisionTree>> trees_;
    std::vector<double> feature_importance_;

    std::vector<double> compute_gradients(const std::vector<double>& y,
                                          const std::vector<double>& pred) const;
    std::vector<double> compute_hessians(const std::vector<double>& y,
                                         const std::vector<double>& pred) const;
};

/**
 * Time Series Gradient Boosting
 *
 * Wrapper around GradientBoosting specifically for time series forecasting.
 * Automatically creates lag features and handles time series specific operations.
 */
class TimeSeriesGradientBoosting {
public:
    /**
     * Constructor
     * @param n_lags Number of lag features
     * @param n_estimators Number of boosting rounds
     * @param max_depth Maximum tree depth
     * @param learning_rate Shrinkage parameter
     */
    TimeSeriesGradientBoosting(int n_lags = 7,
                                int n_estimators = 100,
                                int max_depth = 6,
                                double learning_rate = 0.1);

    /**
     * Enable time-based features
     * @param use_hour Include hour of day
     * @param use_dayofweek Include day of week
     * @param use_month Include month
     */
    void add_time_features(bool use_hour = false,
                          bool use_dayofweek = true,
                          bool use_month = true);

    /**
     * Add rolling statistics as features
     * @param windows Vector of window sizes
     */
    void add_rolling_features(const std::vector<int>& windows);

    /**
     * Fit the model
     * @param data Time series data
     * @param timestamps Optional timestamps (for time-based features)
     */
    void fit(const std::vector<double>& data,
             const std::vector<double>& timestamps = {});

    /**
     * Forecast future values
     * @param steps Number of steps to forecast
     */
    ForecastResult forecast(int steps) const;

    /**
     * Get fitted values
     */
    std::vector<double> fitted_values() const;

    bool is_fitted() const { return fitted_; }

    /**
     * Get feature importance
     */
    std::vector<std::pair<std::string, double>> feature_importance() const;

private:
    int n_lags_;
    std::unique_ptr<GradientBoosting> model_;

    bool use_hour_, use_dayofweek_, use_month_;
    std::vector<int> rolling_windows_;

    std::vector<double> data_;
    std::vector<double> timestamps_;
    std::vector<std::string> feature_names_;
    bool fitted_;

    std::vector<std::vector<double>> create_features(
        const std::vector<double>& data,
        const std::vector<double>& timestamps) const;

    std::vector<double> create_single_features(
        const std::vector<double>& history,
        double timestamp) const;
};

/**
 * LightGBM-like Gradient Boosting with Histogram-based splits
 *
 * A faster variant using histogram-based split finding.
 * More efficient for large datasets.
 */
class HistogramGradientBoosting {
public:
    HistogramGradientBoosting(int n_estimators = 100,
                               int max_depth = 6,
                               int max_bins = 256,
                               double learning_rate = 0.1);

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<double>& y);

    std::vector<double> predict(const std::vector<std::vector<double>>& X) const;

private:
    int n_estimators_;
    int max_depth_;
    int max_bins_;
    double learning_rate_;

    struct Histogram {
        std::vector<double> gradient_sum;
        std::vector<double> hessian_sum;
        std::vector<int> count;
    };

    std::vector<std::vector<double>> bin_edges_;
    double initial_prediction_;
    std::vector<std::unique_ptr<DecisionTree>> trees_;

    void compute_bin_edges(const std::vector<std::vector<double>>& X);
    std::vector<std::vector<int>> bin_features(
        const std::vector<std::vector<double>>& X) const;
};

} // namespace ts

#endif // TS_GRADIENT_BOOSTING_HPP
