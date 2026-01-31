#include "gradient_boosting.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

namespace ts {

// Decision Tree Implementation
DecisionTree::DecisionTree(int max_depth, int min_samples_split, int min_samples_leaf)
    : max_depth_(max_depth), min_samples_split_(min_samples_split),
      min_samples_leaf_(min_samples_leaf) {}

void DecisionTree::fit(const std::vector<std::vector<double>>& X,
                       const std::vector<double>& y,
                       const std::vector<double>& sample_weights) {
    std::vector<double> weights = sample_weights;
    if (weights.empty()) {
        weights.resize(y.size(), 1.0);
    }

    root_.reset(build_tree(X, y, weights, 0));
}

TreeNode* DecisionTree::build_tree(const std::vector<std::vector<double>>& X,
                                   const std::vector<double>& y,
                                   const std::vector<double>& weights,
                                   int depth) {
    TreeNode* node = new TreeNode();

    // Check stopping conditions
    if (depth >= max_depth_ ||
        static_cast<int>(y.size()) < min_samples_split_ ||
        static_cast<int>(y.size()) <= min_samples_leaf_) {
        node->is_leaf = true;
        node->value = weighted_mean(y, weights);
        return node;
    }

    // Check if all values are the same
    bool all_same = true;
    for (size_t i = 1; i < y.size(); ++i) {
        if (std::abs(y[i] - y[0]) > 1e-10) {
            all_same = false;
            break;
        }
    }
    if (all_same) {
        node->is_leaf = true;
        node->value = y[0];
        return node;
    }

    // Find best split
    int best_feature = -1;
    double best_threshold = 0.0;
    double best_gain = -std::numeric_limits<double>::max();

    find_best_split(X, y, weights, best_feature, best_threshold, best_gain);

    if (best_feature < 0 || best_gain <= 0) {
        node->is_leaf = true;
        node->value = weighted_mean(y, weights);
        return node;
    }

    // Split data
    std::vector<std::vector<double>> X_left, X_right;
    std::vector<double> y_left, y_right;
    std::vector<double> w_left, w_right;

    for (size_t i = 0; i < X.size(); ++i) {
        if (X[i][best_feature] <= best_threshold) {
            X_left.push_back(X[i]);
            y_left.push_back(y[i]);
            w_left.push_back(weights[i]);
        } else {
            X_right.push_back(X[i]);
            y_right.push_back(y[i]);
            w_right.push_back(weights[i]);
        }
    }

    // Check minimum leaf size
    if (static_cast<int>(y_left.size()) < min_samples_leaf_ ||
        static_cast<int>(y_right.size()) < min_samples_leaf_) {
        node->is_leaf = true;
        node->value = weighted_mean(y, weights);
        return node;
    }

    node->is_leaf = false;
    node->feature_index = best_feature;
    node->threshold = best_threshold;
    node->left.reset(build_tree(X_left, y_left, w_left, depth + 1));
    node->right.reset(build_tree(X_right, y_right, w_right, depth + 1));

    return node;
}

void DecisionTree::find_best_split(const std::vector<std::vector<double>>& X,
                                   const std::vector<double>& y,
                                   const std::vector<double>& weights,
                                   int& best_feature, double& best_threshold,
                                   double& best_gain) const {
    int n_features = static_cast<int>(X[0].size());
    double parent_var = compute_variance(y, weights);

    best_feature = -1;
    best_threshold = 0.0;
    best_gain = -std::numeric_limits<double>::max();

    for (int f = 0; f < n_features; ++f) {
        // Get unique values for this feature
        std::vector<double> values;
        for (const auto& x : X) {
            values.push_back(x[f]);
        }
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());

        // Try midpoints as thresholds
        for (size_t i = 0; i < values.size() - 1; ++i) {
            double threshold = (values[i] + values[i + 1]) / 2.0;

            std::vector<double> y_left, y_right;
            std::vector<double> w_left, w_right;

            for (size_t j = 0; j < X.size(); ++j) {
                if (X[j][f] <= threshold) {
                    y_left.push_back(y[j]);
                    w_left.push_back(weights[j]);
                } else {
                    y_right.push_back(y[j]);
                    w_right.push_back(weights[j]);
                }
            }

            if (y_left.empty() || y_right.empty()) continue;

            double var_left = compute_variance(y_left, w_left);
            double var_right = compute_variance(y_right, w_right);

            double w_sum_left = std::accumulate(w_left.begin(), w_left.end(), 0.0);
            double w_sum_right = std::accumulate(w_right.begin(), w_right.end(), 0.0);
            double w_sum = w_sum_left + w_sum_right;

            double weighted_var = (w_sum_left * var_left + w_sum_right * var_right) / w_sum;
            double gain = parent_var - weighted_var;

            if (gain > best_gain) {
                best_gain = gain;
                best_feature = f;
                best_threshold = threshold;
            }
        }
    }
}

double DecisionTree::compute_variance(const std::vector<double>& y,
                                      const std::vector<double>& weights) const {
    if (y.empty()) return 0.0;

    double w_sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    double mean = 0.0;
    for (size_t i = 0; i < y.size(); ++i) {
        mean += weights[i] * y[i];
    }
    mean /= w_sum;

    double var = 0.0;
    for (size_t i = 0; i < y.size(); ++i) {
        var += weights[i] * (y[i] - mean) * (y[i] - mean);
    }
    return var / w_sum;
}

double DecisionTree::weighted_mean(const std::vector<double>& y,
                                   const std::vector<double>& weights) const {
    if (y.empty()) return 0.0;

    double w_sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    double sum = 0.0;
    for (size_t i = 0; i < y.size(); ++i) {
        sum += weights[i] * y[i];
    }
    return sum / w_sum;
}

std::vector<double> DecisionTree::predict(const std::vector<std::vector<double>>& X) const {
    std::vector<double> predictions(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        predictions[i] = predict_single(X[i]);
    }
    return predictions;
}

double DecisionTree::predict_single(const std::vector<double>& x) const {
    if (!root_) return 0.0;

    TreeNode* node = root_.get();
    while (!node->is_leaf) {
        if (x[node->feature_index] <= node->threshold) {
            node = node->left.get();
        } else {
            node = node->right.get();
        }
    }
    return node->value;
}

// Gradient Boosting Implementation
GradientBoosting::GradientBoosting(int n_estimators, int max_depth,
                                   double learning_rate, double subsample,
                                   double colsample, double reg_lambda,
                                   double reg_alpha, LossFunction loss)
    : n_estimators_(n_estimators), max_depth_(max_depth),
      learning_rate_(learning_rate), subsample_(subsample),
      colsample_(colsample), reg_lambda_(reg_lambda),
      reg_alpha_(reg_alpha), loss_(loss), initial_prediction_(0) {}

void GradientBoosting::fit(const std::vector<std::vector<double>>& X,
                           const std::vector<double>& y) {
    int n = static_cast<int>(y.size());
    int n_features = static_cast<int>(X[0].size());

    // Initialize prediction with mean
    initial_prediction_ = stats::mean(y);
    std::vector<double> predictions(n, initial_prediction_);

    feature_importance_.resize(n_features, 0.0);
    trees_.clear();

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int iter = 0; iter < n_estimators_; ++iter) {
        // Compute gradients
        std::vector<double> gradients = compute_gradients(y, predictions);
        std::vector<double> hessians = compute_hessians(y, predictions);

        // Subsample
        std::vector<int> sample_indices;
        if (subsample_ < 1.0) {
            std::bernoulli_distribution dist(subsample_);
            for (int i = 0; i < n; ++i) {
                if (dist(gen)) {
                    sample_indices.push_back(i);
                }
            }
        } else {
            for (int i = 0; i < n; ++i) {
                sample_indices.push_back(i);
            }
        }

        if (sample_indices.empty()) continue;

        // Column subsample
        std::vector<int> feature_indices;
        if (colsample_ < 1.0) {
            std::bernoulli_distribution dist(colsample_);
            for (int f = 0; f < n_features; ++f) {
                if (dist(gen)) {
                    feature_indices.push_back(f);
                }
            }
        } else {
            for (int f = 0; f < n_features; ++f) {
                feature_indices.push_back(f);
            }
        }

        if (feature_indices.empty()) continue;

        // Create subsampled dataset
        std::vector<std::vector<double>> X_sub;
        std::vector<double> grad_sub;
        std::vector<double> hess_sub;

        for (int idx : sample_indices) {
            std::vector<double> x_row;
            for (int f : feature_indices) {
                x_row.push_back(X[idx][f]);
            }
            X_sub.push_back(x_row);
            grad_sub.push_back(-gradients[idx]);  // Negative gradient for descent
            hess_sub.push_back(hessians[idx]);
        }

        // Fit tree to negative gradients (residuals)
        auto tree = std::make_unique<DecisionTree>(max_depth_, 2, 1);
        tree->fit(X_sub, grad_sub, hess_sub);

        // Update predictions
        for (int i = 0; i < n; ++i) {
            std::vector<double> x_row;
            for (int f : feature_indices) {
                x_row.push_back(X[i][f]);
            }
            predictions[i] += learning_rate_ * tree->predict_single(x_row);
        }

        trees_.push_back(std::move(tree));
    }
}

std::vector<double> GradientBoosting::compute_gradients(const std::vector<double>& y,
                                                        const std::vector<double>& pred) const {
    std::vector<double> gradients(y.size());

    for (size_t i = 0; i < y.size(); ++i) {
        switch (loss_) {
            case LossFunction::MSE:
                gradients[i] = pred[i] - y[i];
                break;
            case LossFunction::MAE:
                gradients[i] = (pred[i] > y[i]) ? 1.0 : -1.0;
                break;
            case LossFunction::HUBER: {
                double delta = 1.0;
                double diff = pred[i] - y[i];
                if (std::abs(diff) <= delta) {
                    gradients[i] = diff;
                } else {
                    gradients[i] = delta * ((diff > 0) ? 1.0 : -1.0);
                }
                break;
            }
        }
    }

    return gradients;
}

std::vector<double> GradientBoosting::compute_hessians(const std::vector<double>& y,
                                                       const std::vector<double>& pred) const {
    std::vector<double> hessians(y.size());

    for (size_t i = 0; i < y.size(); ++i) {
        switch (loss_) {
            case LossFunction::MSE:
                hessians[i] = 1.0;
                break;
            case LossFunction::MAE:
                hessians[i] = 1.0;  // Constant for numerical stability
                break;
            case LossFunction::HUBER: {
                double delta = 1.0;
                double diff = std::abs(pred[i] - y[i]);
                hessians[i] = (diff <= delta) ? 1.0 : 0.0;
                break;
            }
        }
    }

    return hessians;
}

std::vector<double> GradientBoosting::predict(const std::vector<std::vector<double>>& X) const {
    std::vector<double> predictions(X.size(), initial_prediction_);

    for (const auto& tree : trees_) {
        std::vector<double> tree_preds = tree->predict(X);
        for (size_t i = 0; i < X.size(); ++i) {
            predictions[i] += learning_rate_ * tree_preds[i];
        }
    }

    return predictions;
}

// Time Series Gradient Boosting Implementation
TimeSeriesGradientBoosting::TimeSeriesGradientBoosting(int n_lags, int n_estimators,
                                                         int max_depth, double learning_rate)
    : n_lags_(n_lags), use_hour_(false), use_dayofweek_(true), use_month_(true),
      fitted_(false) {
    model_ = std::make_unique<GradientBoosting>(n_estimators, max_depth, learning_rate);
}

void TimeSeriesGradientBoosting::add_time_features(bool use_hour, bool use_dayofweek,
                                                   bool use_month) {
    use_hour_ = use_hour;
    use_dayofweek_ = use_dayofweek;
    use_month_ = use_month;
}

void TimeSeriesGradientBoosting::add_rolling_features(const std::vector<int>& windows) {
    rolling_windows_ = windows;
}

void TimeSeriesGradientBoosting::fit(const std::vector<double>& data,
                                     const std::vector<double>& timestamps) {
    if (data.size() < static_cast<size_t>(n_lags_ + 1)) {
        throw std::invalid_argument("Insufficient data for the specified lag");
    }

    data_ = data;
    timestamps_ = timestamps;

    if (timestamps_.empty()) {
        timestamps_.resize(data_.size());
        for (size_t i = 0; i < data_.size(); ++i) {
            timestamps_[i] = static_cast<double>(i);
        }
    }

    // Create features
    auto [X, y] = [this]() {
        std::vector<std::vector<double>> X;
        std::vector<double> y;

        for (size_t t = n_lags_; t < data_.size(); ++t) {
            std::vector<double> features;

            // Lag features
            for (int i = 1; i <= n_lags_; ++i) {
                features.push_back(data_[t - i]);
            }

            // Rolling statistics
            for (int window : rolling_windows_) {
                if (static_cast<int>(t) >= window) {
                    double sum = 0.0, sum_sq = 0.0;
                    double min_val = data_[t - 1], max_val = data_[t - 1];

                    for (int i = 1; i <= window; ++i) {
                        double val = data_[t - i];
                        sum += val;
                        sum_sq += val * val;
                        min_val = std::min(min_val, val);
                        max_val = std::max(max_val, val);
                    }

                    double mean = sum / window;
                    double var = (sum_sq / window) - (mean * mean);

                    features.push_back(mean);
                    features.push_back(std::sqrt(std::max(0.0, var)));
                    features.push_back(min_val);
                    features.push_back(max_val);
                }
            }

            // Time-based features (simplified)
            if (use_dayofweek_) {
                int day = static_cast<int>(timestamps_[t]) % 7;
                features.push_back(static_cast<double>(day));
            }

            if (use_month_) {
                int month = (static_cast<int>(timestamps_[t]) / 30) % 12;
                features.push_back(static_cast<double>(month));
            }

            X.push_back(features);
            y.push_back(data_[t]);
        }

        return std::make_pair(X, y);
    }();

    // Build feature names
    feature_names_.clear();
    for (int i = 1; i <= n_lags_; ++i) {
        feature_names_.push_back("lag_" + std::to_string(i));
    }
    for (int window : rolling_windows_) {
        feature_names_.push_back("rolling_mean_" + std::to_string(window));
        feature_names_.push_back("rolling_std_" + std::to_string(window));
        feature_names_.push_back("rolling_min_" + std::to_string(window));
        feature_names_.push_back("rolling_max_" + std::to_string(window));
    }
    if (use_dayofweek_) feature_names_.push_back("dayofweek");
    if (use_month_) feature_names_.push_back("month");

    model_->fit(X, y);
    fitted_ = true;
}

std::vector<double> TimeSeriesGradientBoosting::create_single_features(
    const std::vector<double>& history, double timestamp) const {

    std::vector<double> features;

    // Lag features
    for (int i = 1; i <= n_lags_; ++i) {
        int idx = static_cast<int>(history.size()) - i;
        if (idx >= 0) {
            features.push_back(history[idx]);
        } else {
            features.push_back(0.0);
        }
    }

    // Rolling statistics
    for (int window : rolling_windows_) {
        int n = static_cast<int>(history.size());
        if (n >= window) {
            double sum = 0.0, sum_sq = 0.0;
            double min_val = history[n - 1], max_val = history[n - 1];

            for (int i = 1; i <= window && n - i >= 0; ++i) {
                double val = history[n - i];
                sum += val;
                sum_sq += val * val;
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }

            double mean = sum / window;
            double var = (sum_sq / window) - (mean * mean);

            features.push_back(mean);
            features.push_back(std::sqrt(std::max(0.0, var)));
            features.push_back(min_val);
            features.push_back(max_val);
        }
    }

    // Time features
    if (use_dayofweek_) {
        int day = static_cast<int>(timestamp) % 7;
        features.push_back(static_cast<double>(day));
    }

    if (use_month_) {
        int month = (static_cast<int>(timestamp) / 30) % 12;
        features.push_back(static_cast<double>(month));
    }

    return features;
}

ForecastResult TimeSeriesGradientBoosting::forecast(int steps) const {
    if (!fitted_) {
        throw std::runtime_error("Model must be fitted before forecasting");
    }

    ForecastResult result;
    result.predictions.resize(steps);
    result.lower_bound.resize(steps);
    result.upper_bound.resize(steps);

    std::vector<double> history = data_;
    double last_timestamp = timestamps_.empty() ? data_.size() : timestamps_.back();

    for (int h = 0; h < steps; ++h) {
        double timestamp = last_timestamp + h + 1;
        std::vector<double> features = create_single_features(history, timestamp);

        std::vector<std::vector<double>> X_single = {features};
        std::vector<double> pred = model_->predict(X_single);

        result.predictions[h] = pred[0];
        history.push_back(pred[0]);
    }

    // Confidence intervals
    double sigma = stats::std_dev(data_) * 0.1;
    double z = 1.96;
    for (int h = 0; h < steps; ++h) {
        double se = sigma * std::sqrt(h + 1);
        result.lower_bound[h] = result.predictions[h] - z * se;
        result.upper_bound[h] = result.predictions[h] + z * se;
    }

    return result;
}

std::vector<double> TimeSeriesGradientBoosting::fitted_values() const {
    if (!fitted_) return {};

    std::vector<std::vector<double>> X;
    for (size_t t = n_lags_; t < data_.size(); ++t) {
        double timestamp = timestamps_.empty() ? static_cast<double>(t) : timestamps_[t];
        std::vector<double> history(data_.begin(), data_.begin() + t);
        X.push_back(create_single_features(history, timestamp));
    }

    std::vector<double> result(n_lags_, data_[0]);  // Pad beginning
    std::vector<double> predictions = model_->predict(X);
    result.insert(result.end(), predictions.begin(), predictions.end());

    return result;
}

std::vector<std::pair<std::string, double>> TimeSeriesGradientBoosting::feature_importance() const {
    std::vector<std::pair<std::string, double>> importance;
    auto imp = model_->feature_importance();

    for (size_t i = 0; i < feature_names_.size() && i < imp.size(); ++i) {
        importance.push_back({feature_names_[i], imp[i]});
    }

    std::sort(importance.begin(), importance.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    return importance;
}

// Histogram Gradient Boosting Implementation
HistogramGradientBoosting::HistogramGradientBoosting(int n_estimators, int max_depth,
                                                       int max_bins, double learning_rate)
    : n_estimators_(n_estimators), max_depth_(max_depth),
      max_bins_(max_bins), learning_rate_(learning_rate), initial_prediction_(0) {}

void HistogramGradientBoosting::compute_bin_edges(const std::vector<std::vector<double>>& X) {
    int n_features = static_cast<int>(X[0].size());
    bin_edges_.resize(n_features);

    for (int f = 0; f < n_features; ++f) {
        std::vector<double> values;
        for (const auto& x : X) {
            values.push_back(x[f]);
        }
        std::sort(values.begin(), values.end());

        int n_bins = std::min(max_bins_, static_cast<int>(values.size()));
        bin_edges_[f].resize(n_bins + 1);

        bin_edges_[f][0] = values.front() - 1e-10;
        for (int i = 1; i < n_bins; ++i) {
            int idx = (i * static_cast<int>(values.size())) / n_bins;
            bin_edges_[f][i] = values[idx];
        }
        bin_edges_[f][n_bins] = values.back() + 1e-10;
    }
}

std::vector<std::vector<int>> HistogramGradientBoosting::bin_features(
    const std::vector<std::vector<double>>& X) const {

    std::vector<std::vector<int>> binned(X.size());

    for (size_t i = 0; i < X.size(); ++i) {
        binned[i].resize(X[i].size());
        for (size_t f = 0; f < X[i].size(); ++f) {
            // Find bin using binary search
            auto it = std::upper_bound(bin_edges_[f].begin(), bin_edges_[f].end(), X[i][f]);
            binned[i][f] = static_cast<int>(std::distance(bin_edges_[f].begin(), it)) - 1;
            binned[i][f] = std::max(0, std::min(binned[i][f],
                                                 static_cast<int>(bin_edges_[f].size()) - 2));
        }
    }

    return binned;
}

void HistogramGradientBoosting::fit(const std::vector<std::vector<double>>& X,
                                    const std::vector<double>& y) {
    // Compute bin edges
    compute_bin_edges(X);

    // Initialize prediction
    initial_prediction_ = stats::mean(y);
    std::vector<double> predictions(y.size(), initial_prediction_);

    trees_.clear();

    for (int iter = 0; iter < n_estimators_; ++iter) {
        // Compute residuals
        std::vector<double> residuals(y.size());
        for (size_t i = 0; i < y.size(); ++i) {
            residuals[i] = y[i] - predictions[i];
        }

        // Fit tree to residuals
        auto tree = std::make_unique<DecisionTree>(max_depth_, 2, 1);
        tree->fit(X, residuals);

        // Update predictions
        std::vector<double> tree_preds = tree->predict(X);
        for (size_t i = 0; i < y.size(); ++i) {
            predictions[i] += learning_rate_ * tree_preds[i];
        }

        trees_.push_back(std::move(tree));
    }
}

std::vector<double> HistogramGradientBoosting::predict(
    const std::vector<std::vector<double>>& X) const {

    std::vector<double> predictions(X.size(), initial_prediction_);

    for (const auto& tree : trees_) {
        std::vector<double> tree_preds = tree->predict(X);
        for (size_t i = 0; i < X.size(); ++i) {
            predictions[i] += learning_rate_ * tree_preds[i];
        }
    }

    return predictions;
}

} // namespace ts
