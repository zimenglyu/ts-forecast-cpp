#include <iostream>
#include <vector>
#include <cmath>
#include "ts_forecast.hpp"

int main() {
    std::cout << "=== Gradient Boosting for Time Series ===" << std::endl;

    // Generate time series data with patterns
    std::vector<double> data;
    for (int i = 0; i < 200; ++i) {
        double trend = 50.0 + 0.3 * i;
        double weekly = 15.0 * std::sin(2 * M_PI * i / 7.0);
        double noise = 3.0 * (std::rand() % 100 - 50) / 50.0;
        data.push_back(trend + weekly + noise);
    }

    std::cout << "Generated 200 data points with trend and weekly pattern" << std::endl;

    // Basic Time Series Gradient Boosting
    std::cout << "\n--- Time Series Gradient Boosting ---" << std::endl;
    ts::TimeSeriesGradientBoosting gb(7, 50, 4, 0.1);  // 7 lags, 50 trees, depth 4
    gb.add_time_features(false, true, false);  // day of week only
    gb.add_rolling_features({3, 7});  // 3 and 7 day rolling windows

    std::cout << "Fitting model with 7 lags and rolling features..." << std::endl;
    gb.fit(data);

    // Forecast
    int forecast_horizon = 14;
    ts::ForecastResult forecast = gb.forecast(forecast_horizon);

    std::cout << "\n" << forecast_horizon << "-step forecast:" << std::endl;
    std::cout << "Step\tPrediction\tLower\tUpper" << std::endl;
    for (int i = 0; i < forecast_horizon; ++i) {
        std::cout << (i + 1) << "\t"
                  << forecast.predictions[i] << "\t"
                  << forecast.lower_bound[i] << "\t"
                  << forecast.upper_bound[i] << std::endl;
    }

    // Feature importance
    std::cout << "\n--- Feature Importance ---" << std::endl;
    auto importance = gb.feature_importance();
    for (const auto& [name, score] : importance) {
        std::cout << name << ": " << score << std::endl;
    }

    // Compare with fitted values
    std::cout << "\n--- In-sample Fit ---" << std::endl;
    std::vector<double> fitted = gb.fitted_values();
    if (!fitted.empty()) {
        // Skip the first n_lags values
        std::vector<double> actual(data.begin() + 7, data.end());
        std::vector<double> predicted(fitted.begin() + 7, fitted.end());

        if (actual.size() == predicted.size()) {
            ts::Metrics metrics = ts::evaluate(actual, predicted);
            std::cout << "In-sample metrics:" << std::endl;
            std::cout << metrics.to_string() << std::endl;
        }
    }

    // Train/test evaluation
    std::cout << "\n--- Out-of-sample Evaluation ---" << std::endl;
    auto [train, test] = ts::train_test_split(data, 0.2);

    ts::TimeSeriesGradientBoosting gb_eval(7, 100, 5, 0.05);
    gb_eval.add_rolling_features({7});
    gb_eval.fit(train);

    ts::ForecastResult eval_forecast = gb_eval.forecast(static_cast<int>(test.size()));

    // Trim if needed
    std::vector<double> predictions = eval_forecast.predictions;
    if (predictions.size() > test.size()) {
        predictions.resize(test.size());
    }

    ts::Metrics eval_metrics = ts::evaluate(test, predictions);
    std::cout << "Out-of-sample metrics:" << std::endl;
    std::cout << eval_metrics.to_string() << std::endl;

    // Basic Gradient Boosting Example
    std::cout << "\n=== Basic Gradient Boosting Regressor ===" << std::endl;

    // Create feature matrix from lags
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    int n_lags = 5;

    for (size_t t = n_lags; t < data.size(); ++t) {
        std::vector<double> features;
        for (int i = 1; i <= n_lags; ++i) {
            features.push_back(data[t - i]);
        }
        X.push_back(features);
        y.push_back(data[t]);
    }

    // Split data
    size_t split = static_cast<size_t>(X.size() * 0.8);
    std::vector<std::vector<double>> X_train(X.begin(), X.begin() + split);
    std::vector<std::vector<double>> X_test(X.begin() + split, X.end());
    std::vector<double> y_train(y.begin(), y.begin() + split);
    std::vector<double> y_test(y.begin() + split, y.end());

    // Fit model
    ts::GradientBoosting basic_gb(100, 4, 0.1, 0.8, 1.0, 1.0, 0.0,
                                  ts::GradientBoosting::LossFunction::MSE);
    basic_gb.fit(X_train, y_train);

    std::vector<double> basic_pred = basic_gb.predict(X_test);
    ts::Metrics basic_metrics = ts::evaluate(y_test, basic_pred);

    std::cout << "Basic GB out-of-sample metrics:" << std::endl;
    std::cout << basic_metrics.to_string() << std::endl;

    // Compare different loss functions
    std::cout << "\n--- Comparing Loss Functions ---" << std::endl;

    // Add some outliers to test robustness
    std::vector<double> y_train_outliers = y_train;
    y_train_outliers[10] *= 2.0;  // Double one value
    y_train_outliers[50] *= 0.5;  // Halve another

    ts::GradientBoosting gb_mse(50, 4, 0.1, 1.0, 1.0, 1.0, 0.0,
                                ts::GradientBoosting::LossFunction::MSE);
    gb_mse.fit(X_train, y_train_outliers);
    std::vector<double> pred_mse = gb_mse.predict(X_test);

    ts::GradientBoosting gb_huber(50, 4, 0.1, 1.0, 1.0, 1.0, 0.0,
                                  ts::GradientBoosting::LossFunction::HUBER);
    gb_huber.fit(X_train, y_train_outliers);
    std::vector<double> pred_huber = gb_huber.predict(X_test);

    std::cout << "With outliers in training data:" << std::endl;
    std::cout << "MSE Loss - RMSE: " << ts::evaluate(y_test, pred_mse).rmse << std::endl;
    std::cout << "Huber Loss - RMSE: " << ts::evaluate(y_test, pred_huber).rmse << std::endl;

    // Histogram Gradient Boosting
    std::cout << "\n=== Histogram Gradient Boosting ===" << std::endl;
    ts::HistogramGradientBoosting hgb(100, 5, 128, 0.1);
    hgb.fit(X_train, y_train);

    std::vector<double> hgb_pred = hgb.predict(X_test);
    ts::Metrics hgb_metrics = ts::evaluate(y_test, hgb_pred);

    std::cout << "Histogram GB metrics:" << std::endl;
    std::cout << hgb_metrics.to_string() << std::endl;

    return 0;
}
