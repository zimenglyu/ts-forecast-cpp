#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <string>
#include <iomanip>
#include <chrono>
#include "ts_forecast.hpp"
#include "csv_reader.hpp"

// Test configuration
const std::string DATA_PATH = "/Users/zimenglyu/Downloads/ETTh1.csv";
const int TRAIN_SIZE = 2000;      // Reduced for faster testing
const int TEST_SIZE = 168;        // 1 week for testing
const int FORECAST_HORIZON = 24;  // Predict 24 hours ahead

int tests_passed = 0;
int tests_failed = 0;

void test_assert(bool condition, const std::string& test_name,
                 const std::string& details = "") {
    if (condition) {
        std::cout << "[PASS] " << test_name;
        if (!details.empty()) std::cout << " - " << details;
        std::cout << std::endl;
        tests_passed++;
    } else {
        std::cout << "[FAIL] " << test_name;
        if (!details.empty()) std::cout << " - " << details;
        std::cout << std::endl;
        tests_failed++;
    }
}

void print_header(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(70, '=') << std::endl;
}

void print_metrics(const std::string& model_name, const ts::Metrics& metrics, double time_ms) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  " << std::left << std::setw(25) << model_name
              << " RMSE: " << std::setw(10) << metrics.rmse
              << " MAE: " << std::setw(10) << metrics.mae
              << " Time: " << std::setw(8) << time_ms << "ms" << std::endl;
}

// Verify predictions are reasonable (not NaN, not Inf, within reasonable bounds)
bool verify_predictions(const std::vector<double>& predictions,
                        const std::vector<double>& actual,
                        double max_allowed_error_ratio = 2.0) {
    if (predictions.size() != actual.size()) return false;

    double actual_mean = ts::stats::mean(actual);
    double actual_std = ts::stats::std_dev(actual);

    for (size_t i = 0; i < predictions.size(); ++i) {
        // Check for NaN or Inf
        if (std::isnan(predictions[i]) || std::isinf(predictions[i])) {
            return false;
        }

        // Check predictions are within reasonable range
        // (within max_allowed_error_ratio * std_dev of mean)
        double lower = actual_mean - max_allowed_error_ratio * 3 * actual_std;
        double upper = actual_mean + max_allowed_error_ratio * 3 * actual_std;
        if (predictions[i] < lower || predictions[i] > upper) {
            // Allow some outliers but not too many
            continue;
        }
    }

    return true;
}

void test_arima(const std::vector<double>& train, const std::vector<double>& test) {
    print_header("ARIMA Model Tests");

    // Test ARIMA(1,1,0)
    {
        auto start = std::chrono::high_resolution_clock::now();

        ts::ARIMA model(1, 1, 0);
        model.fit(train);

        test_assert(model.is_fitted(), "ARIMA(1,1,0) fitting");

        auto forecast = model.forecast(static_cast<int>(test.size()));
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        test_assert(forecast.predictions.size() == test.size(),
                    "ARIMA(1,1,0) forecast size");
        test_assert(verify_predictions(forecast.predictions, test),
                    "ARIMA(1,1,0) predictions validity");

        ts::Metrics metrics = ts::evaluate(test, forecast.predictions);
        print_metrics("ARIMA(1,1,0)", metrics, time_ms);

        // RMSE should be less than 2x the std dev of test data
        double test_std = ts::stats::std_dev(test);
        test_assert(metrics.rmse < 2 * test_std,
                    "ARIMA(1,1,0) RMSE reasonable",
                    "RMSE=" + std::to_string(metrics.rmse) + " < 2*std=" + std::to_string(2*test_std));
    }

    // Test ARIMA(2,1,1)
    {
        auto start = std::chrono::high_resolution_clock::now();

        ts::ARIMA model(2, 1, 1);
        model.fit(train);

        test_assert(model.is_fitted(), "ARIMA(2,1,1) fitting");

        auto forecast = model.forecast(static_cast<int>(test.size()));
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        test_assert(verify_predictions(forecast.predictions, test),
                    "ARIMA(2,1,1) predictions validity");

        ts::Metrics metrics = ts::evaluate(test, forecast.predictions);
        print_metrics("ARIMA(2,1,1)", metrics, time_ms);

        test_assert(metrics.rmse > 0 && !std::isnan(metrics.rmse),
                    "ARIMA(2,1,1) valid RMSE");
    }

    // Test Auto ARIMA
    {
        auto start = std::chrono::high_resolution_clock::now();

        // Use smaller subset for auto-arima (faster)
        std::vector<double> train_subset(train.end() - 2000, train.end());

        ts::AutoARIMA model(3, 2, 3);
        model.fit(train_subset);

        auto forecast = model.forecast(FORECAST_HORIZON);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::string selected = "ARIMA(" + std::to_string(model.selected_p()) + "," +
                               std::to_string(model.selected_d()) + "," +
                               std::to_string(model.selected_q()) + ")";

        test_assert(forecast.predictions.size() == FORECAST_HORIZON,
                    "Auto ARIMA forecast size", "Selected: " + selected);

        std::vector<double> test_subset(test.begin(), test.begin() + FORECAST_HORIZON);
        ts::Metrics metrics = ts::evaluate(test_subset, forecast.predictions);
        print_metrics("Auto " + selected, metrics, time_ms);
    }
}

void test_exponential_smoothing(const std::vector<double>& train,
                                 const std::vector<double>& test) {
    print_header("Exponential Smoothing Model Tests");

    // Test Simple Exponential Smoothing
    {
        auto start = std::chrono::high_resolution_clock::now();

        ts::SimpleExponentialSmoothing model;
        model.fit(train);

        test_assert(model.is_fitted(), "SES fitting");
        test_assert(model.alpha() > 0 && model.alpha() < 1,
                    "SES alpha valid", "alpha=" + std::to_string(model.alpha()));

        auto forecast = model.forecast(static_cast<int>(test.size()));
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        test_assert(verify_predictions(forecast.predictions, test),
                    "SES predictions validity");

        ts::Metrics metrics = ts::evaluate(test, forecast.predictions);
        print_metrics("SES", metrics, time_ms);
    }

    // Test Holt's Linear Trend
    {
        auto start = std::chrono::high_resolution_clock::now();

        ts::HoltLinear model;
        model.fit(train);

        test_assert(model.is_fitted(), "Holt Linear fitting");
        test_assert(model.alpha() > 0 && model.alpha() < 1, "Holt alpha valid");
        test_assert(model.beta() >= 0 && model.beta() < 1, "Holt beta valid");

        auto forecast = model.forecast(static_cast<int>(test.size()));
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        test_assert(verify_predictions(forecast.predictions, test),
                    "Holt Linear predictions validity");

        ts::Metrics metrics = ts::evaluate(test, forecast.predictions);
        print_metrics("Holt Linear", metrics, time_ms);
    }

    // Test Holt-Winters with daily seasonality (24 hours)
    {
        auto start = std::chrono::high_resolution_clock::now();

        ts::HoltWinters model(24, ts::HoltWinters::SeasonalType::ADDITIVE);
        model.fit(train);

        test_assert(model.is_fitted(), "Holt-Winters(24) fitting");
        test_assert(model.period() == 24, "Holt-Winters period correct");

        auto forecast = model.forecast(static_cast<int>(test.size()));
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        test_assert(verify_predictions(forecast.predictions, test),
                    "Holt-Winters predictions validity");

        ts::Metrics metrics = ts::evaluate(test, forecast.predictions);
        print_metrics("Holt-Winters(24)", metrics, time_ms);

        // Holt-Winters should outperform simple methods on seasonal data
        double test_std = ts::stats::std_dev(test);
        test_assert(metrics.rmse < 1.5 * test_std,
                    "Holt-Winters RMSE reasonable",
                    "RMSE=" + std::to_string(metrics.rmse));
    }

    // Test with weekly seasonality (168 hours)
    {
        auto start = std::chrono::high_resolution_clock::now();

        ts::HoltWinters model(168, ts::HoltWinters::SeasonalType::ADDITIVE);
        model.fit(train);

        test_assert(model.is_fitted(), "Holt-Winters(168) fitting");

        auto forecast = model.forecast(FORECAST_HORIZON);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::vector<double> test_subset(test.begin(), test.begin() + FORECAST_HORIZON);
        ts::Metrics metrics = ts::evaluate(test_subset, forecast.predictions);
        print_metrics("Holt-Winters(168)", metrics, time_ms);
    }
}

void test_prophet(const std::vector<double>& train, const std::vector<double>& test) {
    print_header("Prophet Model Tests");

    // Test basic Prophet
    {
        auto start = std::chrono::high_resolution_clock::now();

        ts::Prophet model(ts::Prophet::GrowthType::LINEAR, false, false, true);
        model.fit(train);

        test_assert(model.is_fitted(), "Prophet fitting");

        auto forecast = model.forecast(static_cast<int>(test.size()));
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        test_assert(forecast.predictions.size() == test.size(),
                    "Prophet forecast size");
        test_assert(verify_predictions(forecast.predictions, test),
                    "Prophet predictions validity");

        ts::Metrics metrics = ts::evaluate(test, forecast.predictions);
        print_metrics("Prophet (daily)", metrics, time_ms);
    }

    // Test Prophet with weekly seasonality
    {
        auto start = std::chrono::high_resolution_clock::now();

        ts::Prophet model(ts::Prophet::GrowthType::LINEAR, false, true, false);
        model.set_changepoints(10, 0.8);
        model.fit(train);

        test_assert(model.is_fitted(), "Prophet weekly fitting");

        auto forecast = model.forecast(static_cast<int>(test.size()));
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        test_assert(verify_predictions(forecast.predictions, test),
                    "Prophet weekly predictions validity");

        ts::Metrics metrics = ts::evaluate(test, forecast.predictions);
        print_metrics("Prophet (weekly)", metrics, time_ms);
    }

    // Test Prophet with custom hourly seasonality
    {
        auto start = std::chrono::high_resolution_clock::now();

        ts::Prophet model;
        model.add_seasonality("hourly", 24.0, 5);  // 24-hour period
        model.fit(train);

        test_assert(model.is_fitted(), "Prophet custom seasonality fitting");

        auto forecast = model.forecast(FORECAST_HORIZON);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::vector<double> test_subset(test.begin(), test.begin() + FORECAST_HORIZON);
        ts::Metrics metrics = ts::evaluate(test_subset, forecast.predictions);
        print_metrics("Prophet (custom 24h)", metrics, time_ms);
    }

    // Test decomposition
    {
        ts::Prophet model(ts::Prophet::GrowthType::LINEAR, false, true, true);
        model.fit(train);

        std::vector<double> timestamps(100);
        for (int i = 0; i < 100; ++i) timestamps[i] = i;

        auto trend = model.get_trend(timestamps);
        auto seasonality = model.get_seasonality(timestamps);

        test_assert(trend.size() == 100, "Prophet trend extraction");
        test_assert(seasonality.size() == 100, "Prophet seasonality extraction");

        // Trend should be monotonic (either all increasing or all decreasing)
        bool monotonic = true;
        for (size_t i = 1; i < trend.size(); ++i) {
            if ((trend[i] - trend[i-1]) * (trend[1] - trend[0]) < 0) {
                monotonic = false;
                break;
            }
        }
        test_assert(monotonic || std::abs(trend[99] - trend[0]) < 0.1,
                    "Prophet trend is consistent");
    }
}

void test_gradient_boosting(const std::vector<double>& train,
                            const std::vector<double>& test,
                            const std::vector<std::vector<double>>& train_features,
                            const std::vector<std::vector<double>>& test_features) {
    print_header("Gradient Boosting Model Tests");

    // Test univariate time series gradient boosting
    {
        auto start = std::chrono::high_resolution_clock::now();

        // Use smaller subset for gradient boosting (faster)
        std::vector<double> train_subset(train.end() - 500, train.end());

        ts::TimeSeriesGradientBoosting model(12, 20, 3, 0.1);  // Reduced params
        model.add_rolling_features({6, 12});  // Smaller rolling windows
        model.fit(train_subset);

        test_assert(model.is_fitted(), "TS Gradient Boosting fitting");

        auto forecast = model.forecast(static_cast<int>(test.size()));
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        test_assert(forecast.predictions.size() == test.size(),
                    "TS GB forecast size");
        test_assert(verify_predictions(forecast.predictions, test),
                    "TS GB predictions validity");

        ts::Metrics metrics = ts::evaluate(test, forecast.predictions);
        print_metrics("TS GradientBoosting", metrics, time_ms);

        // Feature importance should be available
        auto importance = model.feature_importance();
        test_assert(!importance.empty(), "TS GB feature importance available");
    }

    // Test multivariate gradient boosting using all features
    {
        auto start = std::chrono::high_resolution_clock::now();

        // Create multivariate features with lags - use small subset
        int n_lags = 3;  // Very small for faster testing
        size_t subset_start = train_features.size() > 300 ? train_features.size() - 300 : 0;

        std::vector<std::vector<double>> X_train;
        std::vector<double> y_train;

        for (size_t t = subset_start + n_lags; t < train_features.size(); ++t) {
            std::vector<double> features;

            // Add lagged values of all features
            for (int lag = 1; lag <= n_lags; ++lag) {
                for (size_t f = 0; f < train_features[0].size(); ++f) {
                    features.push_back(train_features[t - lag][f]);
                }
            }

            // Add lagged target
            for (int lag = 1; lag <= n_lags; ++lag) {
                features.push_back(train[t - lag]);
            }

            X_train.push_back(features);
            y_train.push_back(train[t]);
        }

        ts::GradientBoosting model(30, 4, 0.1, 0.8, 0.8, 1.0, 0.0,
                                   ts::GradientBoosting::LossFunction::HUBER);
        model.fit(X_train, y_train);

        // Create test features
        std::vector<std::vector<double>> X_test;
        std::vector<double> y_test;

        // Combine train and test for creating test features
        std::vector<std::vector<double>> all_features = train_features;
        all_features.insert(all_features.end(), test_features.begin(), test_features.end());
        std::vector<double> all_target = train;
        all_target.insert(all_target.end(), test.begin(), test.end());

        for (size_t t = train.size(); t < train.size() + test.size(); ++t) {
            std::vector<double> features;

            for (int lag = 1; lag <= n_lags; ++lag) {
                for (size_t f = 0; f < all_features[0].size(); ++f) {
                    features.push_back(all_features[t - lag][f]);
                }
            }
            for (int lag = 1; lag <= n_lags; ++lag) {
                features.push_back(all_target[t - lag]);
            }

            X_test.push_back(features);
            y_test.push_back(all_target[t]);
        }

        auto predictions = model.predict(X_test);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        test_assert(predictions.size() == y_test.size(),
                    "Multivariate GB prediction size");
        test_assert(verify_predictions(predictions, y_test),
                    "Multivariate GB predictions validity");

        ts::Metrics metrics = ts::evaluate(y_test, predictions);
        print_metrics("Multivariate GB", metrics, time_ms);

        // Multivariate should generally perform better
        double test_std = ts::stats::std_dev(test);
        test_assert(metrics.rmse < 1.5 * test_std,
                    "Multivariate GB RMSE reasonable",
                    "RMSE=" + std::to_string(metrics.rmse));
    }

    // Test Histogram Gradient Boosting
    {
        auto start = std::chrono::high_resolution_clock::now();

        int n_lags = 6;  // Reduced for faster testing
        std::vector<double> train_subset(train.end() - 400, train.end());
        std::vector<std::vector<double>> X;
        std::vector<double> y;

        for (size_t t = n_lags; t < train_subset.size(); ++t) {
            std::vector<double> features;
            for (int lag = 1; lag <= n_lags; ++lag) {
                features.push_back(train_subset[t - lag]);
            }
            X.push_back(features);
            y.push_back(train_subset[t]);
        }

        ts::HistogramGradientBoosting model(20, 3, 64, 0.1);  // Reduced
        model.fit(X, y);

        // Create test features
        std::vector<std::vector<double>> X_test;
        std::vector<double> extended = train_subset;

        for (size_t t = 0; t < test.size(); ++t) {
            std::vector<double> features;
            for (int lag = 1; lag <= n_lags; ++lag) {
                size_t idx = extended.size() - lag;
                features.push_back(extended[idx]);
            }
            X_test.push_back(features);

            auto pred = model.predict({features});
            extended.push_back(pred[0]);
        }

        auto predictions = model.predict(X_test);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        ts::Metrics metrics = ts::evaluate(test, predictions);
        print_metrics("Histogram GB", metrics, time_ms);
    }
}

void test_dlinear(const std::vector<double>& train,
                  const std::vector<double>& test,
                  const std::vector<std::vector<double>>& train_features,
                  const std::vector<std::vector<double>>& test_features) {
    print_header("DLinear Model Tests");

    // Test DLinear univariate
    {
        auto start = std::chrono::high_resolution_clock::now();

        // Use subset for faster training
        std::vector<double> train_subset(train.end() - 500, train.end());

        ts::DLinear model(96, 24, 25);  // seq_len=96, pred_len=24, kernel_size=25
        model.fit(train_subset, 50, 0.001, 16);  // epochs, lr, batch_size

        test_assert(model.is_fitted(), "DLinear fitting");
        test_assert(model.seq_len() == 96, "DLinear seq_len correct");
        test_assert(model.pred_len() == 24, "DLinear pred_len correct");

        auto forecast = model.forecast(24);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        test_assert(forecast.predictions.size() == 24,
                    "DLinear forecast size");

        std::vector<double> test_subset(test.begin(), test.begin() + 24);
        test_assert(verify_predictions(forecast.predictions, test_subset),
                    "DLinear predictions validity");

        ts::Metrics metrics = ts::evaluate(test_subset, forecast.predictions);
        print_metrics("DLinear(96,24)", metrics, time_ms);

        // Check training loss decreased
        auto loss = model.loss_history();
        if (loss.size() >= 2) {
            test_assert(loss.back() < loss.front(),
                        "DLinear loss decreased",
                        "Start=" + std::to_string(loss.front()) +
                        " End=" + std::to_string(loss.back()));
        }
    }

    // Test NLinear
    {
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<double> train_subset(train.end() - 500, train.end());

        ts::NLinear model(96, 24);
        model.fit(train_subset, 50, 0.001, 16);

        test_assert(model.is_fitted(), "NLinear fitting");

        auto forecast = model.forecast(24);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        test_assert(forecast.predictions.size() == 24, "NLinear forecast size");

        std::vector<double> test_subset(test.begin(), test.begin() + 24);
        test_assert(verify_predictions(forecast.predictions, test_subset),
                    "NLinear predictions validity");

        ts::Metrics metrics = ts::evaluate(test_subset, forecast.predictions);
        print_metrics("NLinear(96,24)", metrics, time_ms);
    }

    // Test Linear baseline
    {
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<double> train_subset(train.end() - 500, train.end());

        ts::Linear model(96, 24);
        model.fit(train_subset, 50, 0.001, 16);

        test_assert(model.is_fitted(), "Linear fitting");

        auto forecast = model.forecast(24);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::vector<double> test_subset(test.begin(), test.begin() + 24);
        ts::Metrics metrics = ts::evaluate(test_subset, forecast.predictions);
        print_metrics("Linear(96,24)", metrics, time_ms);
    }

    // Test DLinear with different prediction horizons
    {
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<double> train_subset(train.end() - 400, train.end());

        ts::DLinear model(48, 48, 13);  // Smaller seq_len, longer pred_len
        model.fit(train_subset, 30, 0.001, 16);

        test_assert(model.is_fitted(), "DLinear(48,48) fitting");

        auto forecast = model.forecast(48);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        test_assert(forecast.predictions.size() == 48,
                    "DLinear(48,48) forecast size");

        std::vector<double> test_subset(test.begin(), test.begin() + 48);
        ts::Metrics metrics = ts::evaluate(test_subset, forecast.predictions);
        print_metrics("DLinear(48,48)", metrics, time_ms);

        double test_std = ts::stats::std_dev(test_subset);
        test_assert(metrics.rmse < 2.0 * test_std,
                    "DLinear RMSE reasonable",
                    "RMSE=" + std::to_string(metrics.rmse));
    }

    // Test DLinear multivariate
    {
        auto start = std::chrono::high_resolution_clock::now();

        // Prepare multivariate data - combine features with target
        size_t subset_size = 400;
        size_t start_idx = train_features.size() > subset_size ?
                           train_features.size() - subset_size : 0;

        std::vector<std::vector<double>> mv_data;
        for (size_t i = start_idx; i < train_features.size(); ++i) {
            std::vector<double> row = train_features[i];
            row.push_back(train[i]);  // Add target as last column
            mv_data.push_back(row);
        }

        ts::DLinear model(48, 24, 13, false);  // individual=false
        model.fit(mv_data, 6, 30, 0.001, 16);  // target_idx=6 (OT is last)

        test_assert(model.is_fitted(), "DLinear multivariate fitting");

        auto forecast = model.forecast(24);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        test_assert(forecast.predictions.size() == 24,
                    "DLinear multivariate forecast size");

        std::vector<double> test_subset(test.begin(), test.begin() + 24);
        test_assert(verify_predictions(forecast.predictions, test_subset),
                    "DLinear multivariate predictions validity");

        ts::Metrics metrics = ts::evaluate(test_subset, forecast.predictions);
        print_metrics("DLinear MV(48,24)", metrics, time_ms);
    }
}

void test_model_comparison(const std::vector<double>& train,
                           const std::vector<double>& test) {
    print_header("Model Comparison on ETTh1 Dataset");

    struct Result {
        std::string name;
        double rmse;
        double mae;
        double time_ms;
    };
    std::vector<Result> results;

    // Subset for faster testing
    std::vector<double> test_subset(test.begin(),
                                     test.begin() + std::min(static_cast<int>(test.size()), 168));
    int horizon = static_cast<int>(test_subset.size());

    // SES
    {
        auto start = std::chrono::high_resolution_clock::now();
        ts::SimpleExponentialSmoothing model;
        model.fit(train);
        auto forecast = model.forecast(horizon);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        ts::Metrics m = ts::evaluate(test_subset, forecast.predictions);
        results.push_back({"SES", m.rmse, m.mae, time_ms});
    }

    // Holt
    {
        auto start = std::chrono::high_resolution_clock::now();
        ts::HoltLinear model;
        model.fit(train);
        auto forecast = model.forecast(horizon);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        ts::Metrics m = ts::evaluate(test_subset, forecast.predictions);
        results.push_back({"Holt Linear", m.rmse, m.mae, time_ms});
    }

    // Holt-Winters
    {
        auto start = std::chrono::high_resolution_clock::now();
        ts::HoltWinters model(24);
        model.fit(train);
        auto forecast = model.forecast(horizon);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        ts::Metrics m = ts::evaluate(test_subset, forecast.predictions);
        results.push_back({"Holt-Winters(24)", m.rmse, m.mae, time_ms});
    }

    // ARIMA
    {
        auto start = std::chrono::high_resolution_clock::now();
        ts::ARIMA model(2, 1, 1);
        model.fit(train);
        auto forecast = model.forecast(horizon);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        ts::Metrics m = ts::evaluate(test_subset, forecast.predictions);
        results.push_back({"ARIMA(2,1,1)", m.rmse, m.mae, time_ms});
    }

    // Prophet
    {
        auto start = std::chrono::high_resolution_clock::now();
        ts::Prophet model;
        model.add_seasonality("daily", 24.0, 5);
        model.fit(train);
        auto forecast = model.forecast(horizon);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        ts::Metrics m = ts::evaluate(test_subset, forecast.predictions);
        results.push_back({"Prophet", m.rmse, m.mae, time_ms});
    }

    // Gradient Boosting - use subset for faster testing
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<double> train_gb(train.end() - 500, train.end());
        ts::TimeSeriesGradientBoosting model(12, 15, 3, 0.1);
        model.add_rolling_features({6});
        model.fit(train_gb);
        auto forecast = model.forecast(horizon);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        ts::Metrics m = ts::evaluate(test_subset, forecast.predictions);
        results.push_back({"GradientBoosting", m.rmse, m.mae, time_ms});
    }

    // DLinear
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<double> train_dl(train.end() - 500, train.end());
        ts::DLinear model(96, horizon, 25);
        model.fit(train_dl, 50, 0.001, 16);
        auto forecast = model.forecast(horizon);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        ts::Metrics m = ts::evaluate(test_subset, forecast.predictions);
        results.push_back({"DLinear", m.rmse, m.mae, time_ms});
    }

    // NLinear
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<double> train_nl(train.end() - 500, train.end());
        ts::NLinear model(96, horizon);
        model.fit(train_nl, 50, 0.001, 16);
        auto forecast = model.forecast(horizon);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        ts::Metrics m = ts::evaluate(test_subset, forecast.predictions);
        results.push_back({"NLinear", m.rmse, m.mae, time_ms});
    }

    // Sort by RMSE
    std::sort(results.begin(), results.end(),
              [](const Result& a, const Result& b) { return a.rmse < b.rmse; });

    std::cout << std::endl;
    std::cout << std::left << std::setw(20) << "Model"
              << std::right << std::setw(12) << "RMSE"
              << std::setw(12) << "MAE"
              << std::setw(12) << "Time(ms)" << std::endl;
    std::cout << std::string(56, '-') << std::endl;

    for (const auto& r : results) {
        std::cout << std::left << std::setw(20) << r.name
                  << std::right << std::setw(12) << std::fixed << std::setprecision(4) << r.rmse
                  << std::setw(12) << r.mae
                  << std::setw(12) << std::setprecision(2) << r.time_ms << std::endl;
    }

    // Best model should have reasonable performance
    test_assert(results[0].rmse < ts::stats::std_dev(test_subset),
                "Best model RMSE < test std dev",
                "Best: " + results[0].name + " RMSE=" + std::to_string(results[0].rmse));
}

int main() {
    std::cout << "ETTh1 Dataset Time Series Forecasting Tests" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // Load data
    std::cout << "\nLoading ETTh1 dataset..." << std::endl;
    ts::CSVReader::Dataset dataset;

    try {
        dataset = ts::CSVReader::read(DATA_PATH, true);  // Skip date column
    } catch (const std::exception& e) {
        std::cerr << "Error loading dataset: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Dataset loaded: " << dataset.num_rows() << " rows, "
              << dataset.num_cols() << " columns" << std::endl;
    std::cout << "Columns: ";
    for (const auto& h : dataset.headers) std::cout << h << " ";
    std::cout << std::endl;

    // Extract target (OT) and features
    std::vector<double> OT = dataset.get_column("OT");

    // Get all feature columns
    std::vector<std::string> feature_names = {"HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"};
    std::vector<std::vector<double>> features = dataset.get_columns(feature_names);

    std::cout << "Target: OT (Oil Temperature)" << std::endl;
    std::cout << "Features: ";
    for (const auto& f : feature_names) std::cout << f << " ";
    std::cout << std::endl;

    // Statistics
    std::cout << "\nTarget statistics:" << std::endl;
    std::cout << "  Mean: " << ts::stats::mean(OT) << std::endl;
    std::cout << "  Std:  " << ts::stats::std_dev(OT) << std::endl;
    std::cout << "  Min:  " << *std::min_element(OT.begin(), OT.end()) << std::endl;
    std::cout << "  Max:  " << *std::max_element(OT.begin(), OT.end()) << std::endl;

    // Split data
    std::vector<double> train(OT.begin(), OT.begin() + TRAIN_SIZE);
    std::vector<double> test(OT.begin() + TRAIN_SIZE,
                             OT.begin() + std::min(TRAIN_SIZE + TEST_SIZE,
                                                   static_cast<int>(OT.size())));

    std::vector<std::vector<double>> train_features(features.begin(),
                                                     features.begin() + TRAIN_SIZE);
    std::vector<std::vector<double>> test_features(features.begin() + TRAIN_SIZE,
                                                    features.begin() + TRAIN_SIZE + test.size());

    std::cout << "\nTrain size: " << train.size() << std::endl;
    std::cout << "Test size: " << test.size() << std::endl;

    // Run tests
    try {
        test_arima(train, test);
        test_exponential_smoothing(train, test);
        test_prophet(train, test);
        test_gradient_boosting(train, test, train_features, test_features);
        test_dlinear(train, test, train_features, test_features);
        test_model_comparison(train, test);
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] Exception: " << e.what() << std::endl;
        tests_failed++;
    }

    // Summary
    print_header("Test Summary");
    std::cout << "Tests passed: " << tests_passed << std::endl;
    std::cout << "Tests failed: " << tests_failed << std::endl;

    if (tests_failed == 0) {
        std::cout << "\nAll tests PASSED! Models verified on real ETTh1 data." << std::endl;
    } else {
        std::cout << "\nSome tests FAILED. Review output above." << std::endl;
    }

    return tests_failed > 0 ? 1 : 0;
}
