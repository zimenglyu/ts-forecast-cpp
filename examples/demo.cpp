#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "ts_forecast.hpp"

void print_header(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

int main() {
    std::cout << "Time Series Forecasting Library Demo" << std::endl;
    std::cout << "=====================================\n" << std::endl;

    // Generate sample data with multiple components
    std::vector<double> data;
    int n = 150;

    std::cout << "Generating synthetic time series data..." << std::endl;
    std::cout << "Components: linear trend + weekly seasonality + noise" << std::endl;

    for (int i = 0; i < n; ++i) {
        double trend = 100.0 + 0.5 * i;
        double weekly = 15.0 * std::sin(2 * M_PI * i / 7.0);
        double noise = 5.0 * (std::rand() % 100 - 50) / 50.0;
        data.push_back(trend + weekly + noise);
    }

    // Split into train/test
    auto [train, test] = ts::train_test_split(data, 0.2);
    int test_size = static_cast<int>(test.size());

    std::cout << "Training samples: " << train.size() << std::endl;
    std::cout << "Test samples: " << test.size() << std::endl;

    // Store results for comparison
    struct ModelResult {
        std::string name;
        double rmse;
        double mape;
        std::vector<double> predictions;
    };
    std::vector<ModelResult> results;

    // 1. ARIMA
    print_header("ARIMA(2,1,1)");
    try {
        ts::ARIMA arima(2, 1, 1);
        arima.fit(train);

        ts::ForecastResult forecast = arima.forecast(test_size);
        ts::Metrics metrics = ts::evaluate(test, forecast.predictions);

        std::cout << "RMSE: " << std::fixed << std::setprecision(4) << metrics.rmse << std::endl;
        std::cout << "MAPE: " << metrics.mape << "%" << std::endl;

        results.push_back({"ARIMA(2,1,1)", metrics.rmse, metrics.mape, forecast.predictions});
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }

    // 2. Auto ARIMA
    print_header("Auto ARIMA");
    try {
        ts::AutoARIMA auto_arima(3, 2, 3);
        auto_arima.fit(train);

        std::cout << "Selected: ARIMA(" << auto_arima.selected_p()
                  << "," << auto_arima.selected_d()
                  << "," << auto_arima.selected_q() << ")" << std::endl;

        ts::ForecastResult forecast = auto_arima.forecast(test_size);
        ts::Metrics metrics = ts::evaluate(test, forecast.predictions);

        std::cout << "RMSE: " << metrics.rmse << std::endl;
        std::cout << "MAPE: " << metrics.mape << "%" << std::endl;

        results.push_back({"Auto ARIMA", metrics.rmse, metrics.mape, forecast.predictions});
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }

    // 3. Simple Exponential Smoothing
    print_header("Simple Exponential Smoothing");
    try {
        ts::SimpleExponentialSmoothing ses;
        ses.fit(train);

        std::cout << "Optimized alpha: " << ses.alpha() << std::endl;

        ts::ForecastResult forecast = ses.forecast(test_size);
        ts::Metrics metrics = ts::evaluate(test, forecast.predictions);

        std::cout << "RMSE: " << metrics.rmse << std::endl;
        std::cout << "MAPE: " << metrics.mape << "%" << std::endl;

        results.push_back({"SES", metrics.rmse, metrics.mape, forecast.predictions});
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }

    // 4. Holt's Linear Trend
    print_header("Holt's Linear Trend");
    try {
        ts::HoltLinear holt;
        holt.fit(train);

        std::cout << "Alpha: " << holt.alpha() << ", Beta: " << holt.beta() << std::endl;

        ts::ForecastResult forecast = holt.forecast(test_size);
        ts::Metrics metrics = ts::evaluate(test, forecast.predictions);

        std::cout << "RMSE: " << metrics.rmse << std::endl;
        std::cout << "MAPE: " << metrics.mape << "%" << std::endl;

        results.push_back({"Holt Linear", metrics.rmse, metrics.mape, forecast.predictions});
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }

    // 5. Holt-Winters with weekly seasonality
    print_header("Holt-Winters (period=7)");
    try {
        ts::HoltWinters hw(7, ts::HoltWinters::SeasonalType::ADDITIVE);
        hw.fit(train);

        std::cout << "Alpha: " << hw.alpha() << ", Beta: " << hw.beta()
                  << ", Gamma: " << hw.gamma() << std::endl;

        ts::ForecastResult forecast = hw.forecast(test_size);
        ts::Metrics metrics = ts::evaluate(test, forecast.predictions);

        std::cout << "RMSE: " << metrics.rmse << std::endl;
        std::cout << "MAPE: " << metrics.mape << "%" << std::endl;

        results.push_back({"Holt-Winters", metrics.rmse, metrics.mape, forecast.predictions});
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }

    // 6. Prophet
    print_header("Prophet-like Model");
    try {
        ts::Prophet prophet(ts::Prophet::GrowthType::LINEAR, false, true, false);
        prophet.fit(train);

        ts::ForecastResult forecast = prophet.forecast(test_size);
        ts::Metrics metrics = ts::evaluate(test, forecast.predictions);

        std::cout << "RMSE: " << metrics.rmse << std::endl;
        std::cout << "MAPE: " << metrics.mape << "%" << std::endl;

        results.push_back({"Prophet", metrics.rmse, metrics.mape, forecast.predictions});
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }

    // 7. Gradient Boosting
    print_header("Gradient Boosting");
    try {
        ts::TimeSeriesGradientBoosting gb(7, 50, 4, 0.1);
        gb.add_rolling_features({3, 7});
        gb.fit(train);

        ts::ForecastResult forecast = gb.forecast(test_size);
        ts::Metrics metrics = ts::evaluate(test, forecast.predictions);

        std::cout << "RMSE: " << metrics.rmse << std::endl;
        std::cout << "MAPE: " << metrics.mape << "%" << std::endl;

        results.push_back({"Gradient Boosting", metrics.rmse, metrics.mape, forecast.predictions});
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }

    // 8. Neural Prophet
    print_header("NeuralProphet-like Model");
    try {
        ts::NeuralProphet np(14, 1, false, true);
        np.fit(train, 100, 0.01);

        ts::ForecastResult forecast = np.forecast(test_size);
        ts::Metrics metrics = ts::evaluate(test, forecast.predictions);

        std::cout << "RMSE: " << metrics.rmse << std::endl;
        std::cout << "MAPE: " << metrics.mape << "%" << std::endl;

        results.push_back({"NeuralProphet", metrics.rmse, metrics.mape, forecast.predictions});
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }

    // Summary comparison
    print_header("Model Comparison Summary");
    std::cout << std::left << std::setw(20) << "Model"
              << std::right << std::setw(12) << "RMSE"
              << std::setw(12) << "MAPE (%)" << std::endl;
    std::cout << std::string(44, '-') << std::endl;

    // Sort by RMSE
    std::sort(results.begin(), results.end(),
              [](const ModelResult& a, const ModelResult& b) {
                  return a.rmse < b.rmse;
              });

    for (const auto& result : results) {
        std::cout << std::left << std::setw(20) << result.name
                  << std::right << std::setw(12) << std::fixed << std::setprecision(4) << result.rmse
                  << std::setw(12) << std::setprecision(2) << result.mape << std::endl;
    }

    // Show sample predictions from best model
    if (!results.empty()) {
        std::cout << "\nSample predictions from best model (" << results[0].name << "):" << std::endl;
        std::cout << std::setw(8) << "Step" << std::setw(12) << "Actual" << std::setw(12) << "Predicted" << std::endl;
        for (int i = 0; i < std::min(10, test_size); ++i) {
            std::cout << std::setw(8) << (i + 1)
                      << std::setw(12) << std::fixed << std::setprecision(2) << test[i]
                      << std::setw(12) << results[0].predictions[i] << std::endl;
        }
    }

    std::cout << "\nDemo completed successfully!" << std::endl;

    return 0;
}
