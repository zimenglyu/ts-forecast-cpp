#include <iostream>
#include <vector>
#include <cmath>
#include "ts_forecast.hpp"

int main() {
    std::cout << "=== Exponential Smoothing Examples ===" << std::endl;

    // Generate data with trend and seasonality
    std::vector<double> data;
    int period = 12;  // Monthly data with yearly seasonality

    for (int i = 0; i < 48; ++i) {  // 4 years of monthly data
        double trend = 100.0 + 2.0 * i;
        double seasonal = 20.0 * std::sin(2 * M_PI * i / period);
        double noise = (std::rand() % 10 - 5);
        data.push_back(trend + seasonal + noise);
    }

    std::cout << "Generated 48 monthly data points with trend and seasonality" << std::endl;

    // 1. Simple Exponential Smoothing
    std::cout << "\n--- Simple Exponential Smoothing ---" << std::endl;
    ts::SimpleExponentialSmoothing ses;
    ses.fit(data);
    std::cout << "Optimized alpha: " << ses.alpha() << std::endl;

    ts::ForecastResult ses_forecast = ses.forecast(6);
    std::cout << "Forecasts: ";
    for (double pred : ses_forecast.predictions) {
        std::cout << pred << " ";
    }
    std::cout << std::endl;

    // 2. Holt's Linear Trend
    std::cout << "\n--- Holt's Linear Trend ---" << std::endl;
    ts::HoltLinear holt;
    holt.fit(data);
    std::cout << "Alpha: " << holt.alpha() << ", Beta: " << holt.beta() << std::endl;

    ts::ForecastResult holt_forecast = holt.forecast(6);
    std::cout << "Forecasts: ";
    for (double pred : holt_forecast.predictions) {
        std::cout << pred << " ";
    }
    std::cout << std::endl;

    // 3. Holt-Winters (Additive Seasonality)
    std::cout << "\n--- Holt-Winters (Additive) ---" << std::endl;
    ts::HoltWinters hw_add(period, ts::HoltWinters::SeasonalType::ADDITIVE);
    hw_add.fit(data);
    std::cout << "Alpha: " << hw_add.alpha()
              << ", Beta: " << hw_add.beta()
              << ", Gamma: " << hw_add.gamma() << std::endl;

    ts::ForecastResult hw_add_forecast = hw_add.forecast(12);
    std::cout << "12-month forecast:" << std::endl;
    for (size_t i = 0; i < hw_add_forecast.predictions.size(); ++i) {
        std::cout << "  Month " << (i + 1) << ": "
                  << hw_add_forecast.predictions[i]
                  << " [" << hw_add_forecast.lower_bound[i]
                  << ", " << hw_add_forecast.upper_bound[i] << "]"
                  << std::endl;
    }

    // 4. Holt-Winters (Multiplicative Seasonality)
    std::cout << "\n--- Holt-Winters (Multiplicative) ---" << std::endl;

    // Generate multiplicative seasonal data
    std::vector<double> mult_data;
    for (int i = 0; i < 48; ++i) {
        double trend = 100.0 + 2.0 * i;
        double seasonal = 1.0 + 0.2 * std::sin(2 * M_PI * i / period);
        double noise = 1.0 + 0.02 * (std::rand() % 10 - 5);
        mult_data.push_back(trend * seasonal * noise);
    }

    ts::HoltWinters hw_mult(period, ts::HoltWinters::SeasonalType::MULTIPLICATIVE);
    hw_mult.fit(mult_data);
    std::cout << "Alpha: " << hw_mult.alpha()
              << ", Beta: " << hw_mult.beta()
              << ", Gamma: " << hw_mult.gamma() << std::endl;

    ts::ForecastResult hw_mult_forecast = hw_mult.forecast(6);
    std::cout << "Forecasts: ";
    for (double pred : hw_mult_forecast.predictions) {
        std::cout << pred << " ";
    }
    std::cout << std::endl;

    // 5. Damped Trend
    std::cout << "\n--- Holt Linear with Damped Trend ---" << std::endl;
    ts::HoltLinear holt_damped(-1.0, -1.0, true, 0.95);
    holt_damped.fit(data);
    std::cout << "Alpha: " << holt_damped.alpha()
              << ", Beta: " << holt_damped.beta() << std::endl;

    ts::ForecastResult damped_forecast = holt_damped.forecast(12);
    std::cout << "Damped vs Regular forecasts:" << std::endl;
    for (size_t i = 0; i < damped_forecast.predictions.size(); ++i) {
        std::cout << "  Step " << (i + 1) << ": Damped="
                  << damped_forecast.predictions[i]
                  << ", Regular=" << holt_forecast.predictions[std::min(i, holt_forecast.predictions.size()-1)]
                  << std::endl;
    }

    // Evaluate Holt-Winters on holdout
    std::cout << "\n--- Model Evaluation ---" << std::endl;
    auto [train, test] = ts::train_test_split(data, 0.25);

    ts::HoltWinters hw_eval(period, ts::HoltWinters::SeasonalType::ADDITIVE);
    hw_eval.fit(train);

    ts::ForecastResult eval_forecast = hw_eval.forecast(static_cast<int>(test.size()));
    ts::Metrics metrics = ts::evaluate(test, eval_forecast.predictions);
    std::cout << "Out-of-sample evaluation:" << std::endl;
    std::cout << metrics.to_string() << std::endl;

    return 0;
}
